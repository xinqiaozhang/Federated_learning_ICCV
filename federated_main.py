#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import random
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from torchvision import transforms

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, LeNet5, SimpleDLA, ResNet18, MnistNet
from utils import get_dataset, average_weights, exp_details
from train_eval_utils import test, adjust_learning_rate
from torchpack.mtpack.models.vision.resnet import resnet20
from torchvision import datasets, transforms
import pdb

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
    
def apply_attack_EIFFeL( weights, mal_num = 2, mode = 1, scale = 10, partial_att = 0, num_std = 1.0):
    # mode -- 1: Sign Flip attack, 2:Scaling attack 3:Non-omniscient attack 
    if mode ==1: # 1: Sign Flip attack
        updated_weights = copy.deepcopy(weights)
        # print("weights before",weights[0]['linear.weight'][0,:2])
        if partial_att == 1:
            for i in range(mal_num):
                for key in weights[0].keys():
                    # pdb.set_trace()
                    for x in range(0,weights[0][key].shape[0],3):
                        updated_weights[i][key][x] = - weights[i][key][x] * scale
        elif partial_att < 100 and partial_att>0:
            for i in range(mal_num):
                count = 0
                for key in weights[0].keys():
                    # pdb.set_trace()
                    
                    for x in range(0,weights[0][key].shape[0]):
                        count = count +1
                        if count < int(partial_att/10) +1:
                            updated_weights[i][key][x] = - weights[i][key][x] * scale
                        else:
                            if count == 10:
                                count = 0
                            continue
        else:
            gap = len(weights)/ mal_num  - 1
            for i in range(mal_num):
                calib_index = int(gap) * i
                for key in weights[0].keys():
                    updated_weights[calib_index][key] = - weights[calib_index][key] * scale
    elif  mode ==2: #2:Scaling attack 
        updated_weights = copy.deepcopy(weights)
        if partial_att:
            for i in range(mal_num):
                count = 0
                for key in weights[0].keys():
                    for x in range(0,weights[0][key].shape[0]):
                        count = count +1
                        if count < int(partial_att/10) +1:
                            updated_weights[i][key][x] = weights[i][key][x] * scale
                        else:
                            if count == 10:
                                count = 0
                            continue          
        else:
            gap = len(weights)/ mal_num  - 1
            for i in range(mal_num):
                calib_index = int(gap) * i
                for key in weights[0].keys():
                    updated_weights[calib_index][key] =  weights[calib_index][key] * scale
    elif mode == 3: #3:Non-omniscient attack 
        weights_mean, weights_stdev = average_weights(weights[:mal_num], mal_num = mal_num)
        updated_weights = copy.deepcopy(weights)
        gap = len(weights)/ mal_num  - 1
        for i in range(mal_num):
            calib_index = int(gap) * i
            count = 0
            for index, key in enumerate(weights[0].keys()):
                for x in range(0,weights_mean[key].shape[0]):
                    count = count +1
                    if count < int(partial_att/10) +1:
                        updated_weights[calib_index][key][x] = weights_mean[key][x] - num_std * weights_stdev
                    else:
                        if count == 10:
                            count = 0
                        continue
    return updated_weights

# Pending to implenment
# def apply_attack_FoE(weights, benigh_weights_mean, mal_num = 1, foe_rate = 10):
#         """
#         Simulating the attack method in: https://arxiv.org/abs/1903.03936
#         Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation
#         """
#         FACTOR = foe_rate #make it huge
#         # pdb.set_trace()
#         # mu = benigh_weights_mean
#         # mu = np.mean(benigh_weights_buffer, axis=0)
#         # mal_grad = -FACTOR * mu
#         updated_weights = copy.deepcopy(weights)
#         # pdb.set_trace()
#         for i in range(mal_num):
#             for key in weights[0].keys():
#                 # pdb.set_trace()
#                 updated_weights[i][key] = -FACTOR * benigh_weights_mean[key]
#         return weights

def average_weights(w ,mal_num =2, mode = 0):
    """
    Returns the average of the weights.
    """
    # mal_num = 2
    # pdb.set_trace()
    if len(w) == 1:
        w_avg = copy.deepcopy(w)
        return w_avg, 0
    w_avg = copy.deepcopy(w[0])
    w_std_final = np.zeros((mal_num))
    if mode == 0:
        total_std = []
        for index, tem_w in enumerate(w):
            # if skip_first:
            #     tem_std = np.concatenate([tem_w[usr].view(-1).cpu().numpy().flatten() for usr in tem_w.keys() for tem_w in w])
            #     w_std = np.concatenate((tem_std.flatten(),w_std.flatten()))
            tem_std = np.concatenate([tem_w[usr].view(-1).cpu().numpy().flatten() for usr in tem_w.keys()])
            # pdb.set_trace()
        total_std.append(tem_std)
        w_std_final = np.var(total_std) ** 0.5

         
  
    for key in w_avg.keys():
        for i in range(1, len(w)):  
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
        # pdb.set_trace()
    return w_avg, w_std_final

def flatten_params(params):
    return np.concatenate([params[i].data.cpu().numpy().flatten() for i in params])

def median_mean_k_defense(weights, group_size = 7, rate = 100, check_indeces = 0, mode = 0): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users_count = len(weights)
    users_grads = torch.empty((users_count, flatten_params(weights[0]).shape[0]), dtype=torch.float32).to(device)
    for i in range(users_count):
        users_grads[i] = torch.tensor(flatten_params(weights[i]),dtype=torch.float32)
   
    good_index = []
    if mode != 100:
        bad_index = []
        for check_rounds in range(12):
            i = torch.randint(0, users_grads.shape[1],(1,))
            param_across_users = users_grads.T[i][0].view(-1)
            mean_vector = torch.zeros(group_size).to(device)
            num_each_group = int(len(param_across_users) / group_size)
            if num_each_group * group_size == len(param_across_users):
                for ii in range(group_size):
                    mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])    
            else:
                for ii in range(group_size-1):
                    mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])
                mean_vector[group_size-1] = torch.mean(param_across_users[(group_size-1)*num_each_group:])
                
            med = torch.median(mean_vector)
            x = torch.abs(mean_vector - med)**2
            std = torch.sqrt(torch.mean(x))
            c = (param_across_users - med).abs() > 1*std * rate/100
            indices = c.nonzero()
            if indices.shape[0] == 0 or indices.shape[0] == users_count:
                continue
            bad_index.append(indices.cpu().tolist())
        flat_bad_index = [item for sublist in bad_index for item in sublist]
        flat_bad_index = [item for sublist in flat_bad_index for item in sublist]
        for i in range(users_count):
            if i not in flat_bad_index:
            
                good_index.append(i)

        good_index_length = len(good_index)
    else:
        i = torch.randint(0, users_grads.shape[1],(1,))
        param_across_users = users_grads.T[i][0].view(-1)
        mean_vector = torch.zeros(group_size).to(device)
        num_each_group = int(len(param_across_users) / group_size)
        if num_each_group * group_size == len(param_across_users):
            for ii in range(group_size):
                mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])    
        else:
            for ii in range(group_size-1):
                mean_vector[ii] = torch.mean(param_across_users[ii*num_each_group:(ii+1)*num_each_group])
            mean_vector[group_size-1] = torch.mean(param_across_users[(group_size-1)*num_each_group:])


        med = torch.median(mean_vector)
        x = torch.abs(mean_vector - med)**2
        std = torch.sqrt(torch.mean(x))
        
        c = (param_across_users - med).abs() <= 1*std * rate/100
        good_index = c.nonzero()
        good_index_length = good_index.shape[0]
    update_rate = rate
    
    
    if good_index_length < int(users_count * 0.1): 
        w_avg = 0
    elif good_index_length > int(users_count*0.85):
        w_avg, _ = average_weights(weights, mode = 30)
    else:
        # print("Training----: good_index length is",len(good_index))
        w_pick = [weights[k] for k in good_index]
        w_avg = copy.deepcopy(w_pick[0])
        
        for key in w_avg.keys():
            for i in range(1, len(w_pick)): 
                w_avg[key] += w_pick[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w_pick))

    # pdb.set_trace()
    return w_avg, update_rate, good_index


# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

if __name__ == '__main__':
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)


    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if hasattr(args, 'gpu_id'):
        torch.cuda.set_device(args.gpu_id)
        print('using GPU device ', args.gpu_id)
    device = 'cuda' if args.gpu is not None else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    trainloader_global = DataLoader(train_dataset, batch_size=args.batch_size*args.num_batches_per_step, shuffle=True)
    # trainloader_global = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.batch_size*args.num_batches_per_step, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # train_iterator = iter(cycle(trainloader_global))
    
    # BUILD MODEL
    args.num_classes = 100 if args.dataset=='cifar100' else 10
    if args.model == 'lenet':
        global_model = LeNet5(args=args)
    elif args.model == 'mnist':
        global_model = MnistNet()
    elif args.model == 'resnet20':
        global_model = resnet20(num_classes=args.num_classes)
    elif args.model == 'cnncifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'cnnfmnist':
            global_model = CNNFashion_Mnist(args=args)
    
    # elif args.model == 'mnist':
    #     global_model = CNNMnist(args=args)
    else:
        exit('Error: unrecognized model')

    init_ckpt = '{}_{}_init.ckpt'.format(args.dataset, args.model)
    if os.path.exists(init_ckpt):
        print('Loading initialization')
        state_dict = torch.load(init_ckpt)
        global_model.load_state_dict(state_dict)
    else:
        print('Saving initialization')
        torch.save(global_model.state_dict(), init_ckpt)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    args.lr *= args.num_batches_per_step * args.num_users
    
    print("learning rate is", args.lr)
    global_optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    global_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(global_optimizer, T_max=args.epochs-args.warmup_lr_epochs)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_accuracy = []
    args.mal_ratio = 0.20
    mal_user_num = int(args.mal_ratio * args.num_users) 
    FoE_start_epoch = 10
    median_mean_rate = args.meank_rate

    
    csv_dict = {'0_epoch': [], '1_train_loss': [],'2_lr': [], '3_test_acc': [], '4_test_loss': []}
    df = pd.DataFrame.from_dict(csv_dict, orient='columns')
    path_to_logs = './logs/NonAda_{}_{}_{}_{}_of_{}_att_{}_attr_{}_pass_mmr_{}_ratio_{}_def_{}_lr_{}'.format(args.dataset, args.model, args.epochs, mal_user_num,args.num_users,  args.attack_method, args.attack_rate,median_mean_rate, args.en_partial_att,args.en_defence,args.lr)
    if not os.path.exists(path_to_logs):
        os.makedirs(path_to_logs)

    print("model parameters are:",sum(param.numel() for param in global_model.parameters()))
    local_models = []
    for c in range(args.num_users):
        local_model = LocalUpdate(args=args, model=copy.deepcopy(global_model), dataset=train_dataset,
                                          idxs=user_groups[c], logger=logger)
        local_models.append(local_model)

    num_steps_per_epoch = len(trainloader_global)
    best_test_acc = 0
    update_rate = args.meank_rate
    for epoch in range(args.epochs):
        print('epoch:', epoch)
        global_model.train()
        global_optimizer.zero_grad()
        idxs_users = range(args.num_users)
        step_size = args.num_batches_per_step * args.batch_size
        _r_num_batches_per_step = 1.0 / args.num_batches_per_step
    

        for step, (inputs, targets, indices) in enumerate(tqdm(trainloader_global, desc='train', ncols=0, disable=False)):
            adjust_learning_rate(global_scheduler, epoch, step, num_steps_per_epoch,
                         warmup_lr_epochs=args.warmup_lr_epochs, 
                         schedule_lr_per_epoch=args.schedule_lr_per_epoch, 
                         size=args.num_users)
            inputs = inputs.to(device, non_blocking=True)
            
            targets = targets.to(device, non_blocking=True)
            loss = torch.tensor([0.0])
            benigh_local_grads, local_grads, local_weights = [], [], []
            
            # benigh_local_grads_buffer = []
            for idx in idxs_users:
                _, indices_for_this_user, _ = np.intersect1d(indices, list(user_groups[idx]), return_indices=True)
                # pdb.set_trace()
                user_inputs = inputs[indices_for_this_user]
                user_targets = targets[indices_for_this_user]
                local_model = local_models[idx]
                
                local_model.optimizer.zero_grad()
                local_model.model.train()
                for b in range(0, step_size, args.batch_size):
                    _inputs = user_inputs[b:b+args.batch_size]
                    _targets = user_targets[b:b+args.batch_size]
                    if user_inputs == []:
                        pdb.set_trace()
                    _outputs = local_model.model(_inputs)
                    _loss = local_model.criterion(_outputs, _targets)
                    _loss.mul_(_r_num_batches_per_step)
                    _loss.backward()
                    loss += _loss.item()
                
                #------------------ accumulate local gradients
                grad = {k:p.grad for k, p in local_model.model.named_parameters()}
                # pdb.set_trace()
                local_grads.append(copy.deepcopy(grad))
                if idx >= mal_user_num:
                    benigh_local_grads.append(copy.deepcopy(grad))

            if args.attack_method == 1:        
                    updated_local_grads = apply_attack_EIFFeL(local_grads, mode =1 , scale= 10, partial_att = args.en_partial_att)
            elif args.attack_method == 2:        
                    updated_local_grads = apply_attack_EIFFeL(local_grads, mode =2 , scale= 10, partial_att = args.en_partial_att) 
            elif args.attack_method == 3:        
                    updated_local_grads = apply_attack_EIFFeL(local_grads, mode =3 , scale= 10 ,mal_num = mal_user_num, partial_att = args.en_partial_att, num_std = args.attack_rate ) 
            else:
                updated_local_grads = local_grads


            #------------------ apply defense
            good_index = []
            # if args.en_defence and epoch >1:
            if args.en_defence:
                    global_grads,update_rate, good_index = median_mean_k(updated_local_grads, group_size = args.group_size, rate = update_rate, check_indeces= 0, mode = args.en_partial_att)
            else:
                global_grads,_ = average_weights(updated_local_grads, mal_num = mal_user_num, mode = 1)
 
            #------------------ apply accumulated local gradients
            if global_grads == 0:
                continue
            
            for k, p in global_model.named_parameters():
                if args.num_users == 1:
                # pdb.set_trace()
                    p.grad = global_grads[0][k]
                else:
                    p.grad = global_grads[k]
            global_optimizer.step()
            #------------------ apply accumulated local weights
            loss_avg = loss / len(idxs_users)
            # loss_avg = loss

            state_dict = global_model.state_dict()
            for idx in idxs_users:
                for name, p in local_models[idx].model.named_parameters():
                    p.data = state_dict[name].data
        if global_grads ==0:
            continue            
        test_acc, test_loss = test_inference(args, local_models[0].model, testloader)
        print("Test Accuracy: {:.2f}%".format(100*test_acc))
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
        else:
            print("##acc decresed, good_index is",good_index)
        print("Best Test Accuracy: {:.2f}%\n".format(100*best_test_acc))
        

        csv_dict['0_epoch'] = epoch
        csv_dict['1_train_loss'] = loss_avg.item()
        csv_dict['2_lr'] = local_model.optimizer.param_groups[0]['lr']
        csv_dict['3_test_acc'] = test_acc
        csv_dict['4_test_loss'] = test_loss

        new_row = csv_dict.values()
        df = df.append(csv_dict, ignore_index=True, sort=True)
        df.to_csv(os.path.join(path_to_logs, 'federated_baseline_logs_U[{}]_biglr.csv'.format(args.num_users)))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, testloader)
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*best_test_acc))

    # Saving the objects train_loss and train_accuracy:
    if not os.path.exists('./save/objects/'):
        os.makedirs('./save/objects/')
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
