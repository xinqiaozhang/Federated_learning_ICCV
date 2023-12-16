#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

#0809: bound set to 70%-80% with adaptive adjustment, reduce the adjust rate helps get better result.
# 28 groups vs 7 groups will get lower acc 77.1% vs 91% for 200 users
# High attack scale(4) will not improve defense acc 
from collections import Counter
import os
import copy
import time
import pickle
import random
import numpy as np
from tqdm import tqdm
import pandas as pd
import datetime
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
import wandb
from src.krum import Krum
from src.rfa import RFA
from src.coordinatewise_median import CM
from src.clipping import Clipping
def cycle(iterable):
    while True:
        for x in iterable:
            yield x


enable_wandb = 0
if enable_wandb:
    wandb.init(
            # set the wandb project where this run will be logged
            project="fl_ext_1205",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": 0.02,
            "architecture": "baby-a3c",
            "dataset": "Pong",
            "epochs": 100000,
            }
        )

def apply_attack_EIFFeL( weights, mal_num_list, mode = 1, scale = 10, partial_att = 100, num_std = 1.0):
    # mode -- 1: Sign Flip attack, 2:Scaling attack 3:Non-omniscient attack 
    mal_num = len(mal_num_list)
    if mode ==1: # 1: Sign Flip attack
        updated_weights = copy.deepcopy(weights)
        # print("weights before",weights[0]['linear.weight'][0,:2])
        if partial_att == 100:
            for i in mal_num_list:
                for key in weights[0].keys():
                    # pdb.set_trace()
                    for x in range(0,weights[0][key].shape[0]):
                        updated_weights[i][key][x] = - weights[i][key][x] * scale
        elif partial_att < 100 and partial_att>0:
            for i in mal_num_list:
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
            for i in mal_num_list:
                calib_index = int(gap) * i
                for key in weights[0].keys():
                    updated_weights[calib_index][key] = - weights[calib_index][key] * scale
    elif  mode ==2: #2: Scaling attack 
        updated_weights = copy.deepcopy(weights)
        if partial_att != 100:
            for i in mal_num_list:
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
            # gap = len(weights)/ mal_num  - 1
            for i in mal_num_list:
                # calib_index = int(gap) * i
                for key in weights[0].keys():
                    updated_weights[i][key] =  weights[i][key] * scale
    elif mode == 3: #3: Non-omniscient attack 
        weights_mean, weights_stdev = average_weights(weights[:mal_num], mal_num = mal_num)
        updated_weights = copy.deepcopy(weights)
        gap = len(weights)/ mal_num  - 1
        for i in mal_num_list:
            calib_index = int(gap) * i
            count = 0
            for index, key in enumerate(weights[0].keys()):
                for x in range(0,weights_mean[key].shape[0]):
                    
                    if partial_att == 100:
                        updated_weights[i][key][x] = weights_mean[key][x] - scale * num_std * weights_stdev
                    else:
                        count = count +1
                        if count < int(partial_att/10) +1:
                            # print("updating weights, count is:",count)
                            pdb.set_trace()
                            updated_weights[i][key][x] = weights_mean[key][x] - scale * num_std * weights_stdev
                        else:
                            if count == 10:
                                count = 0
                            continue
    if mode ==5: # 4: Random weights
        updated_weights = copy.deepcopy(weights)
        if partial_att == 100:
            for i in mal_num_list:
                for key in weights[0].keys():
                    # pdb.set_trace()
                    for x in range(0,weights[0][key].shape[0]):
                        # pdb.set_trace()
                        updated_weights[i][key][x] = torch.rand(updated_weights[i][key][x].shape)/scale
    
    return updated_weights

def apply_attack_ext( weights, mal_num_list, mode = 1, scale = 10, partial_att = 100, num_std = 1.0):
    # mode -- 4: Label-flipping attack, 5: Random weights
    mal_num = len(mal_num_list)
    if mode ==4: # 4: Label-flipping attack, Random weights
        updated_weights = copy.deepcopy(weights)
        if partial_att == 100:
            for i in mal_num_list:
                for key in weights[0].keys():
                    # pdb.set_trace()
                    for x in range(0,weights[0][key].shape[0]):
                        updated_weights[i][key][x] = torch.rand(updated_weights[i][key][x].shape)
                        
        # elif partial_att < 100 and partial_att>0:
        #     for i in mal_num_list:
        #         count = 0
        #         for key in weights[0].keys():
        #             # pdb.set_trace()
                    
        #             for x in range(0,weights[0][key].shape[0]):
        #                 count = count +1
        #                 if count < int(partial_att/10) +1:
        #                     updated_weights[i][key][x] = - weights[i][key][x] * scale
        #                 else:
        #                     if count == 10:
        #                         count = 0
        #                     continue
   
    return updated_weights


def apply_attack_BUCKETING( weights, benigh_local_grads, mal_num_list, mode = 1, scale = 10, partial_att = 100, num_std = 1.0):
    mal_num = len(mal_num_list)
    if mode == 1: #bitflipping
        for i in mal_num_list:
                for key in weights[0].keys():
                    for x in range(0,weights[0][key].shape[0]):
                            updated_weights[i][key][x] = - weights[i][key][x]
    elif mode == 2: #labelflipping
        return weights

    # elif mode == 3: #Mimic benigh_local_grads
        
    #     # curr_good_ranks, curr_good_grads = self._get_good_grads()
    #     total_user = len(benigh_local_grads) + len(mal_num_list)
    #     print("total user is", total_user)
    #     curr_good_ranks = [ x for x in range(total_user) if x not in mal_num_list]
    #     curr_good_grads = benigh_local_grads
    #     curr_avg = sum(curr_good_grads) / len(curr_good_grads)

    #     # Update z and mu
    #     if self.t == 0:
    #         self._init_callback(curr_good_grads, curr_avg)
    #     elif self.t < self.warmup:
    #         self._warmup_callback(curr_good_grads, curr_avg)

    #     # Find the target
    #     if self.t < self.warmup:
    #         mv, mi, self._gradient = self._attack_callback(curr_good_grads)
    #         self.target_rank = curr_good_ranks[mi]

    #         # Coordinator log the output
    #         if self.coordinator:
    #             target_rank = curr_good_ranks[mi]
    #             r = {
    #                 "_meta": {"type": "mmc_count"},
    #                 "select": target_rank,
    #                 "value": mv.item(),
    #             }
    #             self.json_logger.info(r)

    #     else:
    #         # Fix device
    #         target_worker = self.simulator.workers[self.target_rank]
    #         self._gradient = target_worker.get_gradient()

    #     self.t += 1
        
        


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

def median_mean_k_defense(weights, group_size = 7, rate = 100, check_indeces = 0, mode = 0, ratio = 1): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users_count = len(weights)
    users_grads = torch.empty((users_count, flatten_params(weights[0]).shape[0]), dtype=torch.float32).to(device)
    for i in range(users_count):
        users_grads[i] = torch.tensor(flatten_params(weights[i]),dtype=torch.float32)
    bound = 0
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
            ######
            # bound = 5*med
            # cc = param_across_users.abs() >= 3*bound
            # ccindices = cc.nonzero()
            # if ccindices.shape[0] == 0 or ccindices.shape[0] == users_count:
            #     print("Not pass bound # is", ccindices.shape[0])
            #     continue
            # else:
            #     print("Not pass bound # is", ccindices.shape[0])
            #     bad_index.append(ccindices.cpu().tolist())
            ######
            
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
        if ratio < 1:
            choose_random_ratio = int(len(good_index)*ratio)
            good_index = good_index[:choose_random_ratio]
        # print("Training----: good_index length is",len(good_index))
        w_pick = [weights[k] for k in good_index]
        w_avg = copy.deepcopy(w_pick[0])
        
        
            
        for key in w_avg.keys():
            for i in range(1, len(w_pick)): 
                w_avg[key] += w_pick[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w_pick))

    # pdb.set_trace()
    return w_avg, update_rate, good_index

def average_weights_bound(weights, mal_user_list, common_list ,group_size = 7, rate = 200, check_indeces = 0, mode = 0,stop_check=0): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users_count = len(weights)
    users_grads = torch.empty((users_count, flatten_params(weights[0]).shape[0]), dtype=torch.float32).to(device)
    for i in range(users_count):
        users_grads[i] = torch.tensor(flatten_params(weights[i]),dtype=torch.float32)
    good_index = []
    while 1:
        i = torch.randint(0, users_grads.shape[1],(1,))
        param_across_users = users_grads.T[i][0].view(-1)
        # pdb.set_trace()
        if torch.count_nonzero(param_across_users) > int(users_count*0.9):
            break
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
    bound = abs(rate/100 * med)
    
    update_rate= 1
    # pdb.set_trace()
    for kk in range(100):
    # kk = 0
    # while 1:
        if kk == 0:
            cc = (param_across_users - med).abs() <= bound
            kk = 1
        else:
            cc = (param_across_users - med).abs() <= bound * update_rate
        good_index = cc.nonzero()
        if good_index.shape[0] <= int(users_count*0.8) and good_index.shape[0]>= int(users_count*0.3):
            if 0:
                print("passed!! Pass bound # is: ", good_index.shape[0]," should be <= ", int(users_count*0.8)," and >= ", int(users_count*0.3))
                if set(mal_user_list).intersection(set(good_index.cpu().numpy().reshape(-1).tolist())) != set():
                    print("set(mal_user_list).intersection(set(good_index)) is:",set(mal_user_list).intersection(set(good_index.cpu().numpy().reshape(-1).tolist())))
                # pdb.set_trace()
            # for tem_x in good_index:
            #     if tem_x.item() >= 40:
            #         pdb.set_trace()
            # if list(set(mal_user_list).intersection(good_index.cpu().numpy().reshape(-1).tolist())) !=  []:
            #     pdb.set_trace()
            for xx in good_index:
                common_list.append(xx.item())
            # common_list.append(good_index)
            # common_list = [item for sublist in common_list for item in sublist]
            # print("Passed!! Pass bound # is: ", good_index.shape[0]," should be <= ", int(users_count*0.7))
            break
        elif good_index.shape[0]< int(users_count*0.3):
            # print("Not pass!! Pass bound # is: ", good_index.shape[0]," should be >= ", int(users_count*0.75))
            # print("not pass!, good_index.shape[0] is:",good_index.shape[0],param_across_users.abs())
            # update_rate = update_rate * 1.1
            update_rate = update_rate * (1+(int(users_count*0.5) - good_index.shape[0]) /users_count/10)
            # update_rate = update_rate * 1.1
            # pdb.set_trace()
            # if good_index.shape[0] == 0:
                # print("update rate should be at least the following to touch the minimum bound", min((param_across_users - med).abs())/bound)
                # print("update rate is", update_rate)
            # pdb.set_trace()
        elif good_index.shape[0]> int(users_count*0.8):
            # print("Not pass!! Pass bound # is: ", good_index.shape[0]," should be <= ", int(users_count*0.8))
            # print("not pass!, good_index.shape[0] is:",good_index.shape[0],param_across_users.abs())
            update_rate = update_rate * (1-(good_index.shape[0]- int(users_count*0.7))/users_count/10)
            # update_rate = update_rate * 0.9
            # update_rate = update_rate * 0.9
    # if kk == 99:
    #     print("reached 100 times")
    #     w_avg = 0
    #     return w_avg, update_rate, good_index, common_list
    #     
    #     pdb.set_trace()
    # print("common list is:",common_list[:5])
    if len(common_list)!=0:
        # print("common_list size is:",len(common_list))
        good_index = np.array([x[0] for x in Counter(common_list).most_common(int(users_count *0.8))])
        # print("using good_index ratio", len(good_index)/users_count/0.8)
        # pdb.set_trace()
        print("# index is", len(good_index), "sorted good index is:",np.sort(good_index))
        for tt in good_index:
            if tt >= 40:
                print("mal in the list", tt)
        # if list(set(mal_user_list).intersection(good_index)) != []:
        #     print("common list is:",common_list[:5],"intersection:",list(set(mal_user_list).intersection(good_index)))
            # pdb.set_trace()
            # exit()
        # else:
        #     print("First 5 good index is:",good_index[:5],"intersection:",list(set(mal_user_list).intersection(good_index)))
    else:
        print("common list is empty")
    # good_index_length = good_index.shape[0]
    # if good_index.shape[0] == 0 or good_index.shape[0] == users_count:
    if good_index.shape[0] == 0 :
    # if good_index.shape[0] == 0 or good_index.shape[0] > users_count *0.80:
        print("good_index.shape[0] is over the threshold, good_index.shape[0] is:",good_index.shape[0])
        w_avg = 0
        return w_avg, update_rate, good_index, common_list

    if stop_check:
        pdb.set_trace()
    # good_index = range(22)
    w_pick = [weights[k] for k in good_index]
    # print("good_index are :",good_index)
    # w_pick = [weights[k] for k in range(users_count)]
    w_avg = copy.deepcopy(w_pick[0])
    
    # pdb.set_trace()
    for key in w_avg.keys():
        for i in range(1, len(w_pick)): 
            w_avg[key] += w_pick[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w_pick))
    return w_avg, update_rate, good_index, common_list

def median_mean_defense(weights, group_size = 7, rate = 100, check_indeces = 0, mode = 0): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users_count = len(weights)
    users_grads = torch.empty((users_count, flatten_params(weights[0]).shape[0]), dtype=torch.float32).to(device)
    for i in range(users_count):
        users_grads[i] = torch.tensor(flatten_params(weights[i]),dtype=torch.float32)
   
    good_index = []
    
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
    # x = torch.abs(mean_vector - med)**2
    # std = torch.sqrt(torch.mean(x))
    
    smallest_value = (param_across_users - med).abs().min()
    # only pick the one closest to med of mean
    c = (param_across_users - med).abs() <= smallest_value
    good_index = c.nonzero()
    good_index_length = good_index.shape[0]
    update_rate = rate
    
    # pdb.set_trace()
    # if good_index_length < int(users_count * 0.1): 
    #     w_avg = 0
    # elif good_index_length > int(users_count*0.85):
    #     w_avg, _ = average_weights(weights, mode = 30)
    # else:
    if 1:
        # print("Training----: good_index length is",len(good_index))
        w_pick = [weights[k] for k in good_index]
        w_avg = copy.deepcopy(w_pick[0])
        
        for key in w_avg.keys():
            for i in range(1, len(w_pick)): 
                w_avg[key] += w_pick[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w_pick))

    # pdb.set_trace()
    return w_avg, update_rate, good_index

def krum_defense(weights, group_size = 7, n= 50,f = 10 ,m = 40):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users_count = len(weights)
    users_grads = []
    mean_vector = []
    num_each_group = int(len(weights) / group_size)
    # if num_each_group * group_size == len(weights):
    for ii in range(group_size):
        w_avg = copy.deepcopy(weights[ii*7]) 
        for key in w_avg.keys():
            for i in range(ii*num_each_group, ii*num_each_group + num_each_group):
                w_avg[key] += weights[i][key]
            w_avg[key] = torch.div(w_avg[key], len(weights))
        mean_vector.append(w_avg)
        
    krum_agg = Krum(n=7,f = 1 ,m = 1)
    w_avg = krum_agg.call(mean_vector)
    update_rate = 100
    good_index = [0]

    return w_avg, update_rate, good_index

def cm_defense(weights,n= 50,f = 10 ,m = 40):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users_count = len(weights)
    users_grads = []
    mean_vector = []
    group_size = 7
    num_each_group = int(len(weights) / group_size)
    # if num_each_group * group_size == len(weights):
    for ii in range(7): # 7 groups, each group is 7 people
        w_avg = copy.deepcopy(weights[ii*7]) 
        for key in w_avg.keys():
            for i in range(ii*num_each_group, ii*num_each_group + num_each_group):
                w_avg[key] += weights[i][key]
            w_avg[key] = torch.div(w_avg[key], len(weights))
            w_avg[key] = w_avg[key].cpu()
            # pdb.set_trace()
            # w_avg.cpu()
        mean_vector.append(w_avg)

    
    # pdb.set_trace()
    cm_agg = CM()
    w_avg = cm_agg.call(mean_vector)
    for x in w_avg.keys():
        w_avg[x] = w_avg[x].to(device)
    # w_avg = krum_agg.call(weights)
    update_rate = 100
    good_index = [0]

    return w_avg, update_rate, good_index

def rfa_defense(weights,n= 50,f = 10 ,m = 40):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users_count = len(weights)
    users_grads = []
    mean_vector = []
    num_each_group = int(len(weights) / 7)
    # if num_each_group * group_size == len(weights):
    for ii in range(7):
        w_avg = copy.deepcopy(weights[ii*7]) 
        for key in w_avg.keys():
            for i in range(ii*num_each_group, ii*num_each_group + num_each_group):
                w_avg[key] += weights[i][key]
            w_avg[key] = torch.div(w_avg[key], len(weights))
        mean_vector.append(w_avg)
        
    rfa_agg = RFA(T= 8)
    w_avg = rfa_agg.call(mean_vector)
    update_rate = 100
    good_index = [0]

    return w_avg, update_rate, good_index

def clip_defense(weights,n= 50,f = 10 ,m = 40):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    users_count = len(weights)
    users_grads = []
    mean_vector = []
    num_each_group = int(len(weights) / 7)
    # if num_each_group * group_size == len(weights):
    for ii in range(7):
        w_avg = copy.deepcopy(weights[ii*7]) 
        for key in w_avg.keys():
            for i in range(ii*num_each_group, ii*num_each_group + num_each_group):
                w_avg[key] += weights[i][key]
            w_avg[key] = torch.div(w_avg[key], len(weights))
        mean_vector.append(w_avg)
        
    clip_agg = Clipping(tau=10.0, n_iter=3)
    ## TODO
    # rfa_agg = RFA(T= 50)
    w_avg = clip_agg.call(mean_vector)
    update_rate = 100
    good_index = [0]

    return w_avg, update_rate, good_index
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

if __name__ == '__main__':
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    current_timestamp = datetime.datetime.now()
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if hasattr(args, 'gpu_id'):
        torch.cuda.set_device(args.gpu_id)
        print('using GPU device ', args.gpu_id)
    # device = 'cuda' if args.gpu is not None else 'cpu'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device is", device)

    print("args.dataset is", args.dataset) 
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
    elif args.model == 'resnet18':
        global_model = ResNet18(num_classes=args.num_classes)    
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

    # pdb.set_trace()
    # total_parameters = sum(p.numel() for p in global_model.parameters())
    # print("total_parameters is", total_parameters)
    init_ckpt = './model/{}_{}_{}_{}_{}_{}_init.ckpt'.format(args.dataset, args.model, args.epochs, args.num_users, args.attack_method, args.en_defence)
    if 0:
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

    # pdb.set_trace()
    args.lr *= args.num_batches_per_step * args.num_users
    # args.lr = args.lr * args.num_batches_per_step * args.num_users
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
    if args.dataset == 'cifar100':
        mal_index_list = random.sample(range(args.num_users),mal_user_num)
    else:
        mal_index_list =   list(range(args.num_users - mal_user_num,args.num_users))        #int(args.mal_ratio * args.num_users) 
    print("mal_index_list is",mal_index_list)
    FoE_start_epoch = 10
    median_mean_rate = args.meank_rate

    
    csv_dict = {'0_epoch': [], '1_train_loss': [],'2_lr': [], '3_test_acc': [], '4_test_loss': []}
    df = pd.DataFrame.from_dict(csv_dict, orient='columns')
    if args.iid:
        path_to_logs = './logs/iid/Adj_{}_{}_{}_{}_att_{}_sca_{}_attr_{}_pass_mmr_{}_iid[{}]_ratio_{}_def_{}_lr_{}_final_aggregation_ratio_{}_group_{}_time_{}'.format(args.dataset, args.model, args.epochs, args.num_users,  args.attack_method,args.attack_scale, args.attack_rate,median_mean_rate, args.iid, args.en_partial_att,args.en_defence,args.lr,args.final_aggregation_ratio, args.group_size,current_timestamp)
    else:
        path_to_logs = './logs/noniid/Adj_{}_{}_{}_{}_att_{}_sca_{}_attr_{}_pass_mmr_{}_iid[{}]_ratio_{}_def_{}_lr_{}_final_aggregation_ratio_{}_group_{}_time_{}'.format(args.dataset, args.model, args.epochs, args.num_users,  args.attack_method,args.attack_scale, args.attack_rate,median_mean_rate, args.iid, args.en_partial_att,args.en_defence,args.lr,args.final_aggregation_ratio, args.group_size,current_timestamp)
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
    common_list = []
    stop_check =0
    for epoch in range(args.epochs):
        print('epoch:', epoch)
        # if epoch == 10:
        #     stop_check= 1
        global_model.train()
        global_optimizer.zero_grad()
        idxs_users = range(args.num_users)
        step_size = args.num_batches_per_step * args.batch_size
        _r_num_batches_per_step = 1.0 / args.num_batches_per_step
    
        global_grads = 0
        for step, (inputs, targets, indices) in enumerate(tqdm(trainloader_global, desc='train', ncols=0, disable=False)):
            # pdb.set_trace()
            adjust_learning_rate(global_scheduler, epoch, step, num_steps_per_epoch,
                         warmup_lr_epochs=args.warmup_lr_epochs, 
                         schedule_lr_per_epoch=args.schedule_lr_per_epoch, 
                         size=args.num_users)
            inputs = inputs.to(device, non_blocking=True)
            
            targets = targets.to(device, non_blocking=True)
            loss = torch.tensor([0.0])
            benigh_local_grads, local_grads, local_weights = [], [], []
            
            # benigh_local_grads_buffer = []
            
            mal_count = 0
            ben_count = 0
            if args.num_users >1:
                for idx in idxs_users:

                    # pdb.set_trace()
                    _, indices_for_this_user, _ = np.intersect1d(indices, list(user_groups[idx]), return_indices=True)
                    # print("user_groups[idx] is",user_groups[idx])
                    # pdb.set_trace()
                    user_inputs = inputs[indices_for_this_user]
                    if user_inputs.shape[0] == 0:
                        break
                    if args.attack_method == 4 and idx in mal_index_list: # label flipping attack
                        # pdb.set_trace()attack
                        
                        # user_targets = 99 - targets[indices_for_this_user]
                        user_targets = targets[indices_for_this_user]*0
                        # pdb.set_trace()
                        # print("min user targets",min(user_targets),"max user targets",max(user_targets))
                        assert min(user_targets) >= 0 and max(user_targets) <= 99
                        # pdb.set_trace()
                        # target
                    else:   
                        user_targets = targets[indices_for_this_user]
                    local_model = local_models[idx]
                    
                    local_model.optimizer.zero_grad()
                    local_model.model.train()
                    
                    _inputs = user_inputs
                    _targets = user_targets
                    _outputs = local_model.model(_inputs)
                    _loss = local_model.criterion(_outputs, _targets)
                    _loss.mul_(_r_num_batches_per_step)
                    _loss.backward()
                    loss += _loss.item()
    
                    #------------------ accumulate local gradients
                    grad = {k:p.grad for k, p in local_model.model.named_parameters()}
                    # pdb.set_trace()
                    
                    local_grads.append(copy.deepcopy(grad))
                    if idx not in mal_index_list:
                        benigh_local_grads.append(copy.deepcopy(grad))
            else:
                local_model = local_models[0]
                    
                local_model.optimizer.zero_grad()
                local_model.model.train()
                
                _inputs = inputs
                _targets = targets
                # if user_inputs.shape != torch.Size([38, 1, 28, 28]):
                #     pdb.set_trace()
                # pdb.set_trace()
                _outputs = local_model.model(_inputs)
                _loss = local_model.criterion(_outputs, _targets)
                _loss.mul_(_r_num_batches_per_step)
                _loss.backward()
                loss += _loss.item()

                #------------------ accumulate local gradients
                grad = {k:p.grad for k, p in local_model.model.named_parameters()}
                # pdb.set_trace()
                
                local_grads.append(copy.deepcopy(grad))
            if args.num_users >1 and user_inputs.shape[0] == 0:
                    continue
            # assert len(local_grads) == mal_user_num
            if args.attack_method == 1:        
                    updated_local_grads = apply_attack_EIFFeL(local_grads, mal_num_list =mal_index_list,  mode =1 , scale= args.attack_scale, partial_att = args.en_partial_att)
                    # scale =3  begin_acc = 52% for mnist non-iid
            elif args.attack_method == 2:        #50
                    updated_local_grads = apply_attack_EIFFeL(local_grads, mal_num_list =mal_index_list,mode =2 , scale= args.attack_scale, partial_att = args.en_partial_att) 
            elif args.attack_method == 3:        #10
                    updated_local_grads = apply_attack_EIFFeL(local_grads, mal_num_list =mal_index_list,mode =3 , scale= args.attack_scale , partial_att = args.en_partial_att, num_std = args.attack_rate ) 
            elif args.attack_method == 4:  
                    updated_local_grads = local_grads
            elif args.attack_method == 5:        #10
                    updated_local_grads = apply_attack_EIFFeL(local_grads, mal_num_list =mal_index_list,mode =5 , scale= args.attack_scale , partial_att = args.en_partial_att, num_std = args.attack_rate ) 
            elif args.attack_method == 51:        #10
                    updated_local_grads = apply_attack_BUCKETING(local_grads, benigh_local_grads,  mal_num_list =mal_index_list,mode =1 , scale= args.attack_scale , partial_att = args.en_partial_att, num_std = args.attack_rate ) 
            elif args.attack_method == 52:        #10
                    updated_local_grads = apply_attack_BUCKETING(local_grads, benigh_local_grads,  mal_num_list =mal_index_list,mode =2 , scale= args.attack_scale , partial_att = args.en_partial_att, num_std = args.attack_rate )  
            elif args.attack_method == 53:        #10
                    updated_local_grads = apply_attack_BUCKETING(local_grads, benigh_local_grads,  mal_num_list =mal_index_list,mode =3 , scale= args.attack_scale , partial_att = args.en_partial_att, num_std = args.attack_rate ) 
            elif args.attack_method == 54:        #10
                    updated_local_grads = apply_attack_BUCKETING(local_grads, benigh_local_grads,  mal_num_list =mal_index_list,mode =4 , scale= args.attack_scale , partial_att = args.en_partial_att, num_std = args.attack_rate ) 
            elif args.attack_method == 55:        #1
                    updated_local_grads = apply_attack_BUCKETING(local_grads, benigh_local_grads,  mal_num_list =mal_index_list,mode =5 , scale= args.attack_scale , partial_att = args.en_partial_att, num_std = args.attack_rate, stop_check= stop_check ) 
            
            
            elif args.attack_method == 0:
                updated_local_grads = local_grads
            else:
                assert 1 == 0

            save_npy = 0
            save_flat= 0
            if save_npy == 1 and args.en_defence ==0 and args.attack_method == 0:
                if save_flat ==1:
                    data_save_path = './all_grad_log/'+ args.dataset +'/'
                    if not os.path.exists(data_save_path):
                        os.makedirs(data_save_path)
                        
                    pdb.set_trace()
                    np.save(data_save_path+"Flat_E{}".format(epoch),abs(updated_local_grads.data.cpu().numpy().flatten()))
                else:
                    for x in range(len(updated_local_grads)):
                        # pdb.set_trace()
                        for index, i in enumerate(updated_local_grads[x]):
                            data_save_path = './all_grad_log/'+ args.dataset +'/'
                            if not os.path.exists(data_save_path):
                                os.makedirs(data_save_path)
                            # if not os.
                            # np.save(data_save_path+"U{}_L{}_E{}".format(x,index,epoch),abs(updated_local_grads[x][i].data.cpu().numpy().flatten()).max())
                        np.save(data_save_path+"U{}_L{}_E{}".format(x,index,epoch),abs(updated_local_grads[x][i].data.cpu().numpy().flatten()))
                continue
            #------------------ apply defense
            good_index = []
            # if args.en_defence and epoch >1:
            if args.en_defence == 1:
                global_grads,update_rate, good_index = median_mean_k_defense(updated_local_grads, group_size = args.group_size, rate = update_rate, check_indeces= 0, mode = args.en_partial_att, ratio = args.final_aggregation_ratio)
            elif args.en_defence == 2:
                global_grads,update_rate, good_index = median_mean_defense(updated_local_grads, group_size = args.group_size, rate = update_rate, check_indeces= 0, mode = args.en_partial_att)
            elif args.en_defence == 31:
                global_grads,update_rate, good_index = krum_defense(updated_local_grads,group_size = 7,n= 50,f = 10 ,m = 1)
                # pdb.set_trace()
            elif args.en_defence == 32:
                global_grads,update_rate, good_index = cm_defense(updated_local_grads,n= 50,f = 10 ,m = 1)
            elif args.en_defence == 33:
                global_grads,update_rate, good_index = rfa_defense(updated_local_grads,n= 50,f = 10 ,m = 1)
            elif args.en_defence == 34:
                global_grads,update_rate, good_index = clip_defense(updated_local_grads,n= 50,f = 10 ,m = 1)
            elif args.en_defence == 10:
                global_grads,update_rate, good_index,common_list = average_weights_bound(updated_local_grads,mal_index_list,common_list, group_size = args.group_size, rate = args.meank_rate, check_indeces= 0, mode = args.en_partial_att, stop_check = stop_check  )
            
            else:
                global_grads,_ = average_weights(updated_local_grads, mal_num = mal_user_num, mode = 1)

            if 0:
                all_grads_list  = []
                # pdb.set_trace()
                # python how to flatten a dict
                for v in global_grads.values():
                    all_grads_list.append(v.cpu().flatten().tolist()) 
                flat_list = [item for sublist in all_grads_list for item in sublist]
                # pdb.set_trace()
                np.save('./grads_data/'+str(epoch)+'_global_grads.npy', np.array(flat_list))
            #------------------ apply accumulated local gradients
            if global_grads == 0:
                # print("global_grads == 0")
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
        
        if save_npy ==1 or global_grads ==0:
            print("global_grads ==0")
            # pdb.set_trace()
            continue            
        test_acc, test_loss = test_inference(args, local_models[0].model, testloader)
        print("Test Accuracy: {:.2f}%".format(100*test_acc))
        if enable_wandb:
            wandb.log({"Time_"+str(current_timestamp)+"_model_"+str(args.model) +"_"+"num_users_"+str(args.num_users)+"_en_defence_" + str(args.en_defence)+"_train acc": test_acc})
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
        elif not args.en_defence:
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
    torch.save(global_model.state_dict(), init_ckpt)
    # Saving the objects train_loss and train_accuracy:
    if not os.path.exists('./save/objects/'):
        os.makedirs('./save/objects/')
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
