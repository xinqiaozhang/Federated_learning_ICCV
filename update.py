#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history=[]

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(val)


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label, _ = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return image, label


# class LocalUpdate(object):
#     def __init__(self, args, dataset, idxs, logger):
#         self.args = args
#         self.logger = logger
#         self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
#         self.device = 'cuda' if args.gpu is not None else 'cpu'
#         # Default criterion set to NLL loss function
#         self.criterion = nn.CrossEntropyLoss().to(self.device) #nn.NLLLoss().to(self.device)

#     def train_val_test(self, dataset, idxs):
#         """
#         Returns train, validation and test dataloaders for a given dataset
#         and user indexes.
#         """
#         # # split indexes for train, validation, and test (80, 10, 10)
#         # idxs_train = idxs[:int(0.8*len(idxs))]
#         # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
#         # idxs_test = idxs[int(0.9*len(idxs)):]

#         # split indexes for train, validation, and test (80, 10, 10)
#         idxs_train = idxs[:int(0.9*len(idxs))]
#         idxs_test = idxs[int(0.9*len(idxs)):]

#         trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
#                                  batch_size=self.args.local_bs, shuffle=True)
#         validloader = None #DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=False)
#         testloader = DataLoader(DatasetSplit(dataset, idxs_test),
#                                 batch_size=int(len(idxs_test)/10), shuffle=False)
#         return trainloader, validloader, testloader

#     def update_weights(self, model, global_round):
#         # Set mode to train model
#         model.train()
        
#         losses = AverageMeter()
#         correct = 0
#         total = 0

#         # Set optimizer for the local updates
#         params = list(model.parameters())
#         trainable_params = []
#         for p in params:
#             if p.requires_grad==True:
#                 trainable_params.append(p)

#         if self.args.optimizer == 'sgd':
#             optimizer = torch.optim.SGD(trainable_params, lr=self.args.lr, momentum=0.9)
#         elif self.args.optimizer == 'adam':
#             optimizer = torch.optim.Adam(trainable_params, lr=self.args.lr, weight_decay=1e-5)

#         with tqdm(total=self.args.local_ep) as pbar:
#             for iter in range(self.args.local_ep):
#                 batch_loss = []
#                 for batch_idx, (images, labels) in enumerate(self.trainloader):
#                     images, labels = images.to(self.device), labels.to(self.device)
                    
#                     outputs = model(images)
#                     loss = self.criterion(outputs, labels)
#                     self.logger.add_scalar('loss', loss.item())
#                     losses.update(loss.data.item(), images.size(0))

#                     _, predicted = outputs.max(1)
#                     total += labels.size(0)
#                     correct += predicted.eq(labels).sum().item()

#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()

#                 pbar.update(1)
#                 pbar.set_description('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(global_round, iter, losses.avg))

#         return model.state_dict(), losses.avg, correct*1./total

    
#     def inference(self, model):
#         """ Returns the inference accuracy and loss.
#         """
#         model.eval()

#         losses = AverageMeter()
#         val_total, val_correct = 0.0, 0.0

#         # computing accuracy on the validation dataset
#         with torch.no_grad():
#             for batch_idx, (images, labels) in enumerate(self.testloader):
#                 images, labels = images.to(self.device), labels.to(self.device)

#                 # Inference
#                 outputs = model(images)
#                 loss = self.criterion(outputs, labels)
#                 losses.update(loss.data.item(), images.size(0))

#                 # Prediction
#                 _, pred_labels = outputs.max(1)
#                 val_correct += torch.sum(torch.eq(pred_labels, labels)).item()
#                 val_total += len(labels)
        
#         val_accuracy = val_correct*1./val_total

#         return val_accuracy


class LocalUpdate(object):
    def __init__(self, args, model, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu is not None else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device) #nn.NLLLoss().to(self.device)

        self.model = model
        
        params = list(self.model.parameters())
        trainable_params = []
        for p in params:
            if p.requires_grad==True:
                trainable_params.append(p)

        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=195)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-5)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # # split indexes for train, validation, and test (80, 10, 10)
        # idxs_train = idxs[:int(0.8*len(idxs))]
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]

        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = None #DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    # def update_weights(self, global_round, batch_idx):
    def update_weights(self, global_round, images, labels):    
        # Set mode to train model
        self.model.train()
        
        losses = AverageMeter()
        correct = 0
        total = 0

        # Set optimizer for the local updates
        # with tqdm(total=self.args.local_ep) as pbar:
            # for iter in range(self.args.local_ep):
        self.optimizer.zero_grad()
        batch_loss = []
        # for idx, (images, labels) in enumerate(self.trainloader):
        # if idx == batch_idx:
        images, labels = images.to(self.device), labels.to(self.device)
        
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.logger.add_scalar('loss', loss.item())
        losses.update(loss.data.item(), images.size(0))

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loss.backward()
        self.optimizer.step()
        
        return self.model.state_dict(), losses.avg, correct*1./total
                        
                    
                    # grads = torch.cat([param.grad.data.view(-1) for param in self.model.parameters()], 0)
                    # self.optimizer.step()
                    # pbar.update(1)
                    # pbar.set_description('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(global_round, iter, losses.avg))
                
            # self.optimizer.step()
            # self.scheduler.step()
            # pbar.update(1)

        return self.model.state_dict(), losses.avg, correct*1./total   #, grads

    
    def inference(self):
        """ Returns the inference accuracy and loss.
        """
        self.model.eval()

        losses = AverageMeter()
        val_total, val_correct = 0.0, 0.0

        # computing accuracy on the validation dataset
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                losses.update(loss.data.item(), images.size(0))

                # Prediction
                _, pred_labels = outputs.max(1)
                val_correct += torch.sum(torch.eq(pred_labels, labels)).item()
                val_total += len(labels)
        
        val_accuracy = val_correct*1./val_total

        return val_accuracy


def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu is not None else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device) #nn.NLLLoss().to(device)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            if args.model == 'mnist':
                images = images.reshape(-1,28*28)
            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

    accuracy = correct/total
    return accuracy, loss/total
