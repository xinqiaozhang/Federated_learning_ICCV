#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from PIL import Image
import pdb
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar100_iid, cifar_noniid, cifar_noniid_cifar100
from torchvision.transforms import ToTensor

class CIFAR10_idx(datasets.CIFAR10):
    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False,):
        super(CIFAR10_idx, self).__init__(root, train=train, transform=transform,
                                      target_transform=target_transform, download=download)
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index 

class CIFAR100_idx(datasets.CIFAR100):
    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False,):
        super(CIFAR100_idx, self).__init__(root, train=train, transform=transform,
                                      target_transform=target_transform, download=download)
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index     
class MINIST_idx(datasets.MNIST):
    def __init__(self, root: str, train: bool = True, transform = None, target_transform = None, download: bool = False,):
        super(MINIST_idx, self).__init__(root, train=train, transform=transform,
                                      target_transform=target_transform, download=download)
    
    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # pdb.set_trace()
        img = img.numpy()
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, index 


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    # pdb.set_trace()
    if args.dataset == 'cifar10':
        data_dir = '../data/cifar/'
        train_transforms = transforms.Compose(
            [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ]
        )
        test_transforms = transforms.Compose(
            [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ]
        )

        train_dataset = CIFAR10_idx(data_dir, train=True, download=True,
                                       transform=train_transforms)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=test_transforms)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        # apply_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))])

        # train_dataset = MINIST_idx(data_dir, train=True, download=True,
        #                                transform=apply_transform)
        train_dataset = MINIST_idx(data_dir, train=True, download=True,
                                       transform=ToTensor())
        # train_dataset = datasets.MNIST(data_dir, train=True, download=True,
        #                                transform=apply_transform)
        
        # test_dataset = datasets.MNIST(data_dir, train=False, download=True,
        #                               transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=ToTensor())
        

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'cifar100':
        # else:
        
        data_dir = '../data/cifar100/'
        print('using cifar100')
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # train_transforms = transforms.Compose(
        #     [
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        #     ]
        # )
        # test_transforms = transforms.Compose(
        #     [
        #     transforms.Resize(32),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        #     ]
        # )

        train_dataset = CIFAR100_idx(data_dir, train=True, download=True,
                                       transform=train_transforms)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=test_transforms)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar100_iid(train_dataset, args.num_users)
            
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid_cifar100(train_dataset, args.num_users)
    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Global Batch size   : {args.batch_size}')
    return
