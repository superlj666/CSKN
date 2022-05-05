import os
from math import ceil
import torch
from torch import float32
import torch.utils.data.distributed
import torch.nn.functional as F
from torchvision import datasets, transforms
# from models import *

def load_data(dataset_name, batch_size = 128, data_dir = './data', kwargs={'num_workers': 2, 'pin_memory': True}):
    if dataset_name == 'MNIST':
        transforms_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir, train=True, download=True, transform=transforms_mnist), batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir, train=False, transform=transforms_mnist), batch_size=1000, shuffle=True, **kwargs)
    elif dataset_name == 'FashionMNIST':
        transform_fashionmnist = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(os.path.join(data_dir, './fashionmnist_data/'), train=True, download=True, transform=transform_fashionmnist), batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(os.path.join(data_dir, './fashionmnist_data/'), train=False, transform=transform_fashionmnist), batch_size=1000, shuffle=False, **kwargs)
    elif dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train), batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test), batch_size=200, shuffle=False, **kwargs)
    elif dataset_name == 'CIFAR100':
        stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform), batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform), batch_size=200, shuffle=False, **kwargs)
    elif dataset_name == 'TinyImagenet':
        # data process from https://github.com/pytorch/examples/blob/main/imagenet/main.py
        # https://www.cnblogs.com/liuyangcode/p/14689893.html
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform_train = transforms.Compose([transforms.RandomResizedCrop(64), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), normalize])
        trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'tiny-imagenet-200/train'), transform=transform_train)
        testset = datasets.ImageFolder(root=os.path.join(data_dir, 'tiny-imagenet-200/val'), transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader
    
def load_resize_data(dataset_name, batch_size = 128, data_dir = './data', kwargs={'num_workers': 2, 'pin_memory': True}):
    if dataset_name == 'MNIST':
        transforms_mnist = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir, train=True, download=True, transform=transforms_mnist), batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir, train=False, transform=transforms_mnist), batch_size=1000, shuffle=True, **kwargs)
    elif dataset_name == 'FashionMNIST':
        transform_fashionmnist = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./fashionmnist_data/', train=True, download=True, transform=transform_fashionmnist), batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./fashionmnist_data/', train=False, transform=transform_fashionmnist), batch_size=1000, shuffle=False, **kwargs)
    else:
        train_loader, test_loader = load_data(dataset_name)

    return train_loader, test_loader