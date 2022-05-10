import argparse
import torch
import time
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
import matplotlib.pyplot as plt
from models import DSKN
from models import train, test

torch.manual_seed(1)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('mnist_non-spectral')
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True}

transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))])


num_repeate = 10
num_epoch = 30
loss_arr = np.zeros((num_repeate, num_epoch))
test_acc_arr = np.zeros((num_repeate, num_epoch))
time_arr = np.zeros((num_repeate, num_epoch))
trace_arr = np.zeros(num_repeate)

lr0 = 1e-4
lr1 = 1e-4
sigma0 = 1e-2
sigma1 = 1e-2
growth_factor = [None, 1, 32, 16, 10, 7, 1]
args={'lambda_0': 1e-7, 'lambda_1': 1e-4, 'log_interval': 1000}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=5)
    param = parser.parse_args()
    num_layers = param.num_layers

    for idx_repeat in range(num_repeate):
        args={'lambda_0': 0, 'lambda_1': 0, 'log_interval': 1000}
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=128, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1000, shuffle=True, **kwargs)

        model = DSKN(784, [2000 for _ in range(num_layers)], 10, sigma0, sigma1, growth_factor[num_layers], True).to(device)
        optimizers = [optim.Adam(model.op_list0.parameters(), lr=lr0), optim.Adam(model.op_list1.parameters(), lr=lr0), optim.Adam(model.fc_out.parameters(), lr=lr1)]
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epoch):
            start = time.time()
            train(logger, args, model, device, criterion, train_loader, optimizers, epoch)
            end = time.time()
            time_arr[idx_repeat, epoch] = end-start

            test_acc, test_loss = test(logger, args, model, device, criterion, test_loader)
            test_acc_arr[idx_repeat, epoch] = test_acc
            loss_arr[idx_repeat, epoch] = test_loss
            print('Layers #{}, Repeate #{}, Epoch #{}: test accuracy {:.2f} %, runing time {}'.format(num_layers, idx_repeat, epoch, test_acc, end-start))

        test_loader = torch.utils.data.DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=10000, shuffle=False)
        dataiter = iter(test_loader)
        images, _ = dataiter.next()
        feature_mapping, _ = model(images.to(device))
        trace = torch.pow(torch.norm(feature_mapping, 'fro'), 2)
        trace_arr[idx_repeat] = trace.item() / len(images)

    checkpoint = {'loss_arr': loss_arr, 'test_acc_arr': test_acc_arr, 'time_arr': time_arr, 'trace_arr': trace_arr}
    torch.save(checkpoint, './results/exp_ablation1_{}layers.pt'.format(num_layers))

