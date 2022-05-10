import argparse
import torch
import time
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import logging
from distutils.util import strtobool
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from models import CSKN
from models import train, test

torch.manual_seed(1)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mnist_non-spectral')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True}

transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))])


num_repeate = 10
num_epoch = 30
loss_arr = np.zeros((num_repeate, num_epoch))
reg_weights_arr = np.zeros((num_repeate, num_epoch))
reg_features_arr = np.zeros((num_repeate, num_epoch))
test_acc_arr = np.zeros((num_repeate, num_epoch))
test_loss_arr = np.zeros((num_repeate, num_epoch))

num_layers = 5
lr0 = 1e-2
lr1 = 1e-2
sigma0 = 1e-2
sigma1 = 1e-2
growth_factor = 7
additional_regularizers = True

args={'lambda_0': 1e-7, 'lambda_1': 1e-6, 'log_interval': 1000}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_size", type=int, default=3)
    param = parser.parse_args()
    kernel_size = param.kernel_size
    print(kernel_size)
    writer = SummaryWriter('./results/convolutional{}'.format(kernel_size))
    
    for idx_repeat in range(num_repeate):
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=128, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1000, shuffle=True, **kwargs)

        model = CSKN(784, 10, 1, 20, 5, kernel_size, sigma0, sigma1, growth_factor, True).to(device)
        optimizers = [optim.Adam(model.op_list0.parameters(), lr=lr0), optim.Adam(model.op_list1.parameters(), lr=lr0), optim.Adam(model.fc_out.parameters(), lr=lr1)]
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epoch):
            loss, reg_weights, reg_features = train(logger, args, model, device, criterion, train_loader, optimizers, epoch, additional_regularizers)
            test_acc, test_loss = test(logger, args, model, device, criterion, test_loader)
            loss_arr[idx_repeat, epoch] = loss
            reg_weights_arr[idx_repeat, epoch] = reg_weights
            reg_features_arr[idx_repeat, epoch] = reg_features
            test_acc_arr[idx_repeat, epoch] = test_acc
            test_loss_arr[idx_repeat, epoch] = test_loss
            writer.add_scalars('Loss/train', {'loss': loss, 'reg_weights': reg_weights, 'reg_features': reg_features}, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)

        print('Repeate #{}: test accuracy {:.2f} %'.format(idx_repeat, test_acc_arr[idx_repeat, epoch-1]))

    writer.close()
    checkpoint = {'loss_arr': loss_arr, 'reg_weights_arr': reg_weights_arr, 'reg_features_arr': reg_features_arr, 'test_acc_arr': test_acc_arr, 'test_loss_arr': test_loss_arr}
    torch.save(checkpoint, './results/exp_ablation3_convolutional{}.pt'.format(kernel_size))
