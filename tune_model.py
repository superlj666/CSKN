"""
A deep MNIST classifier using convolutional layers.

This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import os
import sys
import argparse
import logging
import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nni.utils import merge_parameter
from torchvision import datasets, transforms
from yaml import load
from torch.utils.tensorboard import SummaryWriter
# sys.path.append("..")
from models import CSKN8
from models import train, test
from utils import load_data

logger = logging.getLogger('Tune_CSKN8')

def main(args):
    torch.manual_seed(args['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True}

    additional_regularizers = True
    lr0 = args['lr0']
    lr1 = args['lr1']
    sigma0 = args['kernel_par1']
    sigma1 = args['kernel_par1']
    growth_factor = args['growth_factor']

    if args['dataset'] == 'MNIST': # 6uYFEKv1, 1e-7, 1e-6
        train_loader, test_loader = load_data('MNIST')
        model = CSKN8(512, 10, 1, sigma0, sigma1, growth_factor).to(device)

    elif args['dataset'] == 'FashionMNIST': # CUpHeqkW, 1e-6, 1e-5 
        train_loader, test_loader = load_data('FashionMNIST')
        model = CSKN8(512, 10, 1, sigma0, sigma1, growth_factor).to(device)

    elif args['dataset'] == 'SVHN': # yKVXxYDd, 1e-7, 1e-6
        train_loader, test_loader = load_data('SVHN')        
        model = CSKN8(2048, 10, 3, sigma0, sigma1, growth_factor).to(device)

    elif args['dataset'] == 'CIFAR10': # mLfxNZFM, 1e-6, 1e-6
        train_loader, test_loader = load_data('CIFAR10')
        model = CSKN8(2048, 10, 3, sigma0, sigma1, growth_factor).to(device)

    elif args['dataset'] == 'CIFAR100': # ngRXSVJf, 1e-5, 1e-7
        train_loader, test_loader = load_data('CIFAR100')
        model = CSKN8(2048, 100, 3, sigma0, sigma1, growth_factor).to(device)

    elif args['dataset'] == 'TinyImagenet': # b3DpVlMg, 1e-5, 1e-7
        train_loader, test_loader = load_data('TinyImagenet')
        model = CSKN8(8192, 200, 3, sigma0, sigma1, growth_factor).to(device)


    optimizers = [optim.Adam(model.op_list0.parameters(), lr=lr0), optim.Adam(model.op_list1.parameters(), lr=lr0), optim.Adam(model.fc_out.parameters(), lr=lr1)]
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args['epochs'] + 1):
        [loss, trace_norm, Frobenius_norm] = train(logger, args, model, device, criterion, train_loader, optimizers, epoch, additional_regularizers)
        test_acc, test_loss = test(logger, args, model, device, criterion, test_loader)

        # report intermediate result
        nni.report_intermediate_result(test_acc)
        # nni.report_intermediate_result(loss)
        logger.debug('test accuracy %g', test_acc)
        logger.debug('Pipe send intermediate result done.')

    # report final result
    nni.report_final_result(test_acc)
    # nni.report_final_result(loss)
    logger.debug('Final result is %g', test_acc)
    logger.debug('Send final result done.')

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Tune Hyperparameters')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--lr0', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--lr1', type=float, default=0.00001, metavar='LR', help='learning rate (default: 0.00001)')
    parser.add_argument("--kernel_par0", type=float, default=0.000001, help="kernel hyperparameter0")
    parser.add_argument("--kernel_par1", type=float, default=0.000001, help="kernel hyperparameter1")
    parser.add_argument("--growth_factor", type=float, default=4, help="growth factor for kernel parameters")
    parser.add_argument("--lambda_0", type=float, default=1e-5, help="regularization parameter0")
    parser.add_argument("--lambda_1", type=float, default=1e-7, help="regularization parameter1")
    parser.add_argument("--dataset", type=str, default='SVHN', help="the used dataset")
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
