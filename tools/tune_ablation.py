"""
A deep MNIST classifier using convolutional layers.

This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import os
import argparse
import logging
import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nni.utils import merge_parameter
from torchvision import datasets, transforms
from models import VanillaCNNNet, DNN, DSKN, CSKN, CSKN8
from models import train, test
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger('mnist_AutoML')
writer = SummaryWriter('./results/multilayer')

def main(args):
    torch.manual_seed(args['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1000, shuffle=True, **kwargs)

    if args['method'] == 'non-stationary':
        ### Hyperparameter for non-stationary kernel 
        ### ID: UxYZNIO9
        additional_regularizers = False
        sigma0 = args['kernel_par0'] # 1e-2
        sigma1 = args['kernel_par1'] # 1e-3
        lr1 = args['lr1'] # 1e-2
        model = DSKN(784, [2000], 10, sigma0, sigma1, 1, False).to(device)
        optimizers = [optim.Adam(model.fc_out.parameters(), lr=lr1)]
    elif args['method'] == 'kernel_learning':
        ### Hyperparameter for kernel learning
        additional_regularizers = False
        sigma0 = args['kernel_par0']
        sigma1 = args['kernel_par1']
        lr0 = args['lr0']
        lr1 = args['lr1']
        model = DSKN(784, [2000], 10, sigma0, sigma1, 1, True).to(device)
        optimizers = [optim.Adam(model.op_list0.parameters(), lr=lr0), optim.Adam(model.op_list1.parameters(), lr=lr0), optim.Adam(model.fc_out.parameters(), lr=lr1)]
    elif args['method'] == 'multilayer':
        # VXD1deZv for 2-layers
        # t48lRczm for 3-layers
        # VKyYWk3C for 4-layers
        # VKyYWk3C for 5-layers
        additional_regularizers = False
        sigma0 = 1e-2#args['kernel_par0']
        sigma1 = 1e-2#args['kernel_par1']
        lr0 = 1e-4#args['lr0']
        lr1 = 1e-4#args['lr1']
        growth_factor = args['growth_factor']
        model = DSKN(784, [2000, 2000, 2000, 2000, 2000], 10, sigma0, sigma1, growth_factor, True).to(device)
        optimizers = [optim.Adam(model.op_list0.parameters(), lr=lr0), optim.Adam(model.op_list1.parameters(), lr=lr0), optim.Adam(model.fc_out.parameters(), lr=lr1)]
    elif args['method'] == 'regularizer': # 1e-7, 1e-4, vJ0Sxzab
        additional_regularizers = True
        lr0 = 1e-4
        lr1 = 1e-4
        sigma0 = 1e-2
        sigma1 = 1e-2
        growth_factor = 8
        model = DSKN(784, [2000, 2000, 2000, 2000, 2000], 10, sigma0, sigma1, growth_factor, True).to(device)
        optimizers = [optim.Adam(model.op_list0.parameters(), lr=lr0), optim.Adam(model.op_list1.parameters(), lr=lr0), optim.Adam(model.fc_out.parameters(), lr=lr1)]
    elif args['method'] == 'convolutional': # buXHIrtf
        additional_regularizers = True
        args['lambda_0'] = 1e-7
        args['lambda_1'] = 1e-4
        lr0 = args['lr0']
        lr1 = args['lr1']
        sigma0 = args['kernel_par0']
        sigma1 = args['kernel_par1']
        growth_factor = args['growth_factor']
        model = CSKN(784, 10, 1, 20, 5, 3, sigma0, sigma1, growth_factor, True).to(device)
        optimizers = [optim.Adam(model.op_list0.parameters(), lr=lr0), optim.Adam(model.op_list1.parameters(), lr=lr0), optim.Adam(model.fc_out.parameters(), lr=lr1)]

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args['epochs'] + 1):
        [loss, trace_norm, Frobenius_norm] = train(logger, args, model, device, criterion, train_loader, optimizers, epoch, additional_regularizers)
        test_acc, test_loss = test(logger, args, model, device, criterion, test_loader)

        writer.add_scalars('Loss/train', {'loss': loss, 'trace_norm': trace_norm, 'Frobenius_norm': Frobenius_norm}, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        # report intermediate result
        nni.report_intermediate_result(test_acc)
        # nni.report_intermediate_result(loss)
        logger.debug('test accuracy %g', test_acc)
        logger.debug('Pipe send intermediate result done.')
    writer.close()

    # report final result
    nni.report_final_result(test_acc)
    # nni.report_final_result(loss)
    logger.debug('Final result is %g', test_acc)
    logger.debug('Send final result done.')

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Tune Hyperparameters')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument("--hidden_layers", type=int, default=5, metavar='N', help='hidden layer depth (default: 5)')
    parser.add_argument("--out_channels", type=int, default=5, metavar='N', help='out channels (default: 5)')
    parser.add_argument("--kernel_size", type=int, default=5, metavar='N', help='kernel size (default: 5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--lr0', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--lr1', type=float, default=0.00001, metavar='LR', help='learning rate (default: 0.00001)')
    parser.add_argument("--optimizer", type=str, default='Adam', help="opimizer")
    parser.add_argument("--kernel_par", type=float, default=1e-2, help="kernel hyperparameter")
    parser.add_argument("--kernel_par0", type=float, default=0.01, help="kernel hyperparameter0")
    parser.add_argument("--kernel_par1", type=float, default=0.01, help="kernel hyperparameter1")
    parser.add_argument("--growth_factor", type=float, default=16, help="growth factor for kernel parameters")
    parser.add_argument("--lambda_0", type=float, default=1e-5, help="regularization parameter0")
    parser.add_argument("--lambda_1", type=float, default=1e-4, help="regularization parameter1")
    parser.add_argument("--method", type=str, default='convolutional', help="the method to tune")
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
