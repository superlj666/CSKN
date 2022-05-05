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
from torchvision.models import *
from models import CSKN8, CRFFNet8
from models import train, test
from utils import load_data

torch.manual_seed(1)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('exp3_MNIST')
kwargs = {'num_workers': 1, 'pin_memory': True}


num_repeate = 10
num_epoch = 30
data_dim = 784
loss_arr = np.zeros((num_repeate, num_epoch))
reg_weights_arr = np.zeros((num_repeate, num_epoch))
reg_features_arr = np.zeros((num_repeate, num_epoch))
test_acc_arr = np.zeros((num_repeate, num_epoch))
test_loss_arr = np.zeros((num_repeate, num_epoch))
training_time_arr = np.zeros((num_repeate, num_epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default = 'CSKN8', help='Specific model')
    parser.add_argument("--gpu", type=int, default = 1, help='GPU numer')
    param = parser.parse_args()

    device = torch.device("cuda:{}".format(param.gpu) if torch.cuda.is_available() else "cpu")
    model_name = param.model
    writer = SummaryWriter('./results/exp3_MNIST_{}'.format(model_name))


    args={'lambda_0': 1e-7, 'lambda_1': 1e-6, 'log_interval': 1000}
    if model_name == 'CSKN8':
        lr0 = 1e-4
        lr1 = 1e-6
        sigma0 = 1e-7
        sigma1 = 1e-7
        growth_factor = 1
        additional_regularizers = True
        model = CSKN8(512, 10, 1, sigma0, sigma1, growth_factor).to(device)
        optimizers = [optim.Adam(model.op_list0.parameters(), lr=lr0), optim.Adam(model.op_list1.parameters(), lr=lr0), optim.Adam(model.fc_out.parameters(), lr=lr1)]
    elif model_name == 'CRFFNet8':
        lr0 = 1e-4
        sigma0 = 1e-7
        additional_regularizers = False
        model = CRFFNet8(512, 10, 1, sigma0).to(device)
        optimizers = [optim.Adam(model.parameters(), lr=lr0)]

    criterion = nn.CrossEntropyLoss()
    
    for idx_repeat in range(num_repeate):
        train_loader, test_loader = load_data('MNIST')

        for epoch in range(num_epoch):
            start = time.time()
            loss, reg_weights, reg_features = train(logger, args, model, device, criterion, train_loader, optimizers, epoch, additional_regularizers)
            end = time.time()

            test_acc, test_loss = test(logger, args, model, device, criterion, test_loader)

            loss_arr[idx_repeat, epoch] = loss
            reg_weights_arr[idx_repeat, epoch] = reg_weights
            reg_features_arr[idx_repeat, epoch] = reg_features
            test_acc_arr[idx_repeat, epoch] = test_acc
            test_loss_arr[idx_repeat, epoch] = test_loss
            training_time_arr[idx_repeat, epoch] = end - start

            writer.add_scalars('Loss/train', {'loss': loss, 'reg_weights': reg_weights, 'reg_features': reg_features}, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)

        print('Repeate #{}: test accuracy {:.2f} %'.format(idx_repeat, test_acc_arr[idx_repeat, epoch-1]))

    writer.close()
    checkpoint = {'loss_arr': loss_arr, 'reg_weights_arr': reg_weights_arr, 'reg_features_arr': reg_features_arr, 'test_acc_arr': test_acc_arr, 'test_loss_arr': test_loss_arr, 'training_time_arr': training_time_arr}
    print('Mean test accuracy {:.2f} %'.format(test_acc_arr.mean(0)[-1]))
    
    torch.save(checkpoint, './results/exp3_MNIST_{}.pt'.format(model_name))
