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
import matplotlib.pyplot as plt
import torchvision.models as models
from models import PretrainModel
from utils import load_resize_data

def pretrained_train(logger, args, model, device, criterion, train_loader, optimizer, epoch=5):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()

def pretrained_inference(logger, args, model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy, test_loss

torch.manual_seed(1)
logging.basicConfig(level=logging.INFO)
kwargs = {'num_workers': 1, 'pin_memory': True}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default = 'shufflenet', help='Specific model')
    parser.add_argument("--dataset", type=str, default = 'MNIST', help='Specific dataset')
    parser.add_argument("--repeates", type=int, default = 1, help='the number of repeates')
    parser.add_argument("--epochs", type=int, default = 30, help='the number of epochs')
    parser.add_argument("--gpu", type=int, default = 1, help='GPU numer')
    param = parser.parse_args()
    print(param)

    model_name = param.model
    dataset_name = param.dataset
    num_epoch = param.epochs
    num_repeate = param.repeates
    loss_arr = np.zeros((num_repeate, num_epoch))
    test_acc_arr = np.zeros((num_repeate, num_epoch))
    test_loss_arr = np.zeros((num_repeate, num_epoch))
    training_time_arr = np.zeros((num_repeate, num_epoch))

    device = torch.device("cuda:{}".format(param.gpu) if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('./results/logs/exp_comparison_{}_{}_E{}_R{}'.format(dataset_name, model_name, num_epoch, num_repeate))
    logger = logging.getLogger('exp_comparison_{}_{}_E{}_R{}'.format(dataset_name, model_name, num_epoch, num_repeate))
    args={'log_interval': 1000}

    criterion = nn.CrossEntropyLoss()
    for idx_repeat in range(num_repeate):
        train_loader, test_loader = load_resize_data(dataset_name)

        if dataset_name == 'MNIST':
            model = PretrainModel(model_name, 1, 10).to(device=device)
        elif dataset_name == 'FashionMNIST':
            model = PretrainModel(model_name, 1, 10).to(device=device)
        elif dataset_name == 'CIFAR10':
            model = PretrainModel(model_name, 3, 10).to(device=device)
        elif dataset_name == 'CIFAR100':
            model = PretrainModel(model_name, 3, 100).to(device=device)
        elif dataset_name == 'TinyImagenet':
            model = PretrainModel(model_name, 3, 200).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        for epoch in range(num_epoch):
            start = time.time()
            train_loss = pretrained_train(logger, args, model, device, criterion, train_loader, optimizer, epoch)
            end = time.time()

            test_acc, test_loss = pretrained_inference(logger, args, model, device, criterion, test_loader)

            loss_arr[idx_repeat, epoch] = train_loss
            test_acc_arr[idx_repeat, epoch] = test_acc
            test_loss_arr[idx_repeat, epoch] = test_loss
            training_time_arr[idx_repeat, epoch] = end - start

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)

        print('Repeate #{}: test accuracy {:.2f} %'.format(idx_repeat, test_acc_arr[idx_repeat, epoch-1]))

    writer.close()
    checkpoint = {'loss_arr': loss_arr, 'test_acc_arr': test_acc_arr, 'test_loss_arr': test_loss_arr, 'training_time_arr': training_time_arr}
    print('Mean test accuracy {:.2f} %'.format(test_acc_arr.mean(0)[-1]))
    
    torch.save(checkpoint, './results/exp_comparison_{}_{}_E{}_R{}'.format(dataset_name, model_name, num_epoch, num_repeate))