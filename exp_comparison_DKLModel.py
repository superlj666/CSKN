import argparse
import torch
import time
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import logging
from distutils.util import strtobool
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import *
import gpytorch
import tqdm
from models import DKLModel, DenseNetFeatureExtractor
from utils import load_data


def train(logger, args, model, device, mll, train_loader, optimizer, epoch):
    model.train()
    likelihood.train()
    with gpytorch.settings.num_likelihood_samples(8):
        for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = -mll(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % args['log_interval'] == 0:
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()

def test(logger, args, model, device, mll, test_loader):
    model.eval()
    likelihood.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += (-mll(output, target).item())
            pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            correct += pred.eq(target.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy, test_loss

torch.manual_seed(1)
logging.basicConfig(level=logging.INFO)
kwargs = {'num_workers': 1, 'pin_memory': True}

args={'log_interval': 1000}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'CIFAR10', help='Specific dataset')
    parser.add_argument("--repeate", type=int, default = 10, help='the number of repeates')
    parser.add_argument("--epochs", type=int, default = 30, help='the number of epochs')
    parser.add_argument("--gpu", type=int, default = 1, help='GPU numer')
    param = parser.parse_args()
    print(param)

    num_repeate = param.repeate
    num_epoch = param.epochs
    loss_arr = np.zeros((num_repeate, num_epoch))
    test_acc_arr = np.zeros((num_repeate, num_epoch))
    test_loss_arr = np.zeros((num_repeate, num_epoch))
    training_time_arr = np.zeros((num_repeate, num_epoch))

    device = torch.device("cuda:{}".format(param.gpu) if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('./results/exp3_{}_DKL'.format(param.dataset))
    logger = logging.getLogger('exp3_{}_DKL'.format(param.dataset))

    if param.dataset == 'MNIST':
        num_classes = 10
        in_channels = 1
        num_data = 60000
    elif param.dataset == 'FashionMNIST':
        num_classes = 10
        num_dimension = 784
    elif param.dataset == 'CIFAR10':
        num_classes = 10
        in_channels = 3
        num_data = 60000
    elif param.dataset == 'CIFAR100':
        num_classes = 10
        num_dimension = 784
    elif param.dataset == 'TinyImagenet':
        num_classes = 10
        num_dimension = 784

    feature_extractor = DenseNetFeatureExtractor(block_config=(6, 6, 6), num_classes=num_classes)
    model = DKLModel(feature_extractor, num_dim=1000).to(device)
    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes).to(device)

    lr = 0.1
    optimizer = SGD([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
        {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
        {'params': model.gp_layer.variational_parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
    scheduler = MultiStepLR(optimizer, milestones=[0.5 * num_epoch, 0.75 * num_epoch], gamma=0.1)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=num_data)

    for idx_repeat in range(num_repeate):
        train_loader, test_loader = load_data(param.dataset)

        for epoch in range(num_epoch):
            start = time.time()
            train_loss = train(logger, args, model, device, mll, train_loader, optimizer, epoch)
            end = time.time()

            test_acc, test_loss = test(logger, args, model, device, mll, test_loader)

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
    
    torch.save(checkpoint, './results/exp3_{}_{}.pt'.format(param.dataset, param.model))