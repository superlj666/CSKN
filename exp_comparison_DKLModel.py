import argparse
import torch
import time
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import math
import logging
import torch.nn.functional as F
from distutils.util import strtobool
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import *
import gpytorch
import tqdm
from models import CRFFNet8
from densenet import DenseNet
from utils import load_resize_data

class DenseNetFeatureExtractor(DenseNet):
    # def __init__(self):
    #     super(DenseNetFeatureExtractor, self).__init__()
    #     self.features
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        return out

class FeatureExtractor(CRFFNet8):
    def forward(self, x):
        for i in range(len(self.op_list)):
            x =  torch.cos(self.op_list[i](x))
            x = self.bn_list[i](x)
            if i in self.pool_dict:
                x = self.max_pool(x)
        x = self.avg_pool(x)
        feature_mapping = torch.flatten(x, 1) 
        return feature_mapping

class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res


def train(logger, args, model, likelihood, device, mll, train_loader, optimizer, epoch):
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

def test(logger, args, model, likelihood, device, mll, test_loader):
    model.eval()
    likelihood.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += (-mll(output, target).item())
            output = likelihood(output)
            pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            correct += pred.eq(target.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy, test_loss

torch.manual_seed(1)
logging.basicConfig(level=logging.INFO)
kwargs = {'num_workers': 1, 'pin_memory': True}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = 'MNIST', help='Specific dataset')
    parser.add_argument("--epochs", type=int, default = 30, help='the number of epochs')
    parser.add_argument("--repeates", type=int, default = 10, help='the number of repeates')
    parser.add_argument("--gpu", type=int, default = 1, help='GPU numer')
    param = parser.parse_args()
    print(param)

    dataset_name = param.dataset
    num_epoch = param.epochs
    num_repeate = param.repeates
    loss_arr = np.zeros((num_repeate, num_epoch))
    test_acc_arr = np.zeros((num_repeate, num_epoch))
    test_loss_arr = np.zeros((num_repeate, num_epoch))
    training_time_arr = np.zeros((num_repeate, num_epoch))

    device = torch.device("cuda:{}".format(param.gpu) if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('./results/logs/exp_comparison_{}_DKL_E{}_R{}'.format(dataset_name, num_epoch, num_repeate))
    logger = logging.getLogger('exp_comparison_{}_DKL_E{}_R{}'.format(dataset_name, num_epoch, num_repeate))

    if dataset_name == 'MNIST':
        num_dimension = 512
        num_classes = 10
        in_channels = 1
        num_classes = 10
        num_data = 60000
        sigma0 = 1e-7
    elif dataset_name == 'FashionMNIST':
        num_dimension = 512
        num_classes = 10
        in_channels = 1
        num_classes = 10
        num_data = 60000
        sigma0 = 1e-5
    elif dataset_name == 'CIFAR10':
        num_dimension = 2048
        num_classes = 10
        in_channels = 3
        num_classes = 10
        num_data = 50000
        sigma0 = 1e-4
    elif dataset_name == 'CIFAR100':
        num_dimension = 2048
        num_classes = 100
        in_channels = 3
        num_classes = 100
        num_data = 50000
        sigma0 = 1e-6
    elif dataset_name == 'TinyImagenet':
        num_dimension = 8192
        num_classes = 200
        in_channels = 3
        num_classes = 200
        num_data = 100000
        sigma0 = 1e-7    
    args={'log_interval': 1000}

    for idx_repeat in range(num_repeate):
        # feature_extractor = FeatureExtractor(num_dimension, num_classes, in_channels, sigma0)
        feature_extractor = DenseNetFeatureExtractor(in_channels=in_channels, block_config=(6, 6, 6), num_classes=num_classes)
        num_features = feature_extractor.classifier.in_features
        model = DKLModel(feature_extractor, num_dim=num_features).to(device)
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

        train_loader, test_loader = load_resize_data(dataset_name)

        for epoch in range(num_epoch):
            start = time.time()
            train_loss = train(logger, args, model, likelihood, device, mll, train_loader, optimizer, epoch)
            end = time.time()

            test_acc, test_loss = test(logger, args, model, likelihood, device, mll, test_loader)

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
    
    torch.save(checkpoint, './results/exp_comparison_{}_DKL_E{}_R{}.pt'.format(dataset_name, num_epoch, num_repeate))
