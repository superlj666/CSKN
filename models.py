import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import math
import torchvision
import torchvision.transforms as transforms
import numpy as np
import gpytorch
from torchvision import models

class DNN(nn.Module):
    def __init__(self, input_dimension, output_dimension, num_class, num_layers):
        super().__init__()

        self.fc0 = nn.Linear(input_dimension, output_dimension)
        self.fc_list = []
        for _ in range(num_layers - 1):
            fc_tmp = nn.Linear(output_dimension, output_dimension)
            self.fc_list.append(fc_tmp)
        self.fc_list = nn.ModuleList(self.fc_list)
        self.fc_out = nn.Linear(output_dimension, num_class)

    def forward(self, x):
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc0(x))
        for fc in self.fc_list:
            x = F.relu(fc(x))
        feature_mapping = x
        outputs = self.fc_out(x)
        return [feature_mapping, outputs]

class VanillaCNNNet(nn.Module):
    def __init__(self, num_dimension, num_class, in_channels, out_channels, num_layers, kernel_size):
        super().__init__()
        _padding = int((kernel_size-1)/2)
        self.op0 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=_padding)
        self.convs_list = [nn.Conv2d(out_channels, out_channels, kernel_size, padding=_padding) for _ in range(num_layers - 1)]
        self.convs_list = nn.ModuleList(self.convs_list)
        self.fc_out = nn.Linear(out_channels * num_dimension * num_dimension, num_class)

    def forward(self, x):
        x = F.relu(self.op0(x))
        for conv_layer in self.convs_list:
            x = F.relu(conv_layer(x))
        x = torch.flatten(x, 1) 
        feature_mapping = x
        outputs = self.fc_out(x)
        return [feature_mapping, outputs]

class RFFNet(nn.Module):
    def __init__(self, num_dimension, out_dimension, num_class, sigma, backpropagation=True):
        super().__init__()
        self.out_dimension = out_dimension
        self.sigma = sigma
        self.num_layers = len(out_dimension)
        self.backpropagation = backpropagation

        self.op_list = [nn.Linear(num_dimension, out_dimension[0])]
        for i in range(1, self.num_layers):
            self.op_list.append(nn.Linear(out_dimension[i-1], out_dimension[i]))
        self.op_list = nn.ModuleList(self.op_list)
        self.fc_out = nn.Linear(out_dimension[-1], num_class)
        self.init_weights()

    def init_weights(self):
        for op in self.op_list:
            nn.init.normal_(op.weight, 0, self.sigma)
            nn.init.uniform_(op.bias, a=0, b=2*math.pi)
            op.weight.requires_grad = self.backpropagation
            op.bias.requires_grad = False

    def forward(self, x):
        x = torch.flatten(x, 1) 
        for i, op in enumerate(self.op_list):
            x = np.sqrt(2 / self.out_dimension[i]) * torch.cos(op(x))
        feature_mapping = x
        outputs = self.fc_out(x)
        return [feature_mapping, outputs]

class CRFFNet(nn.Module):
    def __init__(self, num_dimension, num_class, in_channels, out_channels, num_layers, kernel_size, sigma, backpropagation=True):
        super().__init__()
        self.out_dimension = out_channels * num_dimension
        self.sigma = sigma
        self.num_layers = num_layers
        self.backpropagation = backpropagation
        _padding = int((kernel_size-1)/2)
        
        self.op_list = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=_padding)]
        for _ in range(num_layers - 1):
            op_tmp = nn.Conv2d(out_channels, out_channels, kernel_size, padding=_padding)
            self.op_list.append(op_tmp)
        self.op_list = nn.ModuleList(self.op_list)
        self.fc_out = nn.Linear(self.out_dimension, num_class)
        self.init_weights()

    def init_weights(self):
        for op in self.op_list:
            nn.init.normal_(op.weight, 0, self.sigma)
            nn.init.uniform_(op.bias, a=0, b=2*math.pi)
            op.weight.requires_grad = self.backpropagation
            op.bias.requires_grad = False

    def forward(self, x):
        for op in self.op_list:
            x = np.sqrt(2 / self.out_dimension) * torch.cos(op(x))

        x = torch.flatten(x, 1) 
        feature_mapping = x
        outputs = self.fc_out(x)
        return [feature_mapping, outputs]

class DSKN(nn.Module):
    def __init__(self, num_dimension, out_dimension, num_class, sigma0, sigma1, growth_factor, backpropagation=True):
        super().__init__()
        self.out_dimension = out_dimension
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.backpropagation = backpropagation
        self.growth_factor = growth_factor

        self.op_list0 = [nn.Linear(num_dimension, out_dimension[0])]
        self.op_list1 = [nn.Linear(num_dimension, out_dimension[0])]
        for i in range(1, len(out_dimension)):
            op_tmp0 = nn.Linear(out_dimension[i-1], out_dimension[i])
            op_tmp1 = nn.Linear(out_dimension[i-1], out_dimension[i])
            self.op_list0.append(op_tmp0)
            self.op_list1.append(op_tmp1)
        self.op_list0 = nn.ModuleList(self.op_list0)
        self.op_list1 = nn.ModuleList(self.op_list1)
        self.fc_out = nn.Linear(out_dimension[-1], num_class)
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.op_list0)):
            nn.init.normal_(self.op_list0[i].weight, 0, self.sigma0 * math.pow(self.growth_factor, i))
            nn.init.normal_(self.op_list1[i].weight, 0, self.sigma1 * math.pow(self.growth_factor, i))
            nn.init.uniform_(self.op_list0[i].bias, a=0, b=2*math.pi)
            self.op_list0[i].weight.requires_grad = self.backpropagation
            self.op_list1[i].weight.requires_grad = self.backpropagation
            self.op_list0[i].bias.requires_grad = False
            self.op_list1[i].bias = self.op_list0[i].bias

    def forward(self, x):
        x = torch.flatten(x, 1) 
        for i in range(len(self.op_list0)):
            x = 1 / np.sqrt(2 * self.out_dimension[i]) * (torch.cos(self.op_list0[i](x)) + torch.cos(self.op_list1[i](x)))
        feature_mapping = x
        outputs = self.fc_out(x)
        return [feature_mapping, outputs]

class CSKN(nn.Module):
    def __init__(self, num_dimension, num_class, in_channels, out_channels, num_layers, kernel_size, sigma0, sigma1, growth_factor, backpropagation=True):
        super().__init__()
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.num_layers = num_layers
        self.growth_factor = growth_factor
        self.backpropagation = backpropagation
        self.out_dimension = out_channels * num_dimension
        _padding = int((kernel_size-1)/2)

        self.op_list0 = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=_padding)]
        self.op_list1 = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=_padding)]
        for _ in range(num_layers - 1):
            op_tmp0 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=_padding)
            op_tmp1 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=_padding)
            self.op_list0.append(op_tmp0)
            self.op_list1.append(op_tmp1)
        self.op_list0 = nn.ModuleList(self.op_list0)
        self.op_list1 = nn.ModuleList(self.op_list1)
        self.fc_out = nn.Linear(self.out_dimension, num_class)
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.op_list0)):
            nn.init.normal_(self.op_list0[i].weight, 0, self.sigma0 * math.pow(self.growth_factor, i))
            nn.init.normal_(self.op_list1[i].weight, 0, self.sigma1 * math.pow(self.growth_factor, i))
            nn.init.uniform_(self.op_list0[i].bias, a=0, b=2*math.pi)
            self.op_list0[i].weight.requires_grad = self.backpropagation
            self.op_list1[i].weight.requires_grad = self.backpropagation
            self.op_list0[i].bias.requires_grad = False
            self.op_list1[i].bias = self.op_list0[i].bias

    def forward(self, x):
        for i in range(len(self.op_list0)):
            x = 1 / np.sqrt(2 * self.out_dimension) * (torch.cos(self.op_list0[i](x)) + torch.cos(self.op_list1[i](x)))
        x = torch.flatten(x, 1) 
        feature_mapping = x
        outputs = self.fc_out(x)
        return [feature_mapping, outputs]

### [3, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
class CRFFNet8(nn.Module):
    def __init__(self, num_dimension, num_class, in_channels, sigma0, kernel_size=3, out_list=[64, 64, 128, 128, 256, 256, 512], pool_dict = {0, 2, 4, 6}):
        super().__init__()
        self.sigma0 = sigma0
        self.out_list = out_list
        self.pool_dict = pool_dict

        self.op_list = [nn.Conv2d(in_channels, out_list[0], kernel_size, padding=1)]
        self.bn_list = [nn.BatchNorm2d(out_list[0])]
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        for i in range(1, len(out_list)):
            op_tmp = nn.Conv2d(out_list[i-1], out_list[i], kernel_size, padding=1)
            self.op_list.append(op_tmp)
            self.bn_list.append(nn.BatchNorm2d(out_list[i]))
        
        self.op_list = nn.ModuleList(self.op_list)
        self.bn_list = nn.ModuleList(self.bn_list)
        self.fc_out = nn.Linear(num_dimension, num_class)
        
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.op_list)):
            nn.init.normal_(self.op_list[i].weight, 0, self.sigma0)
            nn.init.uniform_(self.op_list[i].bias, a=0, b=2*math.pi)
            self.op_list[i].bias.requires_grad = False

    def forward(self, x):
        for i in range(len(self.op_list)):
            x =  torch.cos(self.op_list[i](x))
            x = self.bn_list[i](x)
            if i in self.pool_dict:
                x = self.max_pool(x)
        x = self.avg_pool(x)
        
        feature_mapping = torch.flatten(x, 1) 
        outputs = self.fc_out(feature_mapping)
        return [feature_mapping, outputs]

### [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
class CSKN8(nn.Module):
    def __init__(self, num_dimension, num_class, in_channels, sigma0, sigma1, growth_factor, kernel_size=3, out_list=[64, 64, 128, 128, 256, 256, 512], pool_dict = {0, 2, 4, 6}):
        super().__init__()
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.growth_factor = growth_factor
        self.out_list = out_list
        self.pool_dict = pool_dict

        self.op_list0 = [nn.Conv2d(in_channels, out_list[0], kernel_size, padding=1)]
        self.op_list1 = [nn.Conv2d(in_channels, out_list[0], kernel_size, padding=1)]
        self.bn_list = [nn.BatchNorm2d(out_list[0])]
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        for i in range(1, len(out_list)):
            op_tmp0 = nn.Conv2d(out_list[i-1], out_list[i], kernel_size, padding=1)
            op_tmp1 = nn.Conv2d(out_list[i-1], out_list[i], kernel_size, padding=1)
            self.op_list0.append(op_tmp0)
            self.op_list1.append(op_tmp1)
            self.bn_list.append(nn.BatchNorm2d(out_list[i]))
        
        self.op_list0 = nn.ModuleList(self.op_list0)
        self.op_list1 = nn.ModuleList(self.op_list1)
        self.bn_list = nn.ModuleList(self.bn_list)
        self.fc_out = nn.Linear(num_dimension, num_class)
        
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.op_list0)):
            nn.init.normal_(self.op_list0[i].weight, 0, self.sigma0 * math.pow(self.growth_factor, i))
            nn.init.normal_(self.op_list1[i].weight, 0, self.sigma1 * math.pow(self.growth_factor, i))
            nn.init.uniform_(self.op_list0[i].bias, a=0, b=2*math.pi)
            self.op_list0[i].bias.requires_grad = False
            self.op_list1[i].bias = self.op_list0[i].bias

    def forward(self, x):
        for i in range(len(self.op_list0)):
            x =  torch.cos(self.op_list0[i](x)) + torch.cos(self.op_list1[i](x))
            x = self.bn_list[i](x)
            if i in self.pool_dict:
                x = self.max_pool(x)
        x = self.avg_pool(x)
        
        feature_mapping = torch.flatten(x, 1) 
        outputs = self.fc_out(feature_mapping)
        return [feature_mapping, outputs]

def Gaussian_kernel(X1, X2, sigma):
    X_row = torch.sum(X1**2, 1).reshape(-1, 1)
    X_col = torch.sum(X2**2, 1).reshape(1, -1)
    return torch.exp(- (X_row.repeat(1, len(X_col)) + X_col.repeat(len(X_row), 1) - 2 * X1.mm(X2.T)) * np.power(sigma, 2) / 2)


def train(logger, args, model, device, criterion, train_loader, optimizer_arr, epoch, additional_regularizers = False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        for optimizer in optimizer_arr:
            optimizer.zero_grad()

        feature_mapping, outputs = model(data)
        if additional_regularizers:
            reg_weights = args['lambda_0'] * torch.pow(torch.norm(model.fc_out.weight, 'nuc'), 2)
            reg_features = args['lambda_1'] * torch.pow(torch.norm(feature_mapping, 'fro'), 2)
        else:
            reg_weights = torch.zeros(1, device=device)
            reg_features = torch.zeros(1, device=device)

        loss = criterion(outputs, target)
        obj = loss + reg_weights + reg_features

        obj.backward()
        for optimizer in optimizer_arr:
            optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tTrace norm: {:.6f} \tFrobenius norm: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), reg_weights.item(), reg_features.item()))
    return [loss.item(), reg_weights.item(), reg_features.item()]

def test(logger, args, model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy, test_loss