import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
import matplotlib.pyplot as plt
from models import VanillaCNNNet, DNN, DSKN, CSKN
from models import train, test

torch.manual_seed(1)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('mnist_non-stationary')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=128, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1000, shuffle=True, **kwargs)

num_layers = 5
sigma0 = 1e-2
sigma1 = 1e-2
lr0 = 1e-4
lr1 = 1e-4
# growth_factor = [None, 1, 32, 16, 16, 8]
growth_factor = 8
model = ASKL(784, [2000 for _ in range(num_layers)], 10, sigma0, sigma1, growth_factor, True).to(device)
optimizers = [optim.Adam(model.op_list0.parameters(), lr0), optim.Adam(model.op_list1.parameters(), lr0), optim.Adam(model.fc_out.parameters(), lr1)]


intial_weight = model.state_dict()
checkpoint = {'intial_weight': intial_weight}
torch.save(checkpoint, './results/exp_ablation1_{}layer_initial.pt'.format(num_layers))

args={'lambda_0': 0, 'lambda_1': 0, 'log_interval': 1000}
criterion = nn.CrossEntropyLoss()
SKN_accuracy = []
for epoch in range(10):
    train(logger, args, model, device, criterion, train_loader, optimizers, epoch)
    test_acc, _ = test(logger, args, model, device, criterion, test_loader)
    SKN_accuracy.append(test_acc)
print('test accuracy {:.2f} %'.format(test_acc))

final_weight = model.state_dict()
checkpoint = {'intial_weight': intial_weight, 'final_weight': final_weight, 'SKN_accuracy': SKN_accuracy}

torch.save(checkpoint, './results/exp_ablation1_{}layer_final.pt'.format(num_layers))