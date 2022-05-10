import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import logging
from models import DSKN
from models import train, test

torch.manual_seed(1)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mnist_non-stationary')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=128, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1000, shuffle=True, **kwargs)

sigma0 = 1e-2
sigma1 = 1e-6
lr0 = 1e-3
lr1 = 1e-3
model = DSKN(784, [2000], 10, sigma0, sigma1, 1, True).to(device)
optimizers = [optim.Adam(model.op_list0.parameters(), lr0), optim.Adam(model.op_list1.parameters(), lr0), optim.Adam(model.fc_out.parameters(), lr1)]

# record initilized spectral density
with torch.no_grad():
    RFFNet_init_weight0 = model.op_list0[0].weight.cpu().detach().numpy()
    RFFNet_init_weight1 = model.op_list1[0].weight.cpu().detach().numpy()

args={'lambda_0': 0, 'lambda_1': 0, 'log_interval': 1000}
criterion = nn.CrossEntropyLoss()
DSKN_accuracy = []
for epoch in range(30):
    train(logger, args, model, device, criterion, train_loader, optimizers, epoch)
    test_acc, _ = test(logger, args, model, device, criterion, test_loader)
    DSKN_accuracy.append(test_acc)
print('test accuracy {:.2f} %'.format(test_acc))

# record optimized spectral density
with torch.no_grad():
    RFFNet_final_weight0 = model.op_list0[0].weight.cpu().detach().numpy()
    RFFNet_final_weight1 = model.op_list1[0].weight.cpu().detach().numpy()

checkpoint = {'model': model, 'optimizer': optimizers, 'DSKN_accuracy': DSKN_accuracy, 'weights0': [RFFNet_init_weight0, RFFNet_final_weight0], 'weights1': [RFFNet_init_weight1, RFFNet_final_weight1]}

torch.save(checkpoint, './results/exp_ablation0_DSKN.pt')