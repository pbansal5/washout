import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import random
import argparse
from model import ResNet18

os.environ['PYTHONHASHSEED']=str(42) 
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model-file', '-m', type=str,help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true',help='placeholder')
parser.add_argument('--load', '-l', action='store_true', help='Should load from checkpoint?')
parser.add_argument('--max-epoch', '-e', type=int, default=100, help='max_epoch')
args = parser.parse_args()

dataDir = '../data/'
checkpointDir = '../checkpoints/'

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

trainset = torchvision.datasets.CIFAR10(
    root=dataDir, train=True, download=True,transform=transform_test)

net = ResNet18().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
max_epochs = int(args.max_epoch)

assert os.path.isdir(checkpointDir), 'Error: no checkpoint directory found!'
checkpoint = torch.load(os.path.join(checkpointDir, 'garbage.pth'))
net.load_state_dict(checkpoint['net'])

adv_egs = np.load('stats/adv_egs.npy')
perturbation = torch.from_numpy(np.zeros(len(adv_egs))).float().cuda()+np.inf
net.eval()
for idx,i in enumerate(adv_egs):
    transformed_trainset = []
    inputs = trainset.__getitem__(i)[0].unsqueeze(0)
    targets = torch.LongTensor(np.array(trainset.targets)[[i]].tolist())#.unsqueeze(0)
    inputs, targets = inputs.cuda(), targets.cuda()
    inputs_adv = inputs.clone().detach().requires_grad_(False)
    eps = 0.1
    step = 0.005
    reached = False
    count = 0
    while (count < 100):
        inputs_adv_ = inputs_adv.clone().detach().requires_grad_(True)
        outputs = net(inputs_adv_)
        _, predicted = outputs.max(1)
        if (predicted[0] != targets[0]):
            reached = True
            eps = eps*0.8
            step = step*0.8
        loss = criterion(outputs, targets)
        loss.backward()
        inputs_adv += step*inputs_adv_.grad.sign()
        inputs_adv = torch.min(torch.max(inputs_adv,inputs-eps),inputs+eps)
        count += 1
    if (reached):
        perturbation[idx] = eps
        print (idx,eps)
np.save('stats/adv_eps.npy',perturbation.cpu().numpy())

