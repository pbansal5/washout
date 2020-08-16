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
import argparse

from model import ResNet18

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model-file', '-m', type=str,help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true',help='placeholder')
args = parser.parse_args()

dataDir = '../data/'
checkpointDir = '../checkpoints/'
batch_size = 32

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=dataDir, train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root=dataDir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=True)

net = ResNet18().cuda()

def compute_gradient_avg (unforgetable):

    # return unit norm avg_grad
    
    criterion_grad = nn.CrossEntropyLoss(reduction='sum')
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        net(inputs).mean().backward()
        break
    grad = []
    for x in net.parameters():
        grad.append(x.grad.view(-1))
    grad = torch.zeros(torch.cat(grad).shape).cuda()

    net.train()
    for i in range(0,len(unforgetable),batch_size):
        grad_ = []
        batch_ind = unforgetable[i:min(i+batch_size,len(unforgetable))]
        transformed_trainset = []
        for ind in batch_ind:
            transformed_trainset.append(trainset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_trainset)
        targets = torch.LongTensor(np.array(trainset.train_labels)[batch_ind].tolist())
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        optimizer.zero_grad()
        loss = criterion_grad(outputs, targets)
        loss.backward()
        for x in net.parameters():
            grad_.append(x.grad.view(-1))
        grad += torch.cat(grad_)
    grad /= unforgetable.shape[0]
    return grad/torch.norm(grad)


array = np.load('stats/num_forget.npy')
unforgetable = np.where(array<=1)[0]
forgetable_examples = np.where(array>1)[0]
avg_grad = compute_gradient_avg(unforgetable)

def train(forgetable_examples,avg_grad):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
    max_epochs = 350
    step = 0
    writer = SummaryWriter(log_dir = 'runs/run1')
    for epoch in range(max_epochs):
        if (epoch == 40):
            optimizer = optim.SGD(net.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
        if (epoch == 100):
            optimizer = optim.SGD(net.parameters(), lr=0.001,momentum=0.9, weight_decay=5e-4)
            
        print ("Starting Epoch : %d"%epoch)
        shuff = torch.from_numpy(np.random.permutation(forgetable_examples))
        net.train()
        for i in range(0,len(forgetable_examples),batch_size):
            batch_ind = shuff[i:min(i+batch_size,len(forgetable_examples))]
            transformed_trainset = []
            for ind in batch_ind:
                transformed_trainset.append(trainset.__getitem__(ind)[0])
            inputs = torch.stack(transformed_trainset)
            targets = torch.LongTensor(np.array(trainset.train_labels)[batch_ind].tolist())
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            ############# DO SHIT ##############
            grad_ = []
            for x in net.parameters():
                grad_.append(x.grad.view(-1))
            grad_ = torch.cat(grad_)
            grad_ -= torch.dot(grad_,avg_grad)*avg_grad
            prev_index = 0
            for x in net.parameters():
                x.grad = grad_[prev_index:prev_index+x.grad.size()]
            assert prev_index = grad_.shape[0]
            ####################################
            optimizer.step()
            writer.add_scalar('iteration/loss',loss,step)
            step += 1
        loss,acc = test(net,testloader)
        writer.add_scalar('validation/loss',loss,step)
        writer.add_scalar('validation/acc',acc,step)


train(forgetable_examples,avg_grad)
