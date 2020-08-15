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
    testset, batch_size=100, shuffle=False, num_workers=2)

net = ResNet18().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
max_epochs = 350

def test (net,testloader):
    net.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()*targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        test_loss /= total
        correct /= total
        print ("Avg Loss : %f,Acc : %f"%(test_loss,correct))
    return test_loss,correct

if (args.test):
    if (args.model_file == None):
        exit()
    net.load_state_dict(torch.load(args.model_file))
    test(net,testloader)
    
else:
    predictions = torch.from_numpy(np.zeros(len(trainset))).float().cuda()
    num_tries = torch.from_numpy(np.zeros(len(trainset))).float().cuda()
    step = 0
    writer = SummaryWriter(log_dir = 'runs/run1')
    for epoch in range(max_epochs):
        if (epoch == 40):
            optimizer = optim.SGD(net.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
        if (epoch == 100):
            optimizer = optim.SGD(net.parameters(), lr=0.001,momentum=0.9, weight_decay=5e-4)
            
        print ("Starting Epoch : %d"%epoch)
        shuff = torch.from_numpy(np.random.permutation(np.arange(len(trainset))))
        net.train()
        for i in range(0,len(trainset),batch_size):
            batch_ind = shuff[i:min(i+batch_size,len(trainset))]
            transformed_trainset = []
            for ind in batch_ind:
                transformed_trainset.append(trainset.__getitem__(ind)[0])
            inputs = torch.stack(transformed_trainset)
            targets = torch.LongTensor(np.array(trainset.train_labels)[batch_ind].tolist())
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            old_predictions = predictions[batch_ind]
            new_predictions = predicted.eq(targets).float()
            diff_pred = old_predictions - new_predictions
            num_tries[batch_ind[(diff_pred > 0).nonzero()]] += 1
            predictions[batch_ind] = new_predictions
            
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            writer.add_scalar('iteration/loss',loss,step)
            step += 1
        loss,acc = test(net,testloader)
        writer.add_scalar('validation/loss',loss,step)
        writer.add_scalar('validation/acc',acc,step)
        np.save('stats/num_forget.npy',num_tries.cpu().numpy())    
