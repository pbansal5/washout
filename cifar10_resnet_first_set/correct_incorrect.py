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

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--model-file', '-m', type=str,help='resume from checkpoint')
parser.add_argument('--test', '-t', action='store_true',help='placeholder')
parser.add_argument('--orthogonal', '-o', action='store_true',help='placeholder')

args = parser.parse_args()
dataDir = '../data/'
checkpointDir = '../checkpoints/'
batch_size = 128
save_every= 10

random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)


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
optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
checkpoint = torch.load(os.path.join(checkpointDir, 'forget_ckpt.pth'))
start_epoch = int(checkpoint['epoch'])+1
print ("Path : %s, Epoch : %d, Acc : %f" %(os.path.join(checkpointDir, 'forget_ckpt.pth'),start_epoch-1,checkpoint['acc']))
net.load_state_dict(checkpoint['net'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
criterion = nn.CrossEntropyLoss()

def compute_correct_incorrect():
    net.eval()
    correct_ones = torch.from_numpy(np.full(len(trainset), False)).cuda()
    with torch.no_grad():
        for i in range(0,len(trainset),batch_size):
            batch_ind = torch.from_numpy(np.arange(i, min(i+batch_size,len(trainset))))
            transformed_trainset=[]
            for ind in batch_ind:
                transformed_trainset.append(trainset.__getitem__(ind)[0])
            inputs = torch.stack(transformed_trainset)
            targets = torch.LongTensor(np.array(trainset.targets)[batch_ind].tolist())
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            correct= predicted.eq(targets)
            correct_ones[batch_ind[correct]] = True
    return correct_ones.cpu().numpy()

def compute_gradient_avg (unforgetable):
    global optimizer
    global net
    # return unit norm avg_grad
    
    criterion_grad = nn.CrossEntropyLoss(reduction='sum')
    net.train()
    for i in range(0,len(unforgetable),batch_size):
        batch_ind = unforgetable[i:min(i+batch_size,len(unforgetable))]
        transformed_trainset = []
        for ind in batch_ind:
            transformed_trainset.append(trainset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_trainset)
        targets = torch.LongTensor(np.array(trainset.targets)[batch_ind].tolist())
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion_grad(outputs, targets)
        loss.backward()

    grad_ = []
    for x in net.parameters():
        grad_.append(x.grad.view(-1))
    grad = torch.cat(grad_)
    grad /= unforgetable.shape[0]
    return grad/torch.norm(grad)



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

def train(orthogonal):
    global optimizer
    global net
    best_acc = 0
    max_epochs = 350
    step = 0
    writer = SummaryWriter(log_dir = 'runs/run_correct_incorrect')
    for epoch in range(start_epoch,max_epochs):
        array = compute_correct_incorrect()
        unforgetable = np.where(array)[0]
        forgetable_examples = np.where(~array)[0]
        print("Beginning computation for Average Gradient")
        avg_grad = compute_gradient_avg(unforgetable)
        print("Computed Average Gradient")

        if (epoch == 30):
            optimizer = optim.SGD(net.parameters(), lr=0.01,momentum=0.9, weight_decay=5e-4)
        if (epoch == 50):
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
            targets = torch.LongTensor(np.array(trainset.targets)[batch_ind].tolist())
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            ############# DO ORTHOGONAL STUFF ##############
            if (orthogonal) : 
                grad_ = []
                for x in net.parameters():
                    grad_.append(x.grad.view(-1))
                grad_ = torch.cat(grad_)
                grad_ -= torch.dot(grad_,avg_grad)*avg_grad
                prev_index = 0
                for x in net.parameters():
                    num_elem= torch.numel(x.grad)
                    x.grad = grad_[prev_index:(prev_index+num_elem)].view(x.grad.size()).clone().detach_()
                    prev_index+=num_elem
                assert prev_index == grad_.shape[0]
            ####################################
            optimizer.step()
            writer.add_scalar('iteration/loss',loss,step)
            step += 1
        loss,acc = test(net,testloader)
        writer.add_scalar('validation/loss',loss,step)
        writer.add_scalar('validation/acc',acc,step)
        if (epoch%save_every)==0:
            if acc > best_acc:
                if not os.path.isdir(checkpointDir):
                    os.mkdir(checkpointDir)
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(checkpointDir,'pgrad_correct_incorrect_ckpt.pth'))
                best_acc = acc


train(args.orthogonal)
