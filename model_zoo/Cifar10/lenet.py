'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['LeNet_adv', 'LeNet', 'create_lenet_cifar10']

class LeNet_adv(nn.Module):
    def __init__(self, w=1):
        super(LeNet_adv, self).__init__()
        self.w = int(w)        
        self.conv1 = nn.Conv2d(3, 6*self.w, 5)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6*self.w, 16*self.w, 5)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1   = nn.Linear(16*5*5*self.w, 120*self.w)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2   = nn.Linear(120*self.w, 84*self.w)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3   = nn.Linear(84*self.w, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.pool1(out)
        out = self.relu2(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNet(nn.Module):
    def __init__(self, w=1):
        super(LeNet, self).__init__()
        self.w = int(w)        
        self.conv1 = nn.Conv2d(3, 6*self.w, 5)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6*self.w, 16*self.w, 5)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1   = nn.Linear(16*5*5*self.w, 120*self.w)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2   = nn.Linear(120*self.w, 84*self.w)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc3   = nn.Linear(84*self.w, 10)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.pool1(out)
        out = self.relu2(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out

"""
class LeNet_adv(nn.Module):
    def __init__(self,w = 1):
        super(LeNet_adv, self).__init__()
        self.w = int(w)        
        self.conv1 = nn.Conv2d(3, 6*self.w, 5)
        self.conv2 = nn.Conv2d(6*self.w, 16*self.w, 5)
        self.fc1   = nn.Linear(16*5*5*self.w, 120*self.w)
        self.fc2   = nn.Linear(120*self.w, 84*self.w)
        self.fc3   = nn.Linear(84*self.w, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class LeNet(nn.Module):
    def __init__(self,w = 1):
        super(LeNet, self).__init__()
        self.w = int(w)        
        self.conv1 = nn.Conv2d(3, 6*self.w, 5)
        self.conv2 = nn.Conv2d(6*self.w, 16*self.w, 5)
        self.fc1   = nn.Linear(16*5*5*self.w, 120*self.w)
        self.fc2   = nn.Linear(120*self.w, 84*self.w)
        self.fc3   = nn.Linear(84*self.w, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
    
"""


def create_lenet_cifar10(arch, pretrained, width_mult=8.0, device=None):
    if arch == 'lenet':
        model = LeNet(w=width_mult)
        if pretrained:
            pass
    elif arch =='lenet_v2':
        model = LeNet_adv(w=width_mult)
        if pretrained:
            pass
    else:
        raise ValueError('Not support this kind of LeNet model !!!')
    return model