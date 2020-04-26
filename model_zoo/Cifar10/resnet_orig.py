import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Access root folder
sys.path.append(BASE_DIR)
from modules import *

#Credit to https://github.com/akamaster/pytorch_resnet_cifar10

__all__ = ['resnet_orig', 'create_resnet_orig_cifar10']

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                This line is different from distiller's cifar10 model configuration.
                LamdaLayer may be replaced by original torch
                """
                #self.shortcut = LambdaLayer(lambda x:
                #                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
                self.shortcut = LambdaLayer(lambda x:
                                          nn.ConstantPad3d((0, 0, 0, 0, planes//4, planes//4), 0)(x[:, :, ::2, ::2]))
                #self.shortcut = nn.Sequential(
                #    nn.ConstantPad3d((0, 0, 0, 0, planes//4, planes//4), 0) 
                #)
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
        self.add = EltwiseAdd()
        self.relu2 = nn.ReLU(inplace=False)
        
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #out += self.shortcut(x)
        out = self.add(out, self.shortcut(x))
        out = self.relu2(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        #self.avg_pool2d = 
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #print(out.size()[3])
        #out = F.avg_pool2d(out, out.size()[3])
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet_orig(pretrained=True, device='cpu'):
    net = ResNet(BasicBlock, [3, 3, 3])
    if pretrained:
        #script_dir = os.path.dirname(__file__)
        #state_dict = torch.load(script_dir + '/state_dicts/resnet_orig.pt', map_location=device)
        state_dict = torch.load('/state_dicts/resnet_orig.pt', map_location=device)
        net.load_state_dict(state_dict)
    return net

def create_resnet_orig_cifar10(arch, pretrained, device=None):
    if arch == 'resnet_orig':
        model = resnet_orig(pretrained=pretrained, device=device)
    else:
        raise ValueError('Not support this kind of resnet_orig model !!!')
    return model


if __name__ == "__main__":
    model = resnet_orig(False)
    print(model)
    c = model.forward(torch.ones(1,3,32,32))
    print(c)
    