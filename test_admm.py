import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from numpy import linalg as LA
import datetime
from tensorboardX import SummaryWriter
import scipy.misc 
from pruning import admm_pruner
import model_zoo as mz
from scheduler import ParameterMasker, create_model_masks_dict
import time 
#def __init__(self, name, class_name, pruning_ratio, rho, sparsity_type, masked_progressive, 
#                 admm_epoch, initial_lr, multi_rho, model):

_prune_ratio = {
    #"conv1.weight: #  0.75
    "conv2.weight": 0.75,
    "conv3.weight": 0.75,
    "conv4.weight": 0.75,
    "conv5.weight": 0.75,
    "conv6.weight": 0.75,
    "conv7.weight": 0.75,
    "conv8.weight": 0.75,
    "conv9.weight": 0.75,
    "conv10.weight": 0.75,
    "conv11.weight": 0.75,
    "conv12.weight": 0.75,
    "conv13.weight": 0.75,
    "conv14.weight": 0.75,
    "conv15.weight": 0.75,
    "conv16.weight": 0.75,
    "conv17.weight": 0.75,
    "conv18.weight": 0.75,
    "conv19.weight": 0.75,
    "conv20.weight": 0.75,
    "conv21.weight": 0.75,
    "conv22.weight": 0.75,
    "conv23.weight": 0.75,
    "conv24.weight": 0.75,
    "conv25.weight": 0.75,
    "conv26.weight": 0.75,
    "conv27.weight": 0.75,
    "fc1.weight": 0.75,
}

prune_ratios = {}
conv_names = []
bn_names = []
fc_names = []
name_encoder = {}

def prepare_pruning(model):
    _extract_layer_names(model)
    
    for good_name, ratio in _prune_ratio.items():
        _encode(good_name)

    for good_name,ratio in _prune_ratio.items():
        prune_ratios[name_encoder[good_name]] = ratio
    #for k in self.prune_ratios.keys():
    #    self.rhos[k] = rho  # this version we assume all rhos are equal
    print ('<========={} conv names'.format(len(conv_names)))
    print (conv_names)
    print ('<========={} bn names'.format(len(bn_names)))
    print (bn_names)
    print ('<========={} targeted pruned layers'.format(len(prune_ratios)))
    print (prune_ratios.keys())
    for k, v in prune_ratios.items():
        print ('target sparsity in {} is {}'.format(k,v))

def _extract_layer_names(model):#, conv_names, bn_names, fc_names):
    """
    Store layer name of different types in arrays for indexing
    """
    for name, W in model.named_modules():             
        name += '.weight'  # name in named_modules looks like module.features.0. We add .weight into it
        print(name)
        # This is only for the CNN compression usage.
        if isinstance(W,nn.Conv2d):
            conv_names.append(name)
        if isinstance(W,nn.BatchNorm2d):
            bn_names.append(name)
        if isinstance(W,nn.Linear):
            fc_names.append(name)


def _encode(name):
    """
    Examples:
    conv1.weight -> conv           1                weight
                    conv1-> prefix   weight->postfix        
                    conv->layer_type  1-> layer_id + 1  weight-> postfix
    Use buffer for efficient look up  
    """
    prefix,postfix = name.split('.')
    dot_position = prefix.find('.')
    layer_id = ''
    for s in prefix:
        if s.isdigit():
            layer_id+=s
    id_length = len(layer_id)         
    layer_type = prefix[:-id_length]
    layer_id = int(layer_id)-1
    if layer_type =='conv' and len(conv_names)!=0:
        name_encoder[name] = conv_names[layer_id]
    elif layer_type =='fc' and len(fc_names)!=0:
        name_encoder[name] =  fc_names[layer_id]
    elif layer_type =='bn' and len(bn_names)!=0:
        name_encoder[name] =  bn_names[layer_id]             


dataset = 'imagenet'
arch = 'mobilenet_v1'
pretrained = True 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_function_dict = mz.data_function_dict
model_function = model_function_dict[dataset]
model = model_function_dict[dataset]['mobilenet'](arch, pretrained)
#model.to(device)
model = torch.nn.DataParallel(model).cuda()

prepare_pruning(model)
pruner = admm_pruner.ADMMPruner("admm_pruner", "ADMM_puner", pruning_ratio=prune_ratios, rho=0.001, 
                    sparsity_type="channel", masked_progressive=False, admm_epoch=10, 
                    initial_lr=0.01, multi_rho=True, model=model)


traindir = "/home/swai01/imagenet_datasets/raw-data/train"
valdir = "/home/swai01/imagenet_datasets/raw-data/train"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

val_dataset = datasets.ImageFolder(
    valdir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

batch_size = 32
train_sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    num_workers=4, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=4, pin_memory=True)


optimizer = None
optimizer = torch.optim.SGD(model.parameters(), 0.01,
                            momentum=0.9, weight_decay=1e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                      step_size=30*len(train_loader), 
                                      gamma=0.1)

class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss

criterion = CrossEntropyLossMaybeSmooth(smooth_eps=0.0).to(device)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



zeros_mask_dict = create_model_masks_dict(model)
batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()
print_freq = 50
model.train()
for epo in range(5):
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        pruner.admm_adjust_learning_rate(optimizer, epo)
        input = input.to(device)
        target = target.to(device)
        data = input 
        output = model(input)
        ce_loss = criterion(output, target, smooth=False) # Just a cross entropy.
        pruner.admm_update(model, epo, i, zeros_mask_dict)
        
        ce_loss, admm_loss = pruner.append_admm_loss(model, ce_loss, pruner.sparsity_type)
        
        print(pruner.ADMM_Z["module.model.1.3.weight"][:,10,:,:])
        
        for name, W in model.named_parameters():
            if name == "module.model.1.3.weight":
                print(W[:,10,:,:])
        # I have already checked my implementation is correct..
        # the value corresponding to the weightm matrix is same.
        assert 1==2
        mixed_loss = ce_loss + admm_loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(ce_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        optimizer.zero_grad()
        mixed_loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epo, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5)) 
            print ("cross_entropy loss: {}".format(ce_loss))    
            print("Sum of admm loss dict: {}.".format(admm_loss))
            #assert 1 == 2