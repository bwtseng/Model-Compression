
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import pyvww
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
import torchnet.meter as tnt
import classifier as cls
import checkpoint as ckpt
import performance_tracker as pt
import math
from collections import OrderedDict
import policy





dataloaders = {}
dataset_sizes = {}
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = pyvww.pytorch.VisualWakeWordsClassification(root="/home/bwtseng/Downloads/visualwakewords/coco/all", annFile="/home/bwtseng/Downloads/visualwakewords/vww_datasets/annotations/instances_train.json",
                    transform=data_transforms['train'])
dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_dataset = pyvww.pytorch.VisualWakeWordsClassification(root="/home/bwtseng/Downloads/visualwakewords/coco/all", annFile="/home/bwtseng/Downloads/visualwakewords/vww_datasets/annotations/instances_val.json",
                transform=data_transforms['val'])
dataloaders['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

dataset_sizes['train'] = len(train_dataset)
dataset_sizes['val'] = len(val_dataset)

def train(args, model, criterion, optimizer, device, model_path, tracker, lr_scheduler, epoch=25):

    total_samples = dataset_sizes["train"]
    batch_size = dataloaders["train"].batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)    
    

    OVERALL_LOSS_KEY = 'Overall Loss'
    OBJECTIVE_LOSS_KEY = 'Objective Loss'

    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
    #model.train()
    #end = time.time()

    for epo in range(epoch):

        classerr = tnt.ClassErrorMeter(accuracy=True, topk=[1])
        batch_time = tnt.AverageValueMeter()
        data_time = tnt.AverageValueMeter()
        end = time.time()
        model.train()
        acc_stats = []
        for train_step, data in enumerate(dataloaders['train'], 0):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            classerr.add(output.detach(), labels)
            acc_stats.append(classerr.value(1))
            losses[OBJECTIVE_LOSS_KEY].add(loss.item())
            
            # *****************************
            # Training step
            # *****************************

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_time.add(time.time()-end)
            steps_completed = (train_step+1)

            if steps_completed % 30 == 0 :
                print('Epoch: [{}][{:5d}/{:5d}]  \033[0;37;41mObjective Loss {:.5f}\033[0m'
                        '\033[0;37;42m\tTop 1 {:.5f}  \033[0m' 
                        '\033[0;37;40m\tLR {:.5f}  Time {:.5f}\033[0m.'.format(epo, 
                                                    steps_completed, 
                                                    int(steps_per_epoch), 
                                                    losses['Objective Loss'].mean, 
                                                    classerr.value(1),
                                                    optimizer.param_groups[0]['lr'], 
                                                    batch_time.mean))    

            end = time.time()
        
        _validate(model, criterion, device)

        tracker.step(model, epoch, top1=classerr.value(1))
        best_score = tracker.best_scores()[0]
        is_best = epoch == best_score.epoch
        checkpoint_extras = {'current_top1': classerr.value(1),
                                'best_top1': best_score.top1,
                                'best_epoch': best_score.epoch}
        ckpt.save_checkpoint(args.epoch, args.arch, model, optimizer=optimizer, is_best=is_best,
                                name=args.name, dir = args.model_path)
        lr_scheduler.step()

    return classerr.value(1), losses['Objective Loss'].mean

def _validate(model, criterion, device):
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=[1])
    losses = {'objective_loss': tnt.AverageValueMeter()}
    batch_time = tnt.AverageValueMeter()
    total_samples = len(dataloaders["val"].sampler)
    batch_size = dataloaders["val"].batch_size
    total_steps = total_samples / batch_size

    # *******************
    # Evaluation
    # *******************
    model.eval()
    end = time.time()
    with torch.no_grad():
        for validation_step, data in enumerate(dataloaders["val"]):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            output = model(inputs)

            loss = criterion(output, labels)
            losses['objective_loss'].add(loss.item())
            classerr.add(output.detach(), labels)
            steps_completed = (validation_step+1)
            
            batch_time.add(time.time() - end)
            end = time.time()  
            steps_completed = (validation_step+1)
            #Record log using _log_validation_progress function 
            if steps_completed % 30 == 0 : 
                print('Test [{:5d}/{:5d}] \033[0;37;41mLoss {:.5f}\033[0'  
                        '\033[0;37;42m\tTop1 {:.5f}\033[m'  
                        '\tTime {:.5f}.'.format(steps_completed, 
                                            int(total_steps), 
                                            losses['objective_loss'].mean,  
                                            classerr.value(1), 
                                            batch_time.mean))


        print('==> \033[0;37;42mTop1 {:.5f} \033[m'   
            '\033[0;37;41m\tLoss: {:.5f}\n\033[m.'.format(
                       classerr.value(1), losses['objective_loss'].mean)) 
    return classerr.value(1), losses['objective_loss'].mean


def test(model, criterion, device):
    print('--- test ---------------------')
    acc, loss = _validate(model, criterion, device)
    return acc, loss

"""
def train_model(model, criterion, optimizer, scheduler, device, model_path, num_epochs=25):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, data in enumerate(dataloaders[phase], 0):
                inputs = data[0].to(device)
                labels = data[1].to(device)

                # zero the parameter gradients (Initialize)
                optimizer.zero_grad()

                # track history during trainig/validation phase
                # This line is to calculate performance during training and doesn't need to compute gradient here.
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # Training basic code here
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # Calculating Loss and Correct number
                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)
            # Get average value:
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                name = 'model_{:03d}.pth'.format(epoch)
                print ("Save model name: {}.".format(name))
                torch.save(model.state_dict(), model_path+name)

            # deep copy the model (checkpoint saved (metrics: Accuracy))
            if phase == 'val' and epoch_acc > best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                if epoch_acc > best_acc:
                    is_best = True
                    best_acc = epoch_acc
                else: 
                    is_best = False

                ## Replace the tar file with the best model so far. This may not be useful.
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': 'mobilenetv1',
                    'state_dict': model.state_dict(),
                    'best_prec1': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, model_path)

        print("\n")

    time_cost = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_cost // 60, time_cost % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
"""



if __name__ == '__main__':

    print("Training set dataloader is prepared: {}.".format(dataloaders['train']))
    print("Validation set dataloader is prepared: {}.".format(dataloaders['val']))
    
    # Some auguments listed here: 
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--learning-rate-decay', '--lrd', default=0.7, type=float,
                    metavar='W', help='learning rate decay (default: 0.7)')

    parser.add_argument('--cpu', default=False, type=bool, help="If GPU is full process.")
    parser.add_argument('--model_path', 
                        default='/home/bwtseng/Downloads/vww_mobilenetv1_distiller/no_pruning_model', 
                        type=str, help='Path to trained model.')
    parser.add_argument('--epoch', default=25, type=int, help="Training epoch u need.")
    parser.add_argument('--arch', default='mobilenetv1', type=str, help='Architecture name.')
    parser.add_argument('--name', default='vww_mobilenet_pure', type=str, help='File name')
    parser.add_argument('--pre_trained', default=False, type=bool, help='Using pretrained model?')
    parser.add_argument('--train', default=True, type=bool, help='Training procedure.')
    parser.add_argument('--test', default=False, type=bool, help='Testing procedure.')
    parser.add_argument('--parallel', default=False, type=bool, help="Parallel or not")
    parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate.")
    parser.add_argument('--num_best_scores', default=1, type=int, help="num_best_score")
    parser.add_argument('--resume_from', default=False, type=bool, help="using the ckpt from local trained model.")
    parser.add_argument('--post_qe_test', default=False, type=bool, help='whether testing with quantization model.')
    args = parser.parse_args()
    print(args)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("You use the {} device to do this task.".format(device))

    if args.train:
        if args.cpu: 
            device = torch.device("cpu")
            net = cls.Net()
            checkpoint = torch.load('mobilenet_sgd_rmsprop_69.526.pth', map_location=lambda storage, loc: storage)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            print("Remove module string in loaded model !!!")
            for k, v in checkpoint['state_dict'].items():
                #print(k)
                name = k[7:] # remove `module.` cuase CPU is not able to load module weight feature.
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
            num_ftrs = net.fc.in_features
            print(num_ftrs)
            net.fc = nn.Linear(num_ftrs, 2)

        else: 
            net = cls.Net()

            if args.parallel:
                model = torch.nn.DataParallel(model, device_ids=device)
                model.is_parallel = args.parallel
            net.arch = args.arch
            net.dataset = "VWW_dataset"
            #net = torch
            net = torch.nn.DataParallel(net).cuda()
            if args.pre_trained:
                net, compress_scheduler, optimizer, start_epoch = ckpt.load_checkpoint(
                net, "/home/bwtseng/Downloads/mobilenet_sgd_rmsprop_69.526.pth", 
                model_device=device)
                optimizer = None 
    
            else:
                #net = cls.Net()#.to(device)
                pass 
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

        num_ftrs = net.module.fc.in_features
        print("Output dimension from old model:{}.".format(num_ftrs))
        net.module.fc = nn.Linear(num_ftrs, 2)
        net.to(device)
        tracker = pt.SparsityAccuracyTracker(args.num_best_scores)    
        tracker.reset()
        # Setting Learning decay scheduler, that is, decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        #net = train_model(net, criterion, optimizer, exp_lr_scheduler,
        #                    devs, net, criterion, optimizer, device, args.model_path, trackerce, args.model_path, lr_scheduler, num_epochs=80)
        acc, losses = train(args, net, criterion, optimizer, device, args.model_path, tracker,
                            exp_lr_scheduler, args.epoch)
        #torch.save(net.state_dict(), args.model_path+'params_final.pth')
    else:
        net = ckpt.load_lean_checkpoint(net, "/home/bwtseng/Downloads/vww_mobilenetv1_distiller/\
                                    no_pruning_model/vww_mobilenet_pure_best.pth.tar",
                            model_device=device)
        top1, loss = test(net, criterion, device) 

