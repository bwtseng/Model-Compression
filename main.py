import argparse
import os 
import time
import copy
import math
import shutil # In pytorch, it's often used. It's totally different from Tensorflow.
import pyvww # In the case leveraging the visual wake worlds dataset.
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import OrderedDict
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
# Local packages, which are revised from Distiller Github repo.
import utility as utl
import distiller
import torchnet.meter as tnt
import torchvision.datasets as datasets
import performance_tracker as pt
import checkpoint as ckpt
import model_zoo as mz 
import summary 
import sensitivity as sa
from functools import partial
import collector
import quantization 
import logging
from config_logger import *


msglogger = logging.getLogger()

def _init_logger(args, script_dir):
    global msglogger
    if script_dir is None or not hasattr(args, "output_dir") or args.output_dir is None:
    #if script_dir is None or not hasattr(args, "model_path") or args.model_path is None:
        msglogger.logdir = None
        return None

    #if not os.path.exists(args.model_path):
    #   os.makedirs(args.model_path)
    if not os.path.exists(args.output_dir):
       os.makedirs(args.output_dir)
    name = args.name
    if args.name == '':
        name = args.arch + "_" + args.dataset + "_log"
    #msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'),
    #                                     args.name, args.model_path, args.verbose)
    msglogger = config_pylogger(os.path.join(script_dir, 'logging.conf'),
                                             name, args.output_dir, args.verbose)
    # Log various details about the execution environment.  It is sometimes useful
    # to refer to past experiment executions and this information may be useful.
    #apputils.log_execution_env_state(
    #    filter(None, [args.compress, args.qe_stats_file]),  # remove both None and empty strings
    #    msglogger.logdir)

    log_execution_env_state(
        filter(None, [args.compress, args.qe_stats_file]),  # remove both None and empty strings
        msglogger.logdir)
    msglogger.debug("Distiller: %s", distiller.__version__)
    return msglogger.logdir

def sensitivity_analysis(model, criterion, device, num_classes, args, sparsities, logger):
    # This sample application can be invoked to execute Sensitivity Analysis on your
    # model.  The ouptut is saved to CSV and PNG.
    #print("Running sensitivity tests")
    msglogger.info("Running Sensitivity Test (analysis).")

    test_fnc = partial(test, criterion=criterion, device=device, num_classes=num_classes, args=args, loggers=logger)
    which_params = [(param_name, torch.std(param).item()) for param_name, param in model.named_parameters()]
    sensitivity = sa.perform_sensitivity_analysis(model, net_params=which_params, sparsities=sparsities,
                                                  test_func=test_fnc, group=args.sensitivity)
    if not os.path.isdir('sensitivity_analysis'):
        os.mkdir('sensitivity_analysis')

    sa.sensitivities_to_png(sensitivity, os.path.join('sensitivity_analysis', 'sensitivity_'+args.sensitivity+'.png'))
    sa.sensitivities_to_csv(sensitivity, os.path.join('sensitivity_analysis', 'sensitivity_'+args.sensitivity+'.csv'))

def data_processing(dataset, data_dir, batch_size, workers=4, split_ratio=0, data_transforms=None):
    """
    Input: 
        dataset : a string to specify the name of the dataset being used in this task.
        datadir: path/to/your/local/dataset, note that under this folder should split into train/val folder.
                 and image should be seperated to the class forlder according to its class.
        batchsize: argument used in the pytorch dataloader function
    Output:
        train and test dataloader in a dictionary format.
    """
    msglogger.debug("Create dataloaders, including training, testing and vlaidation (if split ratio > 0).")
    dataset = dataset.lower()
    dataloaders = {}
    dataset_sizes = {}
    if dataset == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }
        train_dataset = datasets.CIFAR10(data_dir, train=True, transform=data_transforms['train'])
        test_dataset = datasets.CIFAR10(data_dir, train=False, transform=data_transforms['test'])

    elif dataset == "mnist":
        data_transforms ={
            'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        }
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=data_transforms['train'])
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=data_transforms['test'])

    elif dataset == "vww":
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        # Special case: Add into argparse arguments.
        # VWW should direct to the root="/home/bwtseng/Downloads/visualwakewords/coco/all", annFile="/home/bwtseng/Downloads/visualwakewords/vww_datasets/annotations/instances_train.json"
        # root dir:"/home/bwtseng/Downloads/visualwakewords/coco/all"
        train_dataset = pyvww.pytorch.VisualWakeWordsClassification(root=data_dir, 
                    annFile="/home/bwtseng/Downloads/visualwakewords/vww_datasets/annotations/instances_train.json",
                    transform=data_transforms['train'])
        test_dataset = pyvww.pytorch.VisualWakeWordsClassification(root=data_dir, 
                    annFile="/home/bwtseng/Downloads/visualwakewords/vww_datasets/annotations/instances_val.json",
                    transform=data_transforms['val'])

    elif dataset == "fashion_mnist":
        data_transforms ={
            'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]),
            'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        }
        # Special case: Add into argparse arguments.
        # VWW should direct to the root="/home/bwtseng/Downloads/visualwakewords/coco/all", annFile="/home/bwtseng/Downloads/visualwakewords/vww_datasets/annotations/instances_train.json"
        # root dir:"/home/bwtseng/Downloads/visualwakewords/coco/all"
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=data_transforms['train'])
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=data_transforms['test'])

    else:
        try:
            #/home/swai01/imagenet_datasets/raw-data
            train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), 
                                                transform=data_transforms['train'])
            test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), 
                                                transform=data_transforms['test'])
        except ValueError:
            print("please check whether your data path is correct or not.")
    if dataset =='vww':
        num_classes = 2
    else: 
        num_classes = len(train_dataset.classes)
    ## Ready to list the dataset information in printed table. 
    total_num = len(train_dataset)
    test_num = len(test_dataset)
    data_table = ['Type', 'Size']
    table_data = [('train', str(total_num)),('test', str(test_num))]
    dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, 
                                                        batch_size=batch_size, shuffle=True, 
                                                        num_workers=workers, pin_memory=True)
    dataloaders['test'] = torch.utils.data.DataLoader(test_dataset, 
                                                        batch_size=batch_size, shuffle=True, 
                                                        num_workers=workers, pin_memory=True)                
    
    #data_table = ['Type', 'Size']
    dataset_sizes['train'] = total_num
    dataset_sizes['test'] = test_num   
    #table_data = [('train', str(train_num)),('test', str(test_num))]    
    # Split dataset using random_split in Pytorch instead of Scikit-Learng split_train_validation         
    if split_ratio != 0 :
        val_num = int(total_num *0.1)
        train_num = int(total_num - val_num)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_num, val_num])
        dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=workers, pin_memory=True)
        dataloaders['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=workers, pin_memory=True)
        dataset_sizes['train'] = train_num
        dataset_sizes['val'] = val_num
        table_data = [('train', str(train_num)),('val', str(val_num))] + [table_data[-1]]   

    print("\n Datasize table \n {}".format(tabulate(table_data, headers=data_table, tablefmt='grid')))
    dataiter = iter(dataloaders['train'])
    images, label = dataiter.next()
    return dataloaders, dataset_sizes, num_classes, images.shape

model_function_dict = mz.data_function_dict

def create_model(dataset, arch, pretrained, device=None):
    arch = arch.lower()
    dataset = dataset.lower()
    if dataset in model_function_dict.keys(): 
        model_function = model_function_dict[dataset]
        model_name = ''
        for i in model_function.keys():
            if arch.startswith(i):
                model_name = i
        #print(model_name)
        if model_name == '':
            raise ValueError("Not support this model so far.")
        #model_name = arch.split('_')[0]

        try:
            if model_name == 'mobilenet':
                model = model_function_dict[dataset][model_name](arch, pretrained, 1.0, device=device)
            else:
                model = model_function_dict[dataset][model_name](arch, pretrained, device=device)       
            #model_function_dict[dataset][model_name](arch, args.pre_trained)
        except KeyError:
            raise ValueError("Not support this architecture so far.")
    else : 
        raise ValueError("Not support this dataset so far.")
    return model


def _log_training_progress(num_classes, classerr, losses, epoch, steps_completed, steps_per_epoch, 
                           batch_time, optimizer, loggers):
    # Log some statistics
    errs = OrderedDict()
    
    #if not early_exit_mode(args):
    if num_classes >= 10:
        errs['Top1'] = classerr.value(1)
        errs['Top5'] = classerr.value(5)
    else: 
        errs['Top1'] = classerr.value(1)
    """
    else:
        # For Early Exit case, the Top1 and Top5 stats are computed for each exit.
        for exitnum in range(args.num_exits):
            errs['Top1_exit' + str(exitnum)] = args.exiterrors[exitnum].value(1)
            errs['Top5_exit' + str(exitnum)] = args.exiterrors[exitnum].value(5)
    """
    stats_dict = OrderedDict()
    for loss_name, meter in losses.items():
        stats_dict[loss_name] = meter.mean
    stats_dict.update(errs)
    stats_dict['LR'] = optimizer.param_groups[0]['lr']
    stats_dict['Time'] = batch_time.mean
    stats = ('Performance/Training/', stats_dict)

    params = model.named_parameters() if args.log_params_histograms else None
    utl.log_training_progress(stats, params, epoch, steps_completed,
                              steps_per_epoch, args.print_freq, loggers)

def light_train_with_distiller(model, criterion, optimizer, compress_scheduler, device, num_classes, 
                                dataset_sizes, loggers, epoch=1):

    """Training-with-compression loop for one epoch. 
    INPORTANT INFORMATION:
    For each training step in epoch:
        compression_scheduler.on_minibatch_begin(epoch)
        output = model(input)
        loss = criterion(output, target)
        compression_scheduler.before_backward_pass(epoch)
        loss.backward()
        compression_scheduler.before_parameter_optimization(epoch)
        optimizer.step()
        compression_scheduler.on_minibatch_end(epoch)
    """    
    total_samples = dataset_sizes['train']
    batch_size = dataloaders["train"].batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)    
    if num_classes >=5:
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=[1, 5]) 
    else:
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=[1]) # Remove top 5.

    batch_time = tnt.AverageValueMeter()
    data_time = tnt.AverageValueMeter()

    OVERALL_LOSS_KEY = 'Overall Loss'
    OBJECTIVE_LOSS_KEY = 'Objective Loss'

    losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                          (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])

    model.train()
    acc_stats = []
    end = time.time()
    for train_step, data in enumerate(dataloaders["train"], 0):
        inputs = data[0].to(device)
        labels = data[1].to(device)

        if compress_scheduler:
            compress_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)
        output = model(inputs)
        loss = criterion(output, labels)

        # Drop the early exist mode in this first version
        classerr.add(output.detach(), labels)
        acc_stats.append([classerr.value(1), classerr.value(5)])
        losses[OBJECTIVE_LOSS_KEY].add(loss.item())
        """
        if not early_exit_mode(args):
        loss = criterion(output, target)
        # Measure accuracy
        classerr.add(output.detach(), target)
        acc_stats.append([classerr.value(1), classerr.value(5)])
        else:
        # Measure accuracy and record loss
        classerr.add(output[args.num_exits-1].detach(), target) # add the last exit (original exit)
        loss = earlyexit_loss(output, target, criterion, args)
        """

        if compress_scheduler: 
            agg_loss =  compress_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, loss,
                                                                      optimizer=optimizer, return_loss_components=True)
            # should by modified, this may incorporated in the future.
            loss = agg_loss.overall_loss 
            """
            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())
            """
            loss = agg_loss.overall_loss
            losses[OVERALL_LOSS_KEY].add(loss.item())

            for lc in agg_loss.loss_components:
                if lc.name not in losses:
                    losses[lc.name] = tnt.AverageValueMeter()
                losses[lc.name].add(lc.value.item())
        else: 
            losses[OVERALL_LOSS_KEY].add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if compress_scheduler:
            compress_scheduler.before_parameter_optimization(epoch, train_step, steps_per_epoch, optimizer)
        optimizer.step()
        if compress_scheduler:
            compress_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)
        batch_time.add(time.time() - end)
        steps_completed = (train_step + 1)

        if steps_completed % args.print_freq == 0 :
        # Change to log writter version, this can be more simplified.
            _log_training_progress(num_classes, classerr, losses, epoch, steps_completed, steps_per_epoch, 
                                   batch_time, optimizer, loggers)
            """
            if num_classes >= 5:
                print('Epoch: [{}][{:5d}/{:5d}]  \tOverall Loss {:.5f}  \tObjective Loss {:.5f}'
                    '\tTop 1 {:.5f}  Top 5 {:.5f}' 
                    '\tLR {:.5f}  Time {:.5f}.'.format(epoch, 
                                                steps_completed, 
                                                int(steps_per_epoch), 
                                                losses['Overall Loss'].mean,  
                                                losses['Objective Loss'].mean, 
                                                classerr.value(1), classerr.value(5),
                                                optimizer.param_groups[0]['lr'], 
                                                batch_time.mean))
                t, total = summary.weights_sparsity_tbl_summary(model, return_total_sparsity=True)
                #print('Total sparsity: {:0.2f}\n'.format(total))
            else:
                print('Epoch: [{}][{:5d}/{:5d}]  \tOverall Loss {:.5f}  \tObjective Loss {:.5f}'
                    '\tTop 1 {:.5f} \tLR {:.5f}  \tTime {:.5f}.'.format(epoch, 
                                                steps_completed, 
                                                int(steps_per_epoch), 
                                                losses['Overall Loss'].mean,  
                                                losses['Objective Loss'].mean, 
                                                classerr.value(1),
                                                optimizer.param_groups[0]['lr'], 
                                                batch_time.mean))
                t, total = summary.weights_sparsity_tbl_summary(module, return_total_sparsity=True)
                #print('Total sparsity: {:0.2f}\n'.format(total))
            """
        end = time.time()
    utl.log_weights_sparsity(model, epoch, loggers)
    if num_classes >= 5:
        return classerr.value(1), classerr.value(5), losses[OVERALL_LOSS_KEY] # classerr.vlaue(5)
    else:
        return classerr.value(1), losses[OVERALL_LOSS_KEY] # classerr.vlaue(5)

def _log_valiation_progress(num_classes, classerr, losses, epoch, steps_completed, steps_per_epoch, loggers):
    #if not _is_earlyexit(args):
    if num_classes >= 10 :
        stats_dict = OrderedDict([('Loss', losses['objective_loss'].mean),
                                    ('Top1', classerr.value(1)),
                                    ('Top5', classerr.value(5))])
    else:
        stats_dict = OrderedDict([('Loss', losses['objective_loss'].mean),
                                    ('Top1', classerr.value(1))])   
    """
    else:
        stats_dict = OrderedDict()
        for exitnum in range(args.num_exits):
            la_string = 'LossAvg' + str(exitnum)
            stats_dict[la_string] = args.losses_exits[exitnum].mean
            # Because of the nature of ClassErrorMeter, if an exit is never taken during the batch,
            # then accessing the value(k) will cause a divide by zero. So we'll build the OrderedDict
            # accordingly and we will not print for an exit error when that exit is never taken.
            if args.exit_taken[exitnum]:
                t1 = 'Top1_exit' + str(exitnum)
                t5 = 'Top5_exit' + str(exitnum)
                stats_dict[t1] = args.exiterrors[exitnum].value(1)
                stats_dict[t5] = args.exiterrors[exitnum].value(5)
    """
    stats = ('Performance/Validation/', stats_dict)
    utl.log_training_progress(stats, None, epoch, steps_completed,
                                   steps_per_epoch, args.print_freq, loggers)

def _validate(data_group, model, criterion, device, num_classes, loggers, epoch=-1):
    
    if epoch != -1:
        msglogger.info("----Validate (epoch=%d)----", epoch)
    # Open source accelerate package! 
    if num_classes >= 5:
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=[1, 5]) 
    else:
        classerr = tnt.ClassErrorMeter(accuracy=True, topk=[1]) # Remove top 5.
    losses = {'objective_loss': tnt.AverageValueMeter()}
    """
    if _is_earlyexit(args):
        # for Early Exit, we have a list of errors and losses for each of the exits.
        args.exiterrors = []
        args.losses_exits = []
        for exitnum in range(args.num_exits):
            args.exiterrors.append(tnt.ClassErrorMeter(accuracy=True, topk=(1, 5)))
            args.losses_exits.append(tnt.AverageValueMeter())
        args.exit_taken = [0] * args.num_exits
    """
    batch_time = tnt.AverageValueMeter()
    total_samples = len(dataloaders[data_group].sampler)
    batch_size = dataloaders[data_group].batch_size
    total_steps = total_samples / batch_size
    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)
    # Display confusion option should be implmented in the near future.
    """
    if args.display_confusion:
        confusion = tnt.ConfusionMeter(args.num_classes
    """
    # Turn into evaluation model.
    model.eval()
    end = time.time()
    # Starting primiary teating code here.
    with torch.no_grad():
        for validation_step, data in enumerate(dataloaders[data_group]):
            inputs = data[0].to(device)
            labels = data[1].to(device)
            output = model(inputs)
            # Neglect elary exist mode in the first version.
            '''
            if not _is_earlyexit(args):
                # compute loss
                loss = criterion(output, target)
                # measure accuracy and record loss
                losses['objective_loss'].add(loss.item())
                classerr.add(output.detach(), target)
                if args.display_confusion:
                    confusion.add(output.detach(), target)
            else:
                earlyexit_validate_loss(output, target, criterion, args)
            '''
            loss = criterion(output, labels)
            losses['objective_loss'].add(loss.item())
            classerr.add(output.detach(), labels)
            steps_completed = (validation_step+1)
            
            batch_time.add(time.time() - end)
            end = time.time()  
            steps_completed = (validation_step+1)
            #Record log using _log_validation_progress function 
            # "\033[0;37;40m\tExample\033[0m"
            if steps_completed % (args.print_freq) == 0 :   
                _log_valiation_progress(num_classes, classerr, losses, epoch, steps_completed, total_steps, [loggers[1]])
                """                  
                if num_classes >= 5:
                    print('Test [{:5d}/{:5d}] \tLoss {:.5f}'  
                            '\tTop1 {:.5f}  \tTop5 {:.5f}'  
                            '\tTime {:.5f}.'.format(steps_completed, 
                                                int(total_steps), 
                                                losses['objective_loss'].mean,  
                                                classerr.value(1), 
                                                classerr.value(5),
                                                batch_time.mean))
                else:
                    print('Test [{:5d}/{:5d}] Loss {:.5f}'  
                            '\tTop1 {:.5f}'  
                            '\tTime {:.5f}.'.format(steps_completed, 
                                                int(total_steps), 
                                                losses['objective_loss'].mean,  
                                                classerr.value(1), 
                                                batch_time.mean))
                """
    if num_classes >= 5:
        stats = ('Performance/Validation/',
        OrderedDict([('Loss', losses['objective_loss'].mean),
                        ('Top1', classerr.value(1)),
                        ('Top5', classerr.value(5))]))
        distiller.log_training_progress(stats, None, epoch, steps_completed=0,
                                        total_steps=1, log_freq=1, loggers=[loggers[0]])
        print('==> Top1 {:.5f} \t Top5 {:.5f} \tLoss: {:.5f}\n.'.format(
                classerr.value(1), classerr.value(5),  losses['objective_loss'].mean)) 
        return  classerr.value(1),  classerr.value(5), losses['objective_loss'].mean 

    else: 
        stats = ('Performance/Validation/',
        OrderedDict([('Loss', losses['objective_loss'].mean),
                        ('Top1', classerr.value(1))]))
        distiller.log_training_progress(stats, None, epoch, steps_completed=0,
                                        total_steps=1, log_freq=1, loggers=[loggers[0]])
        print('==> Top1 {:.5f} \t Loss: {:.5f}\n.'.format(
                    classerr.value(1),  losses['objective_loss'].mean)) 
        return  classerr.value(1), losses['objective_loss'].mean   

def _log_best_scores(performance_tracker, logger, how_many=-1):
    """Utility to log the best scores.
    This function is currently written for pruning use-cases, but can be generalized.
    """
    assert isinstance(performance_tracker, (pt.SparsityAccuracyTracker))
    if how_many < 1:
        how_many = performance_tracker.max_len
    how_many = min(how_many, performance_tracker.max_len)
    best_scores = performance_tracker.best_scores(how_many)
    for score in best_scores:
        logger.info('==> Best [Top1: %.3f   Top5: %.3f   Sparsity:%.2f   NNZ-Params: %d on epoch: %d]',
                    score.top1, score.top5, score.sparsity, -score.params_nnz_cnt, score.epoch)

def trian_validate_with_scheduling(args, net, criterion, optimizer, compress_scheduler, device, num_classes, 
                                    dataset_sizes, loggers, epoch=1, validate=True, verbose=True):
    # Whtat's collectors_context
    if compress_scheduler:
        compress_scheduler.on_epoch_begin(epoch)
    
    if num_classes >= 10:
        top1, top5, loss = light_train_with_distiller(net, criterion, optimizer, compress_scheduler, 
                        device, num_classes, dataset_sizes, loggers, epoch)
    else:
        top1, loss = light_train_with_distiller(net, criterion, optimizer, compress_scheduler, 
                        device, num_classes, dataset_sizes, loggers, epoch)  

    if validate: 
        """
        top1, loss = _validate(net, criterion, optimizer, lr_scheduler, compress_scheduler,
                                device, epoch) # remove top5 accuracy.
        """
        #_validate()
        if num_classes >= 10:
            if args.split_ratio != 0:
                top1, top5, loss = _validate('val', net, criterion, device, num_classes, loggers, epoch=epoch)  
            else: 
                top1, top5, loss = _validate('test', net, criterion, device, num_classes, loggers, epoch=epoch)  
        else:
            if args.split_ratio != 0:
                top1, loss = _validate('val', net, criterion, device, num_classes, loggers, epoch=epoch)   
            else:
                top1, loss = _validate('test', net, criterion, device, num_classes, loggers, epoch=epoch) 

    
    #t, total = summary.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
    #print("\nParameters:\n" + str(t))
    #print('Total sparsity: {:0.2f}\n'.format(total))

    if compress_scheduler:
        compress_scheduler.on_epoch_end(epoch, optimizer, metrics={'min':loss, 'max':top1})

    # Build performance tracker object whilst saving it.
    tracker = pt.SparsityAccuracyTracker(args.num_best_scores)    
    if num_classes >= 10:
        tracker.step(net, epoch, top1=top1, top5=top5)
    else:
        tracker.step(net, epoch, top1=top1)
    #tracker.step(net, epoch, top1=top1, top5=top5) #, top5=top5)
    _log_best_scores(tracker, msglogger)
    best_score = tracker.best_scores()[0]
    is_best = epoch == best_score.epoch
    checkpoint_extras = {'current_top1': top1,
                            'best_top1': best_score.top1,
                            'best_epoch': best_score.epoch}
                            
    #if not os.path.isdir(args.model_path):
    #    os.mkdir(args.model_path)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    name = args.name
    if args.name == '':
        name = args.arch + "_" + args.dataset + "_saved"
    ckpt.save_checkpoint(args.epoch, args.arch, net, optimizer=optimizer,
                         scheduler=compress_scheduler, extras=checkpoint_extras,
                         #is_best=is_best, name=name, dir=args.model_path)
                         is_best=is_best, name=name, dir=msglogger.logdir)

    if num_classes >=5:
        return top1, top5, loss, tracker
    else:
        return top1, loss, tracker

def test(model, criterion, device, num_classes, loggers, args=None, parameter_name=None):
    """ Model Testing Phase """
    if args.sensitivity: 
        msglogger.info("Testing sensitivity of {}.".format(parameter_name))
        #print("Testing sensitivity of {}.".format(parameter_name))
    elif args.train:
        msglogger.info("-----Validate training effectness-----")
        #print("-----Validate training effectness-----")
    elif args.test:
        msglogger.info("-----Testing-----")
        #print("-----Testing-----")
    elif args.post_qq_calibration:
        msglogger.info("Quantization collects activation/weights/batchnorm statistic property.")
        #print("Quantization collects activation/weights/batchnorm statistic property.")
    elif args.post_qe_test:
        msglogger.info("----Post Quantization Testing----")
    else:
        raise ValueError("Please refer to the testing function for more detailed information.")
    """
    if args is None:
        args = ClassifierCompressor.mock_args()

    if activations_collectors is None:
        activations_collectors = utl.create_activation_stats_collectors(model, None)
    """
    # Can be modified! I think this is too complex for us, we just need to be specific.
    # with collectors_context(activations_collectors["test"]) as collectors:
    if num_classes >= 5:
        top1, top5, lossses = _validate('test', model, criterion, device, num_classes, loggers=loggers)
        return top1, top5, lossses
    else:
        top1, lossses = _validate('test', model, criterion, device, num_classes, loggers=loggers)
        return top1, lossses  

def quantize_and_test_model(test_loader, model, criterion, args, device, 
                            num_classes, loggers=None, scheduler=None, save_flag=True):
    """Collect stats using test_loader (when stats file is absent),
    clone the model and quantize the clone, and finally, test it.
    args.device is allowed to differ from the model's device.
    When args.qe_calibration is set to None, uses 0.05 instead.
    scheduler - pass scheduler to store it in checkpoint
    save_flag - defaults to save both quantization statistics and checkpoint.
    """
    if hasattr(model, 'quantizer_metadata') and \
            model.quantizer_metadata['type'] == distiller.quantization.PostTrainLinearQuantizer:
        raise RuntimeError('Trying to invoke post-training quantization on a model that has already been post-'
                           'train quantized. Model was likely loaded from a checkpoint. Please run again without '
                           'passing the --quantize-eval flag')
    if not (args.qe_dynamic or args.qe_stats_file or args.qe_config_file):
        args_copy = copy.deepcopy(args)
        args_copy.qe_calibration = args.qe_calibration if args.qe_calibration is not None else 0.05

        # set stats into args stats field
        args.qe_stats_file = acts_quant_stats_collection(
            model, criterion, loggers, args_copy, save_to_file=save_flag)

    args_qe = copy.deepcopy(args)
    if device == 'cpu':
        # NOTE: Even though args.device is CPU, we allow here that model is not in CPU.
        qe_model = distiller.make_non_parallel_copy(model).cpu()
    else:
        qe_model = copy.deepcopy(model).to(device)

    quantizer = quantization.PostTrainLinearQuantizer.from_args(qe_model, args_qe)
    dummy_input = utl.get_dummy_input(input_shape=model.input_shape)
    quantizer.prepare_model(dummy_input)

    if args.qe_convert_pytorch:
        qe_model = _convert_ptq_to_pytorch(qe_model, args_qe)

    test_res = test(model=qe_model, criterion=criterion, device=device, num_classes=num_classes, args=args_qe, logger=loggers)

    if save_flag:
        checkpoint_name = 'quantized'
        #apputils.save_checkpoint(0, args_qe.arch, qe_model, scheduler=scheduler,
        #    name='_'.join([args_qe.name, checkpoint_name]) if args_qe.name else checkpoint_name,
        #    dir=msglogger.logdir, extras={'quantized_top1': test_res[0]})
        ckpt.save_checkpoint(0, args_qe.arch, qe_model, scheduler=scheduler,
            name='_'.join([args_qe.name, checkpoint_name]) if args_qe.name else checkpoint_name,
            #dir=args.model_path, extras={'quantized_top1': test_res[0]})
            dir=msglogger.logdir, extras={'quantized_top1': test_res[0]})
    del qe_model
    return test_res

if __name__ == '__main__':
    # Imagenet dataset path: /home/swai01/imagenet_datasets/raw-data
    # Some auguments listed here: 
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--learning-rate-decay', '--lrd', default=0.7, type=float,
                    metavar='W', help='learning rate decay (default: 0.7)')
    parser.add_argument('--workers', default=4, type=int, help='number of dataloader workers')
    parser.add_argument('--dataset', type=str, required=True, help='Specify the dataset for creating model and loaders')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the datafolder and it includes train/test ')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch_size for the dataloaders.')
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.1, help='Split training set into two gropus.')
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--cpu', default=False, action='store_true', help="If GPU is full process.")
    parser.add_argument('--train', default=False, action='store_true', help="Training phase.")
    parser.add_argument('--test', default=False, action='store_true', help="Testing phase.")
    parser.add_argument('--model_path', default='/home/bwtseng/Downloads/model_compression/model_save/', type=str, help='Path to trained model.')
    parser.add_argument('--compress', type=str, help='Path to compress configure file.')
    parser.add_argument('--arch', '-a', type=str, required=True, help='Name of used Architecture.')
    parser.add_argument('--name', default='', type=str, help='Save file name.')
    parser.add_argument('--num_best_scores', default=1, type=int, help="num_best_score")
    parser.add_argument('--epoch', default=100, type=int, help="Epoch")
    parser.add_argument('--parallel', default=False, action='store_true', help="Parallel or not")
    parser.add_argument('--pre_trained', default=False, action='store_true', help="using pretrained model from imagenet.")
    parser.add_argument('--resume_from', default=False, action='store_true', help="using the ckpt from local trained model.")
    parser.add_argument('--post_qe_test', default=False, action='store_true', help='whether testing with quantization model.')
    parser.add_argument('--sensitivity', '--sa', choices=['element', 'filter', 'channel'],
                        type=lambda s: s.lower(), help='test the sensitivity of layers to pruning')
    parser.add_argument('--sensitivity_range', '--sr', type=float, nargs=3, default=[0.0, 0.95, 0.05],
                        help='an optional parameter for sensitivity testing '
                             'providing the range of sparsities to test.\n'
                             'This is equivalent to creating sensitivities = np.arange(start, stop, step)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Emit debug log messages')
    parser.add_argument('--log_params_histograms', action='store_true', default=False,
                        help='log the parameter tensors histograms to file '
                             '(WARNING: this can use significant disk space)')
    parser.add_argument('--print_freq', type=int, default=100, help='Record frequency')
    parser.add_argument('--output_dir', type=str, default='/home/bwtseng/Downloads/model_compression/model_save/', 
                        help='Path to your saved file. ')
    distiller.quantization.add_post_train_quant_args(parser, add_lapq_args=True)
    args = parser.parse_args()
    print("\n Argument property: {}.".format(args))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Leverage {} device to run this task.".format(device))
    log_dir = _init_logger(args, script_dir=os.path.dirname(__file__))

    if not log_dir:
        pylogger = tflogger = NullLogger()
    else: 
        # **********************************************************************
        # Wrap the msglogger into this two modules, one for tensorboard visualizarion
        # the other is just send the msglogger to connect other summary functions.
        # **********************************************************************
        tflogger = TensorBoardLogger(msglogger.logdir)
        pylogger = PythonLogger(msglogger)

    #Build data transformer if dataset is loaded using ImageFolder function.
    data_transforms = None
    if args.dataset.lower() not in ['cifar10', 'mnist', 'vww']:
        # ***************************
        # Take imagenet as an example: if using the datafolder, please input the data transform.
        # ***************************
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        if data_transforms is None:
            raise ValueError("Please input your transform function! ")

    dataloaders, dataset_sizes, num_classes, input_shape = data_processing(args.dataset, args.data_dir, args.batch_size,
                                  split_ratio=args.split_ratio, data_transforms=data_transforms, workers=args.workers)
    
    
    for i in dataloaders.keys():
        print("{} data set dataloader is prepared: {}.".format(i , dataloaders[i]))
 
    #model = create_model(args.dataset, args.arch, device=None) 

    if args.cpu: 
        device = torch.device("cpu")
        model = create_model(args.dataset, args.arch, args.pre_trained, device=device)

    else: 
        model = create_model(args.dataset, args.arch, args.pre_trained, device=device)
        """
        if parallel:
            if arch.startswith('alexnet') or arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
            else:
                model = torch.nn.DataParallel(model, device_ids=device_ids)
        """
        if args.parallel:
            model = torch.nn.DataParallel(model, device_ids=device)
            model.is_parallel = args.parallel

        model = torch.nn.DataParallel(model).cuda()

    # ****************************************
    # May have further use in the near future.
    # ****************************************
    transfer_learning= False
    if transfer_learning:
        checkpoint = torch.load('/home/bwtseng/Downloads/mobilenet_sgd_rmsprop_69.526.pth')
        #checkpoint = torch.load('/home/bwtseng/Downloads/mobilenet_sgd_68.848.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])


    # Can change the output layer here, and it's just for transfer learning !
    # For instance model.module.fc = nn.Linear(num_ftrs, 2)
    #num_ftrs = model.module.fc.in_features # This line is very unstable, becuase each model is followed different code style..
    model.arch = args.arch
    model.dataset = args.dataset
    input_shape = (1,) + input_shape[1:]
    model.input_shape = input_shape
    # ***************************************************
    # Define loss, optimiation, weight scheduler, and compress schedule here.
    # ***************************************************
    criterion = nn.CrossEntropyLoss().to(device)
    # Setting weight decay scheduler (TBD)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = None 
    compress_scheduler = None
    if args.train:
        if args.resume_from:
            # Load checkpoint for post training form the pre-trained model.
            #model, compress_scheduler, optimizer, start_epoch = ckpt.load_checkpoint(
            #model, os.path.join('/home/bwtseng/Downloads/', args.model_path, name), 
            #model_device=device)      
            model, compress_scheduler, optimizer, start_epoch = ckpt.load_checkpoint(
                model, os.path.join('/home/bwtseng/Downloads/', args.model_path, name), 
                model_device=device)  
            #optimizer = None
            if optimizer is None: 
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
                print("Do build optimizer")

            if compress_scheduler is None:
                if args.compress:
                    compress_scheduler = utl.file_config(model, optimizer, args.compress, None, None)
                    print("Do load compress")

            model.to(device)
            print("\nStart Training")
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            # Setting Learning decay scheduler, that is, decay LR by a factor of 0.1 every 7 epochs
            # For example: exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
            # Note that all of this can be configured using the YAML file.
            if args.compress:
                compress_scheduler = utl.file_config(model, optimizer, args.compress, None, None)
            model.to(device)
            print("\nStart Training")

        # ***************************************************
        # Print the initial sparsity of this model, and please check whether the pruning 
        # weight name is correct or not. 
        # ***************************************************
        #t, total = summary.weights_sparsity_tbl_summary(model, return_total_sparsity=True)
        #print("\nSparsity: {}.".format(total))
        #print("\nParameters Table: {}".format(str(t)))
        for epoch in range(args.epoch):
            print("\n")
            if num_classes >= 5 :
                top1, top5, losses, tracker = trian_validate_with_scheduling(args, 
                                                    model, criterion, optimizer, 
                                                    compress_scheduler, device, num_classes, dataset_sizes, 
                                                    loggers = [tflogger, pylogger],
                                                    epoch=epoch)
                                                    
            else:
                top1, losses, tracker = trian_validate_with_scheduling(args, 
                                                    model, criterion, optimizer, 
                                                    compress_scheduler, device, num_classes, dataset_sizes,
                                                    loggers = [tflogger, pylogger],
                                                    epoch=epoch)         
            
    elif args.sensitivity:
        try :
            #mdoel = ckpt.load_lean_checkpoint(model, os.path.join(
            #                            args.model_path, args.name+'_best.pth.tar'),
            #                            model_device=device)
            mdoel = ckpt.load_lean_checkpoint(model, arg.model_path, model_device=device)
            #msglogger.info("Successfully load model from the checkpoint {%s}", os.path.join(
            #                args.model_path, args.name+'_best.pth.tar'))
            msglogger.info("Successfully load model from the checkpoint {%s}", args.model_path)
        except:
            pass

        sensitivities = np.arange(*args.sensitivity_range)
        sensitivity_analysis(model, criterion, device, num_classes, args, sensitivities, logger=[tflogger, pylogger])

    else: 
        #Note: load_lean_checkpoint is implemented for testing phase only.
        try :
            #mdoel = ckpt.load_lean_checkpoint(model, os.path.join(
            #                            args.model_path, args.name+'_best.pth.tar'),
            #                            model_device=device)
            mdoel = ckpt.load_lean_checkpoint(model, args.model_path,
                                        model_device=device)

            #msglogger.info("Successfully load model from the checkpoint {%s}", os.path.join(
            #                            args.model_path, args.name+'_best.pth.tar'))
            msglogger.info("Successfully load model from the checkpoint {%s}", args.model_path)
        except:
            raise ValueError("Please input correct model path !")

        if args.test:
            #t, total = summary.weights_sparsity_tbl_summary(model, return_total_sparsity=True)
            #print('Total sparsity: {:0.2f}\n'.format(total))
            #print("{}\n".format(str(t)))
            #f = open('log.txt', 'wt')
            #f.write("{}\n".format(str(t)))
            #f.write('Total sparsity: {:0.2f}\n'.format(total))
            if num_classes >= 5:
                top1, top5, loss = test(model, criterion, device, num_classes, [tflogger, pylogger] ,args)
                #f.write('Top1 accuracy {:0.2f}, Top 5 accuracy {:0.2f}\n'.format(top1, top5))
            else:
                top1, loss = test(model, criterion, device, num_classes, [tflogger, pylogger], args)
                #f.write('Top1 accuracy {:0.2f}\n'.format(top1))         
            #f.close()

        if args.qe_calibration and not (args.test and args.quantize_eval):
            test_fn = partial(test, criterion=criterion, device=device, num_classes=num_classes)
            cmodel = utl.make_non_parallel_copy(model)
            collector.collect_quant_stats(cmodel, test_fn, classes=None, 
                                        inplace_runtime_check=True, disable_inplace_attrs=True,
                                        save_dir=args.model_path)

        # ***************************************************
        # For Post Quantization mechanism, but so far I'm not sure whether it's correct or not.
        # ***************************************************
        if args.post_qe_test:
            quantize_and_test_model(dataloaders['test'], model, criterion, args, device, num_classes, loggers=pylogger)
