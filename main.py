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

# TODO: how to support binary classfication? It should be took into account.

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
    dataloaders = {}
    dataset_sizes = {}
    if dataset == "cifar10":
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
        }
        train_dataset = datasets.CIFAR10(data_dir, transform=data_transforms['train'])
        test_dataset = datasets.CIFAR10(data_dir, transform=data_transforms['test'])

    elif dataset == "MNIST":
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
        train_dataset = datasets.MNIST(data_dir, transform=data_transforms['train'])
        test_dataset = datasets.MNIST(data_dir, transform=data_transforms['test'])

    elif dataset == "VWW":
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
        # Special case: Add into argparse arguments.
        # VWW should direct to the root="/home/bwtseng/Downloads/visualwakewords/coco/all", annFile="/home/bwtseng/Downloads/visualwakewords/vww_datasets/annotations/instances_train.json"
        train_dataset = pyvww.pytorch.VisualWakeWordsClassification(root=data_dir, 
                    annFile="/home/bwtseng/Downloads/visualwakewords/vww_datasets/annotations/instances_train.json",
                    transform=data_transforms['train'])
        val_dataset = pyvww.pytorch.VisualWakeWordsClassification(root=data_dir, 
                    annFile="/home/bwtseng/Downloads/visualwakewords/vww_datasets/annotations/instances_val.json",
                    transform=data_transforms['val'])
    else:
        try:
            #/home/swai01/imagenet_datasets/raw-data
            train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), 
                                                transform=data_transforms['train'])
            test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), 
                                                transform=data_transforms['test'])
        except ValueError:
            print("please check whether your data path is correct or not.")
    ## Ready to list the dataset information in printed table. 
    total_num = len(train_dataset)
    test_num = len(test_dataset)
    data_table = ['Type', 'Size']
    table_data = [('train', str(total_num)),('val', str(test_num))]
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
        dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        dataloaders['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        dataset_sizes['train'] = train_num
        dataset_sizes['val'] = val_num
        table_data = [('train', str(train_num)),('val', str(val_num))] + [table_data[-1]]   

    print("\n Datasize table \n {}".format(tabulate(table_data, headers=data_table, tablefmt='grid')))
    return dataloaders, dataset_sizes

model_function_dict = mz.data_function_dict

def create_model(dataset, arch, device=None):
    if dataset in model_function_dict.keys(): 
        model_function = model_function_dict[dataset]
        model_name = ''
        for i in model_function.keys():
            if arch.startswith(i):
                model_name = i
        if model_name == '':
            raise ValueError("Not support this model so far.")
        #model_name = arch.split('_')[0]

        try:
            if model_name == 'mobilenet':
                model = model_function_dict[dataset][model_name](arch, 
                                        args.pre_trained, 1.0, device=device)
            else:
                model = model_function_dict[dataset][model_name](arch, 
                                                                args.pre_trained)       
            #model_function_dict[dataset][model_name](arch, args.pre_trained)
        except KeyError:
            raise ValueError("Not support this architecture so far.")
    else : 
        raise ValueError("Not support this dataset so far.")
    return model

def light_train_with_distiller(model, criterion, optimizer, compress_scheduler, device, epoch=1):

    total_samples = dataset_sizes['train']
    batch_size = dataloaders["train"].batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)    

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=[1, 5]) # It seems that binary can not use top5 accuracy (topk=[1,5]).
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
        steps_completed = (train_step+1)


        # "\033[0;37;40m\tExample\033[0m"
        if steps_completed % 1000 == 0 : 
            print('Epoch: [{}][{:5d}/{:5d}]  \033[0;37;41mOverall Loss {:.5f}  Objective Loss {:.5f}\033[0m'
                    '\033[0;37;42m\tTop 1 {:.5f}  Top 5 {:.5f}\033[0m' 
                    '\033[0;37;40m\tLR {:.5f}  Time {:.5f}\033[0m.'.format(epoch, 
                                                steps_completed, 
                                                int(steps_per_epoch), 
                                                losses['Overall Loss'].mean,  
                                                losses['Objective Loss'].mean, 
                                                classerr.value(1), classerr.value(5),
                                                optimizer.param_groups[0]['lr'], 
                                                batch_time.mean))
            t, total = summary.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
            print('Total sparsity: {:0.2f}\n'.format(total))
        end = time.time()

    return classerr.value(1), classerr.value(5), losses[OVERALL_LOSS_KEY] # classerr.vlaue(5)

#def _validate(model, criterion, optimizer, lr_scheduler, compress_scheduler, device, epoch=1):
def _validate(data_group, model, criterion, device):
    # Open source accelerate package! 
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=[1, 5]) # Remove top 5.
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
            if steps_completed % 300 == 0 : 
                print('Test [{:5d}/{:5d}] \033[0;37;41mLoss {:.5f}\033[0'  
                        '\033[0;37;42m\tTop1 {:.5f}  Top5 {:.5f}\033[m'  
                        '\tTime {:.5f}.'.format(steps_completed, 
                                            int(total_steps), 
                                            losses['objective_loss'].mean,  
                                            classerr.value(1), 
                                            classerr.value(5),
                                            batch_time.mean))
        
        print('==> \033[0;37;42mTop1 {:.5f}  Top5 {:.5f}\033[m'   
            '\033[0;37;41m\tLoss: {:.5f}\n\033[m.'.format(
                       classerr.value(1), classerr.value(5),  losses['objective_loss'].mean)) 

    return  classerr.value(1),  classerr.value(5), losses['objective_loss'].mean               
    #return  classerr.value(1), classerr.value(5), losses['objective_loss'].mean #top1, top5, losses

def trian_validate_with_scheduling(args, net, criterion, optimizer, compress_scheduler, device, 
                                    epoch=1, validate=True, verbose=True):
    # Whtat's collectors_context
    if compress_scheduler:
        compress_scheduler.on_epoch_begin(epoch)

    top1, top5, loss = light_train_with_distiller(net, criterion, optimizer, compress_scheduler, 
                        device, epoch)  

    if validate: 
        """
        top1, loss = _validate(net, criterion, optimizer, lr_scheduler, compress_scheduler,
                                device, epoch) # remove top5 accuracy.
        """
        top1, top5, loss = _validate('val', net, criterion, device)   
    #print(summary.masks_sparsity_tbl_summary(net, compress_scheduler))
    t, total = summary.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
    print("\nParameters:\n" + str(t))
    print('Total sparsity: {:0.2f}\n'.format(total))

    if compress_scheduler:
        compress_scheduler.on_epoch_end(epoch, optimizer, metrics={'min':loss, 'max':top1})

    # Build performance tracker object whilst saving it.
    tracker = pt.SparsityAccuracyTracker(args.num_best_scores)    
    tracker.step(net, epoch, top1=top1, top5=top5) #, top5=top5)
    best_score = tracker.best_scores()[0]
    is_best = epoch == best_score.epoch
    checkpoint_extras = {'current_top1': top1,
                            'best_top1': best_score.top1,
                            'best_epoch': best_score.epoch}
                            
    # args.arch = Architecture name
    ckpt.save_checkpoint(args.epoch, args.arch, net, optimizer=optimizer,
                                     scheduler=compress_scheduler, extras=checkpoint_extras,
                                     is_best=is_best, name=args.name, dir=args.model_path)
    return top1, top5, loss, tracker




def test(model, criterion, device, loggers=None, activations_collectors=None, args=None):
    """Model Test"""
    print('--- test ---------------------')
    """
    if args is None:
        args = ClassifierCompressor.mock_args()

    if activations_collectors is None:
        activations_collectors = utl.create_activation_stats_collectors(model, None)
    """
    # Can be modified! I think this is too complex for us, we just need to be specific.
    # with collectors_context(activations_collectors["test"]) as collectors:
    top1, top5, lossses = _validate('test', model, criterion, device)
    return top1, top5, lossses



if __name__ == '__main__':

    # Some auguments listed here: 
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--learning-rate-decay', '--lrd', default=0.7, type=float,
                    metavar='W', help='learning rate decay (default: 0.7)')
    parser.add_argument('--dataset', type=str, default='imagenet', help='Specify the dataset for creating model and loaders')
    parser.add_argument('--data_dir', type=str, default='/home/swai01/imagenet_datasets/raw-data', help='Path to the datafolder and it includes train/test ')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch_size for the dataloaders.')
    parser.add_argument('--split_ratio', '-sr', type=float, default=0.1, help='Split training set into two gropus.')
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--cpu', default=False, action='store_true', help="If GPU is full process.")
    parser.add_argument('--train', default=False, action='store_true', help="Training phase.")
    parser.add_argument('--test', default=False, action='store_true', help="Testing phase.")
    parser.add_argument('--model_path', default='/home/bwtseng/Downloads/vww_mobilenetv1_distiller/model_save/', type=str, help='Path to trained model.')
    parser.add_argument('--compress', default='yaml_file/shufflenet.schedule_agp.yaml', type=str, help='Path to compress configure file.')
    parser.add_argument('--arch', '-a',default='shufflenet_v2_x0_5', type=str, help='Name of used Architecture.')
    parser.add_argument('--name', default='shufflenet_v2_x0_5_saved', type=str, help='Save file name.')
    parser.add_argument('--num_best_scores', default=1, type=int, help="num_best_score")
    parser.add_argument('--epoch', default=1, type=int, help="Epoch")
    parser.add_argument('--parallel', default=False, action='store_true', help="Parallel or not")
    parser.add_argument('--pre_trained', default=False, action='store_true', help="using pretrained model from imagenet.")
    parser.add_argument('--resume_from', default=False, action='store_true', help="using the ckpt from local trained model.")
    parser.add_argument('--post_qe_test', default=False, action='store_true', help='whether testing with quantization model.')
    distiller.quantization.add_post_train_quant_args(parser, add_lapq_args=True)
    args = parser.parse_args()
    print("\n Argument property: {}.".format(args))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Leverage {} device to run this task.".format(device))


    #Build data transformer if dataset is loaded using ImageFolder function.
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }
    dataloaders, datasize = data_processing(args.dataset, args.data_dir, args.batch_size,
                                  split_ratio=args.split_ratio, data_transforms=data_transforms)
    
    for i in dataloaders.keys():
        print("{} data set dataloader is prepared: {}.".format(i , dataloaders[i]))
    #print("Training set dataloader is prepared: {}.".format(dataloaders['train']))
    #print("Validation set dataloader is prepared: {}.".format(dataloaders['val']))
    #print("Testing set dataloader is prepared: {}.".format(dataloaders['val']))
 
    #model = create_model(args.dataset, args.arch, device=None) 

    if args.cpu: 
        device = torch.device("cpu")
        model = create_model(args.dataset, args.arch, device=device)
    else: 
        model = create_model(args.dataset, args.arch, device=device)
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
        #checkpoint = torch.load("/home/bwtseng/Downloads/best.pth.tar")
        checkpoint = torch.load('/home/bwtseng/Downloads/mobilenet_sgd_rmsprop_69.526.pth')
        #checkpoint = torch.load('/home/bwtseng/Downloads/mobilenet_sgd_68.848.pth.tar')
        net.load_state_dict(checkpoint['state_dict'])
        #start_epoch = checkpoint['epoch']
    else:
        pass
    # Can change the output layer here, just for transfer learning !
    # For instance model.module.fc = nn.Linear(num_ftrs, 2)
    num_ftrs = model.module.fc.in_features
    print("Output dimension from old model:{}.".format(num_ftrs))
    model.arch = args.arch
    model.dataset = args.dataset

    # ***************************************************
    # Define loss, optimiation, weight scheduler, and compress schedule here.
    # ***************************************************
    criterion = nn.CrossEntropyLoss().to(device)
    # Setting weight decay scheduler (TBD)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = None 
    if args.train:
        if args.resume_from:
            # Load checkpoint for post training form the pre-trained model.
            net, compress_scheduler, optimizer, start_epoch = ckpt.load_checkpoint(
            net, os.path.join('/home/bwtseng/Downloads/', args.model_path, name), 
            model_device=device)      
            #optimizer = None
            print(optimizer)
            if optimizer is None: 
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
                print("Do build optimizer")

            if compress_scheduler is None:
                compress_scheduler = utl.file_config(net, optimizer, args.compress, None, None)
                print("Do load compress")

            model.to(device)
            print("\nStart Training")
        else:
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            # Setting Learning decay scheduler, that is, decay LR by a factor of 0.1 every 7 epochs
            # For example: exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
            # Note that all of this can be configured using the YAML file.
            compress_scheduler = utl.file_config(net, optimizer, args.compress, None, None)
            model.to(device)
            print("\nStart Training")

        # ***************************************************
        # Print the initial sparsity of this model, and please check whether the pruning 
        # weight name is correct or not. 
        # ***************************************************
        t, total = summary.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
        print("\nSparsity: {}.".format(total))
        print("\nParameters Table: {}".format(str(t)))
        for epoch in range(args.epoch):
            print("\n")
            top1, top5, losses, tracker = trian_validate_with_scheduling(args, 
                                                    net, criterion, optimizer, 
                                                    compress_scheduler, device, 
                                                    epoch=epoch)
        
    else: 
        #Note: load_lean_checkpoint is designed for testing only.
        net = ckpt.load_lean_checkpoint(net, os.path.join('/home/bwtseng/Downloads/', 
                                        args.model_path, name),
                                        model_device=device)
        top1, top5, loss = test(net, criterion, device)
        t, total = summary.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
        print('Total sparsity: {:0.2f}\n'.format(total))
        if args.qe_calibration and not (args.test and args.quantize_eval):
            test_fn = partial(test, criterion=criterion, device=args.device)
            cmodel = utl.make_non_parallel_copy(net)
            collector.collect_quant_stats(cmodel, test_fn, classes=None, 
                                        inplace_runtime_check=True, disable_inplace_attrs=True,
                                        save_dir="mobilenet_status")
        # ***************************************************
        # For Post Quantization mechanism, but so far I'm not sure whether it's correct or not.
        # ***************************************************
        if args.post_qe_test:
            quantize_and_test_model(dataloaders['val'], net, criterion, args)
