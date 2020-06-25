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
import math 
import utility as utl
import distiller
import torchnet.meter as tnt
import torchvision.datasets as datasets
import performance_tracker as pt
import checkpoint as ckpt
import model_zoo.shufflenetv2 as sfn
import summary 
from collections import OrderedDict
import time 
from torch.utils.data.sampler import SubsetRandomSampler
from tabulate import tabulate 


def imagenet_get_datasets(data_dir, load_train=True, load_test=True):
    """
    Load the ImageNet dataset.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    t
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.ImageFolder(train_dir, train_transform)
    total_num = len(train_dataset)
    train_num = int(total_num * 0.8)
    val_num = int(total_num - train_num)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_num, val_num])
    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset

dataloaders = {}
dataset_sizes = {}
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
#/home/swai01/imagenet_datasets/raw-data
train_dataset = datasets.ImageFolder('/home/swai01/imagenet_datasets/raw-data/train/',
                                        transform=data_transforms['train'])
test_dataset = datasets.ImageFolder('/home/swai01/imagenet_datasets/raw-data/val/', 
                                        transform=data_transforms['test'])

total_num = len(train_dataset)
val_num = int(total_num *0.1)
train_num = int(total_num - val_num)
test_num = len(test_dataset)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_num, val_num])

data_table = ['Type', 'Size']
table_data = [('train', str(train_num)),('val', str(val_num)), ('test', str(test_num))]
print("\n Datasize table \n {}".format(tabulate(table_data, headers=data_table, tablefmt='grid')))
# Do I need to apply sampler function to the DataLoader augumentation?
dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
dataloaders['val'] = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
dataloaders['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
dataset_sizes['train'] = train_num
dataset_sizes['val'] = val_num
dataset_sizes['test'] = test_num

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
            #df = summary.masks_sparsity_tbl_summary(net, compress_scheduler)
            #print(df)
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
    ## Can be modified! I think this is too complex for us, we just need to be specific.
    #with collectors_context(activations_collectors["test"]) as collectors:
    """
    net = ckpt.load_lean_checkpoint(net, 
        "/home/bwtseng/Downloads/vww_mobilenetv1_distiller/model_save/mobielnetv1_saved_best.pth.tar",
         device)
    """
    top1, top5, lossses = _validate('test', model, criterion, device)
    #distiller.log_activation_statistics(-1, "test", loggers, collector=collectors['sparsity'])
    #save_collectors_data(collectors, msglogger.logdir)
    return top1, top5, lossses



if __name__ == '__main__':

    print("Training set dataloader is prepared: {}.".format(dataloaders['train']))
    print("Validation set dataloader is prepared: {}.".format(dataloaders['val']))
    print("Testing set dataloader is prepared: {}.".format(dataloaders['val']))

    # Some auguments listed here: 
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--learning-rate-decay', '--lrd', default=0.7, type=float,
                    metavar='W', help='learning rate decay (default: 0.7)')
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--cpu', default=False, type=bool, help="If GPU is full process.")
    parser.add_argument('--train', default=False, type=bool, help="Training phase.")
    parser.add_argument('--test', default=False, type=bool, help="Testing phase.")
    parser.add_argument('--model_path', default='/home/bwtseng/Downloads/vww_mobilenetv1_distiller/model_save/', type=str, help='Path to trained model.')
    parser.add_argument('--compress', default='yaml_file/shufflenet.schedule_agp.yaml', type=str, help='Path to compress configure file.')
    parser.add_argument('--arch', default='shufflenet_v2_x0_5', type=str, help='Name of used Architecture.')
    parser.add_argument('--name', default='shufflenet_v2_x0_5_saved', type=str, help='Save file name.')
    parser.add_argument('--num_best_scores', default=1, type=int, help="num_best_score")
    parser.add_argument('--epoch', default=1, type=int, help="Epoch")
    parser.add_argument('--parallel', default=False, type=bool, help="Parallel or not")
    parser.add_argument('--pre_trained', default=True, type=bool, help="using pretrained model from imagenet.")
    parser.add_argument('--resume_from', default=False, type=bool, help="using the ckpt from local trained model.")
    parser.add_argument('--post_qe_test', default=False, type=bool, help='whether testing with quantization model.')
    distiller.quantization.add_post_train_quant_args(parser, add_lapq_args=True)
    args = parser.parse_args()
    print("\n Argument property: {}.".format(args))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("You use the {} device to do this task.".format(device))


    if args.cpu: 
        device = torch.device("cpu")
        #net = sfn.shufflenet_v2_x0_5(pretrained=args.pre_trained)
        net = sfn.shufflenet_v2_x1_0(pretrained=args.pre_trained)
        net.arch = args.arch
        net.dataset = "Imagenet"
        checkpoint = torch.load('mobilenet_sgd_rmsprop_69.526.pth', map_location=lambda storage, loc: storage)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        print("Remove module string in loaded model !!!")
        for k, v in checkpoint['state_dict'].items():
            #print(k)
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        #net.fc = nn.Linear(num_ftrs, 2)

    else: 
        #net = sfn.shufflenet_v2_x0_5(pretrained=args.pre_trained)#.to(device)
        net = sfn.shufflenet_v2_x1_0(pretrained=args.pre_trained)

        #device = 'cuda'
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

        net.arch = args.arch
        net.dataset = "Imagenet"
        net = torch.nn.DataParallel(net).cuda()
        #net = torch.nn.DataParallel(net).to(device)

        #   _set_model_input_shape_attr(model, arch, dataset, pretrained, cadene)

        #***************************************************

        
        transfer_learning= False
        if transfer_learning:
            #checkpoint = torch.load("/home/bwtseng/Downloads/best.pth.tar")
            checkpoint = torch.load('/home/bwtseng/Downloads/mobilenet_sgd_rmsprop_69.526.pth')
            #checkpoint = torch.load('/home/bwtseng/Downloads/mobilenet_sgd_68.848.pth.tar')
            net.load_state_dict(checkpoint['state_dict'])
            #start_epoch = checkpoint['epoch']
            t, total = summary.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
            #print('Total sparsity: {:0.2f}\n'.format(total))
            #print(t)
            #assert 1== 2
        else:
            pass
        
        # It seems like transfer learning ? That is, use pre-trained imagenet 1000 classes network while applying to VWW datasets (Directly replace final layers.) 
        num_ftrs = net.module.fc.in_features
        print("Output dimension from old model:{}.".format(num_ftrs))
        #net.module.fc = nn.Linear(num_ftrs, 2)

    #***************************************************
    criterion = nn.CrossEntropyLoss().to(device)
    # Setting weight decay scheduler (?)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = None 
    if args.train:
        if args.resume_from:
            # Load checkpoint for post training form the pre-trained model.
            """
            net, compress_scheduler, optimizer, start_epoch = ckpt.load_checkpoint(
            net, "/home/bwtseng/Downloads/vww_mobilenetv1_distiller/model_save/image_net_mobilenetv1_saved_best.pth.tar", 
            model_device=device)
            """
            net, compress_scheduler, optimizer, start_epoch = ckpt.load_checkpoint(
            net, os.path.join('/home/bwtseng/Downloads/', args.model_path, name), 
            model_device=device)      

            optimizer = None
            print(optimizer)
            if optimizer is None: 
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
                print("Do optimizer")

            if compress_scheduler is None:
                compress_scheduler = utl.file_config(net, optimizer, args.compress, None, None)
                print("Do load compress")

            net.to(device)
            print("\nStart Training")
        else:
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            # Setting Learning decay scheduler, that is, decay LR by a factor of 0.1 every 7 epochs
            #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
            # Set up compreesed config file.
            compress_scheduler = utl.file_config(net, optimizer, args.compress, None, None)
            net.to(device)
            print("\nStart Training")
        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
        t, total = summary.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
        print("Sparsity: {}.".format(total))
        print(str(t))
        for epoch in range(args.epoch):
            print("\n")
            top1, top5, losses, tracker = trian_validate_with_scheduling(args, 
                                                    net, criterion, optimizer, 
                                                    compress_scheduler, device, 
                                                    epoch=epoch)
        
    else: 
        #Note: load_lean_checkpoint is designed for testing only.
        #net = ckpt.load_lean_checkpoint(net, "/home/bwtseng/Downloads/vww_mobilenetv1_distiller/model_save/resnet50_pruned_85_best.pth.tar",
        #                   model_device=device)
        top1, top5, loss = test(net, criterion, device)
        t, total = summary.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
        print('Total sparsity: {:0.2f}\n'.format(total))
        if args.qe_calibration and not (args.test and args.quantize_eval):
            test_fn = partial(test, criterion=criterion, device=args.device)
            cmodel = utl.make_non_parallel_copy(net)
            collector.collect_quant_stats(cmodel, test_fn, classes=None, 
                                        inplace_runtime_check=True, disable_inplace_attrs=True,
                                        save_dir="mobilenet_status")
        if args.post_qe_test:
            quantize_and_test_model(dataloaders['val'], net, criterion, args)


        #top1, loss = test(net, criterion, device)

    #net = train_model(net, criterion, optimizer, exp_lr_scheduler,
    #                       device, args.model_path, num_epochs=80)
    #net.to(device)
    #simple_train_for_distiller(net, criterion, optimizer, exp_lr_scheduler, compress_scheduler, device)

    #torch.save(net.state_dict(), args.model_path+'params_final.pth')

