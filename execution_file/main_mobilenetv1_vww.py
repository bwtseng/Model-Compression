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
import performance_tracker as pt
import checkpoint as ckpt
#import classifier as cls # *
import summary 
from collections import OrderedDict
import time 
import distiller.quantization as quantization
import collector 
from functools import partial 
import model_zoo.classifier as cls 

# from scipy.misc import imread, imresize
# ***************************************
# use random split to split training set!!
# ***************************************
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

"""
def save_checkpoint(state, is_best, model_path, filename='checkpoint.pth.tar'):
    torch.save(state, model_path + filename)
    if is_best:
        shutil.copyfile(filename, maodel_path + 'model_best.pth.tar')
"""

def light_train_with_distiller(model, criterion, optimizer, compress_scheduler, device, epoch=1):
   
   
    total_samples = dataset_sizes["train"]
    batch_size = dataloaders["train"].batch_size
    steps_per_epoch = math.ceil(total_samples / batch_size)    

    classerr = tnt.ClassErrorMeter(accuracy=True, topk=[1]) # It seems that binary can not use top5 accuracy (topk=[1,5]).
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
        acc_stats.append([classerr.value(1)]) # [classerr.value(1), classerr.value(5)]
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


        if steps_completed % 30 == 0 : 

            print('Epoch: [{}][{:5d}/{:5d}]  \033[0;37;41mOverall Loss {:.5f}  Objective Loss {:.5f}\033[0m'
                    '\033[0;37;42m\tTop 1 {:.5f}  \033[0m' 
                    '\033[0;37;40m\tLR {:.5f}  Time {:.5f}\033[0m.'.format(epoch, 
                                                steps_completed, 
                                                int(steps_per_epoch), 
                                                losses['Overall Loss'].mean,  
                                                losses['Objective Loss'].mean, 
                                                classerr.value(1),
                                                optimizer.param_groups[0]['lr'], 
                                                batch_time.mean))
            #t, total = summary.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
            #print('Total sparsity: {:0.2f}\n'.format(total))
        
        end = time.time()

    return classerr.value(1), losses[OVERALL_LOSS_KEY]

#def _validate(model, criterion, optimizer, lr_scheduler, compress_scheduler, device, epoch=1):
def _validate(model, criterion, device):
    # Open source accelerate package! 
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
    total_samples = len(dataloaders["val"].sampler)
    batch_size = dataloaders["val"].batch_size
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
        for validation_step, data in enumerate(dataloaders["val"]):
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


    return  classerr.value(1),  losses['objective_loss'].mean               

def trian_validate_with_scheduling(args, net, criterion, optimizer, compress_scheduler, device, 
                                    tracker, epoch=1, validate=True, verbose=True):
    # Whtat's collectors_context
    if compress_scheduler:
        compress_scheduler.on_epoch_begin(epoch)

    top1, loss = light_train_with_distiller(net, criterion, optimizer, compress_scheduler, 
                        device, epoch) 
    if validate: 
        """
        top1, loss = _validate(net, criterion, optimizer, lr_scheduler, compress_scheduler,
                                device, epoch) # remove top5 accuracy.
        """
        top1, loss = _validate(net, criterion, device) # remove top5 accuracy.    
    #print(summary.masks_sparsity_tbl_summary(net, compress_scheduler))
    t, total = summary.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
    print("\nParameters:\n" + str(t))
    print('Total sparsity: {:0.2f}\n'.format(total))
    if compress_scheduler:
        compress_scheduler.on_epoch_end(epoch, optimizer, metrics={'min':loss, 'max':top1})

    # Build performance tracker object whilst saving it.
    # tracker = pt.SparsityAccuracyTracker(args.num_best_scores)    
    tracker.step(net, epoch, top1=top1) 
    best_score = tracker.best_scores()[0]
    is_best = epoch == best_score.epoch
    checkpoint_extras = {'current_top1': top1,
                            'best_top1': best_score.top1,
                            'best_epoch': best_score.epoch}
                            
    # args.arch = Architecture name
    ckpt.save_checkpoint(args.epoch, args.arch, net, optimizer=optimizer,
                                     scheduler=compress_scheduler, extras=checkpoint_extras,
                                     is_best=is_best, name=args.name, dir=args.model_path)
    return top1, loss, tracker


def quantize_and_test_model(test_loader, model, criterion, args, scheduler=None, save_flag=True):
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
    if args.device == 'cpu':
        # NOTE: Even though args.device is CPU, we allow here that model is not in CPU.
        qe_model = distiller.make_non_parallel_copy(model).cpu()
    else:
        qe_model = copy.deepcopy(model).to(args.device)

    quantizer = quantization.PostTrainLinearQuantizer.from_args(qe_model, args_qe)
    dummy_input = utl.get_dummy_input(input_shape=(1, 3, 224, 224)) # should modifiled! or add to args
    quantizer.prepare_model(dummy_input)

    if args.qe_convert_pytorch:
        qe_model = _convert_ptq_to_pytorch(qe_model, args_qe)
    # should check device 
    test_res = test(qe_model, criterion, args.device)

    if save_flag:
        checkpoint_name = 'quantized'
        ckpt.save_checkpoint(0, args_qe.arch, qe_model, scheduler=scheduler,
            name='_'.join([args_qe.name, checkpoint_name]) if args_qe.name else checkpoint_name,
            dir=args.model_path, extras={'quantized_top1': test_res[0]})

    del qe_model
    return test_res

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
    top1, lossses = _validate(model, criterion, device)
    #distiller.log_activation_statistics(-1, "test", loggers, collector=collectors['sparsity'])
    #save_collectors_data(collectors, msglogger.logdir)
    return top1, lossses



if __name__ == '__main__':

    print("Training set dataloader is prepared: {}.".format(dataloaders['train']))
    print("Validation set dataloader is prepared: {}.".format(dataloaders['val']))
    
    # Some auguments listed here: 
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--device', default='gpu', type=str, help='computing device.')
    parser.add_argument('--learning-rate-decay', '--lrd', default=0.7, type=float,
                    metavar='W', help='learning rate decay (default: 0.7)')
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--cpu', default=False, type=bool, help="If GPU is full process.")
    parser.add_argument('--train', default=False, type=bool, help="Training phase.")
    parser.add_argument('--test', default=False, type=bool, help="Testing phase.")
    parser.add_argument('--model_path', default='/home/bwtseng/Downloads/vww_mobilenetv1_distiller/model_save/', type=str, help='Path to trained model.')
    #parser.add_argument('--compress', default='mobilenet.imagenet.schedule_agp_filters.YAML', type=str, help='Path to compress configure file.')
    parser.add_argument('--compress', default='yaml_file/mobilenet.imagenet.schedule_agp.YAML', type=str, help='Path to compress configure file.')    
    
    parser.add_argument('--arch', default='mobilenetv1', type=str, help='Name of used Architecture.')
    parser.add_argument('--name', default='mobilenetv1_saved_3', type=str, help='Save file name.')
    parser.add_argument('--num_best_scores', default=1, type=int, help="num_best_score")
    parser.add_argument('--epoch', default=25, type=int, help="Epoch")
    parser.add_argument('--parallel', default=False, type=bool, help="Parallel or not")
    parser.add_argument('--pre_trained', default=False, type=bool, help="using pretrained model from imagenet.")
    parser.add_argument('--resume_from', default=False, type=bool, help="using the ckpt from local trained model.")
    
    
    
    distiller.quantization.add_post_train_quant_args(parser, add_lapq_args=True)
    args = parser.parse_args()

    # *******************************************
    # This is the code, which is tested with distiller quantization!.
    # *******************************************
    #net = cls.Net()
    #num_ftrs = net.fc.in_features
    #net.fc = nn.Linear(num_ftrs, 2)
    #net = ckpt.load_lean_checkpoint(net, "/home/bwtseng/Downloads/vww_mobilenetv1_distiller/model_save/mobilenetv1_saved_2_best.pth.tar",
    #                            model_device=args.device)
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    #compress_schedutle = utl.file_config(net, optimizer, args.compress, None, None)
    """
    if args.qe_calibration and not (args.test and args.quantize_eval):
        test_fn = partial(test, criterion=criterion, device=args.device)
        cmodel = utl.make_non_parallel_copy(net)
        collector.collect_quant_stats(cmodel, test_fn, classes=None, 
                                      inplace_runtime_check=True, disable_inplace_attrs=True,
                                      save_dir="mobilenet_status")

    if args.test:
        quantize_and_test_model(dataloaders['val'], net, criterion, args)
    """

    print("\n Argument property: {}.".format(args))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("You use the {} device to do this task.".format(device))


    if args.cpu: 
        device = torch.device("cpu")
        net = cls.Net()
        net.arch = args.arch
        net.dataset = "VWW_dataset"
        checkpoint = torch.load('mobilenet_sgd_rmsprop_69.526.pth', map_location=lambda storage, loc: storage)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        print("Remove module string in loaded model !!!") # Cause CPU can't support module.
        for k, v in checkpoint['state_dict'].items():
            #print(k)
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

        num_ftrs = net.fc.in_features
        print("The dimension before forwarding to output layer is {}.".format(num_ftrs))
        net.fc = nn.Linear(num_ftrs, 2)

    else: 
        net = cls.Net()
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
        net.dataset = "VWW_dataset"
        net = torch.nn.DataParallel(net).cuda()
        #net = torch.nn.DataParallel(net).to(device)

        #   _set_model_input_shape_attr(model, arch, dataset, pretrained, cadene)

        # ***************************************************
        # May have other useage in the near future.
        # ***************************************************
        transfer_learning= False
        if transfer_learning:
            checkpoint = torch.load('/home/bwtseng/Downloads/mobilenet_sgd_rmsprop_69.526.pth')
            net.load_state_dict(checkpoint['state_dict'])
        else:
            pass

        # It seems like transfer learning ? That is, use pre-trained imagenet 1000 classes network while applying to VWW datasets (Directly replace final layers.) 
        #num_ftrs = net.module.fc.in_features
        #print("Output dimension from old model:{}.".format(num_ftrs))
        #net.module.fc = nn.Linear(num_ftrs, 2)

    #***************************************************
    criterion = nn.CrossEntropyLoss()
    # Setting weight decay scheduler (?)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    if args.train:
        if args.resume_from:
            """
            net, compress_scheduler, optimizer, start_epoch = ckpt.load_checkpoint(
            net, "/home/bwtseng/Downloads/vww_mobilenetv1_distiller/model_save/mobilenetv1_saved_best.pth.tar", 
            model_device=device)
            """

            net, compress_scheduler, optimizer, start_epoch = ckpt.load_checkpoint(
            net, "/home/bwtseng/Downloads/mobilenet_sgd_rmsprop_69.526.pth", 
            model_device=device)      

            optimizer = None
            if optimizer is None: 
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
                print("Do optimizer")

            if compress_scheduler is None:
                compress_scheduler = utl.file_config(net, optimizer, args.compress, None, None)
                print("Do load compress")


        else:
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            # Setting Learning decay scheduler, that is, decay LR by a factor of 0.1 every 7 epochs
            #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
            # Set up compreesed config file.
            compress_scheduler = utl.file_config(net, optimizer, args.compress, None, None)
            #net = net.to(device)

        #Drop Learning decay schedule, causes it is deployed in the yaml file.
        #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

        num_ftrs = net.module.fc.in_features
        print("Output dimension from old model:{}.".format(num_ftrs))
        net.module.fc = nn.Linear(num_ftrs, 2)
        net.to(device)

        #t, total = summary.weights_sparsity_tbl_summary(net, return_total_sparsity=True)
        #print("Sparsity: {}.".format(total))
        tracker = pt.SparsityAccuracyTracker(args.num_best_scores)    
        tracker.reset()
        for epoch in range(args.epoch):
            top1, losses, tracker = trian_validate_with_scheduling(args, net, criterion, optimizer, compress_scheduler, device, tracker, epoch=(epoch+1))

    else: 
        if args.test:
            net = ckpt.load_lean_checkpoint(net, "/home/bwtseng/Downloads/vww_mobilenetv1_distiller/model_save/mobilenetv1_saved_best.pth.tar",
                                model_device=device)
            top1, loss = test(net, criterion, device)


            if args.qe_calibration and not (args.test and args.quantize_eval):
                test_fn = partial(test, criterion=criterion, device=args.device)
                cmodel = utl.make_non_parallel_copy(net)
                collector.collect_quant_stats(cmodel, test_fn, classes=None, 
                                            inplace_runtime_check=True, disable_inplace_attrs=True,
                                            save_dir="mobilenet_status")

            if args.post_qe_test:
                quantize_and_test_model(dataloaders['val'], net, criterion, args)

        #top1, loss = test(net, criterion, device)
