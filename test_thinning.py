import utility
from summary_graph import SummaryGraph
import model_zoo as mz 
import checkpoint as ckpt 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import thinning
import torchvision
from torchvision import datasets, models, transforms
import logging
import torchnet.meter as tnt
import time 
msglogger = logging.getLogger(__name__)
"""
    parser.add_argument('--epsilon', default=8.0, type=float, help='PGD model parameter')
    parser.add_argument('--num_steps', default=10, type=int, help='PGD model parameter')
    parser.add_argument('--step_size', default=2.0, type=float, help='PGD model parameter')
    parser.add_argument('--random_start', default=True, type=bool, help='PGD model parameter')
"""

class AttackPGD(nn.Module):
    def __init__(self, basic_model, args=None):
        super(AttackPGD, self).__init__()
        self.basic_model = basic_model
        self.rand = True
        self.step_size = 2 / 255
        self.epsilon = 8 / 255
        self.num_steps = 10


    def forward(self, input):    # do forward in the module.py
        #if not args.attack :
        #    return self.basic_model(input), input

        if len(input) == 2:
            x = input[0].detach()
            target = input[1]
            if self.rand:
                x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
            for i in range(self.num_steps):
                x.requires_grad_()
                with torch.enable_grad():
                    logits = self.basic_model(x)
                    loss = F.cross_entropy(logits, target, size_average=False)
                grad = torch.autograd.grad(loss, [x])[0]
                x = x.detach() + self.step_size*torch.sign(grad.detach())
                # Normalized !
                x = torch.min(torch.max(x, input[0] - self.epsilon), input[0] + self.epsilon)
                x = torch.clamp(x, 0, 1)
            return self.basic_model(input[0]), self.basic_model(x) , x        

        else: 
            #x = input.detach()
            return self.basic_model(input)


def _validate(data_group, model, criterion, device, num_classes, loggers, epoch=-1):
    
    if epoch != -1:
        msglogger.info("----Validate (epoch=%d)----", epoch)
    # Open source accelerate package! 
    if num_classes >= 10:
        classerr_adv = tnt.ClassErrorMeter(accuracy=True, topk=[1, 5]) 
        classerr_ori = tnt.ClassErrorMeter(accuracy=True, topk=[1, 5]) 
        #classerr = tnt.ClassErrorMeter(accuracy=True, topk=[1, 5]) 
    else:
        classerr_adv = tnt.ClassErrorMeter(accuracy=True, topk=[1]) 
        classerr_ori = tnt.ClassErrorMeter(accuracy=True, topk=[1]) 
        #classerr = tnt.ClassErrorMeter(accuracy=True, topk=[1]) # Remove top 5.
    #losses = {'objective_loss': tnt.AverageValueMeter()}
    losses = {'adversarial_loss': tnt.AverageValueMeter(), 
              'natural_loss': tnt.AverageValueMeter()}

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
            print(device)
            inputs = data[0].to(device)
            labels = data[1].to(device)
            #output = model(inputs)
            ori_output= model(inputs)
            # Early exist mode will incorporate in the near future.

            ori_loss = criterion(ori_output, labels)
            #adv_loss = criterion(adv_output, labels)
            #losses['objective_loss'].add(loss.item())
            losses['natural_loss'].add(ori_loss.item())
            #losses['adversarial_loss'].add(adv_loss.item())
            classerr_ori.add(ori_output.detach(), labels)
            #classerr_adv.add(adv_output.detach(), labels)
            steps_completed = (validation_step + 1)
            
            batch_time.add(time.time() - end)
            end = time.time()  
            steps_completed = (validation_step + 1)
            #Record log using _log_validation_progress function 
            # "\033[0;37;40m\tExample\033[0m"
            #if steps_completed % (args.print_freq) == 0 :   
                #classerr = classerr_adv if args.adv_train else classerr_ori
            #   _log_valiation_progress(num_classes, classerr_ori, classerr_adv, losses, epoch, steps_completed, total_steps, [loggers[1]])
            #print(classerr_adv.value(1))
        print(classerr_ori.value(1))
        assert 1==2
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        #x = x.view(-1, 1024)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

device = "cpu"
#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
input_shape = (1, 3, 224, 224)
#model_function_dict = mz.data_function_dict
#model = model_function_dict['cifar10']['lenet']('lenet', False, device=device)  
#model = AttackPGD(model)
model = Net()
#model, compress_scheduler, optimizer, start_epoch = ckpt.load_checkpoint(
#                    model,  'model_save/lenet_cifar10_log_2020.05.11-224719/lenet_cifar10_retrain_checkpoint.pth.tar', model_device=device)  

model_path = 'model_save/52.06_APEX_Channel_AGP_mobilenet_v1_imagenet_log_2020.06.13-000744/mobilenet_v1_imagenet_retrain_best.pth.tar'
model, compress_scheduler, optimizer, start_epoch = ckpt.load_checkpoint(
                    model, model_path, model_device=device)  


criterion = nn.CrossEntropyLoss().to(device)
#input = torch.rand(input_shape)
#output = model(input)
#label = torch.ones([1], dtype=torch.int64)
#loss = criterion(output, label)
#loss.backward()
#optimizer.step()
print(model)
print(optimizer)
model.to(device)
#print(compress_scheduler.zeros_mask_dict['basic_model.fc2.weight'].mask)
dummy_input = utility.get_dummy_input('imagenet', 
                                      utility.model_device(model), 
                                      input_shape=input_shape)
sgraph = SummaryGraph(model, dummy_input)
print(sgraph)
#thinning_recipe = thinning.create_thinning_recipe_filters(sgraph, model, compress_scheduler.zeros_mask_dict)
thinning_recipe = thinning.create_thinning_recipe_channels(sgraph, model, compress_scheduler.zeros_mask_dict)
thinning.apply_and_save_recipe(model, compress_scheduler.zeros_mask_dict, thinning_recipe, optimizer)
print(model)

# Test accuracy of the thinning model.

"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
dataloaders = {}
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ]),
'test': transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ]),
}
train_dataset = datasets.CIFAR10('cifar10/', train=True, transform=data_transforms['train'])
test_dataset = datasets.CIFAR10('cifar10/', train=False, transform=data_transforms['test'])
dataloaders['train'] = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=32, shuffle=True, 
                                                    num_workers=8, pin_memory=True)
dataloaders['test'] = torch.utils.data.DataLoader(test_dataset, 
                                                    batch_size=32, shuffle=False, 
                                                    num_workers=8, pin_memory=True)    



_validate('test', model, criterion, device, 10, msglogger)
"""

traindir = "/home/swai01/imagenet_datasets/raw-data/train"
valdir = "/home/swai01/imagenet_datasets/raw-data/val"
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

batch_size = 64
train_sampler = None
train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                num_workers=4, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
dataloaders = {}
dataloaders['train'] = train_loader
dataloaders['test'] = val_loader
_validate('test', model, criterion, device, 1000, msglogger)