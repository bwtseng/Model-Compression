from pruning import ranked_structures_pruner
from pruning import automated_gradual_pruner
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
import thresholding


'''
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
'''
_prune_ratio = {
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
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

def cache_featuremaps_fwd_hook(module, input, output, intermediate_fms, n_points_per_fm):
    """Create a cached dictionary of each layer's input and output feature-maps.

    For reconstruction of weights, we need to collect pairs of (layer_input, layer_output)
    using a sample subset of the input dataset.
    This is a forward-hook function, invoked from forward_hooks of Convolution layers.
    Use this in conjunction with distiller.features_collector.collect_intermediate_featuremap_samples,
    which orchestrates the process of feature-map collection.

    This foward-hook samples random points in the output feature-maps of 'module'.
    After collecting the feature-map samples, distiller.FMReconstructionChannelPruner can be used.

    Arguments:
        module - the module who's forward_hook is invoked
        input, output - the input and output arguments to the forward_hook
        intermediate_fms - a dictionary of lists of feature-map samples, per layer 
            (use module.distiller_name as key)
        n_points_per_fm - number of points to sample, per feature-map.
    """
    def im2col(x, conv):
        x_unfold = f.unfold(x, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding)
        return x_unfold

    # Sample random (uniform) points in each feature-map.
    # This method is biased toward small feature-maps.
    if isinstance(module, torch.nn.Conv2d):
        randx = np.random.randint(0, output.size(2), n_points_per_fm)
        randy = np.random.randint(0, output.size(3), n_points_per_fm)

    X = input[0]
    if isinstance(module, torch.nn.Linear):
        X = X.detach().cpu().clone()
        Y = output.detach().cpu().clone()
    elif module.kernel_size == (1, 1):
        X = X[:, :, randx, randy].detach().cpu().clone()
        Y = output[:, :, randx, randy].detach().cpu().clone()
    else:
        w, h = X.size(2), X.size(3)
        X = im2col(X.detach().cpu().clone(), module).squeeze()
        w_out = output.size(2)
        pts = randx * w_out + randy
        X = X[:, :, pts].detach().cpu().clone()
        Y = output[:, :, randx, randy].detach().cpu().clone()

    # Preprocess the outputs: transpose the batch and channel dimensions, create a flattened view, and transpose.
    # The outputs originally have shape: (batch size, num channels, feature-map width, feature-map height).
    Y = Y.view(Y.size(0), Y.size(1), -1)
    Y = Y.transpose(2, 1)
    Y = Y.contiguous().view(-1, Y.size(2))

    intermediate_fms['output_fms'][module.distiller_name].append(Y)
    intermediate_fms['input_fms'][module.distiller_name].append(X)


def basic_featuremaps_caching_fwd_hook(module, input, output, intermediate_fms):
    """A trivial function to cache input/output feature-maps
    
    The input feature-maps are appended to a list of input-maps that are input to
    this module.  This list is provided by an external context.  A similar setup
    exists for output feature-maps.
    This function is invoked from the forward-hook of modules and can be called from
    various threads and the modules can exist on multiple GPUs.  Therefore, we use Python
    lists (on the CPU) to protect against race-conditions and synchronize the data.
    Using the CPU to store the lists also benefits from the larger CPU DRAM.
    """
    intermediate_fms['output_fms'][module.distiller_name].append(output)
    intermediate_fms['input_fms'][module.distiller_name].append(input[0])

    
def collect_intermediate_featuremap_samples(model, forward_fn, module_filter_fn, 
                                            fm_caching_fwd_hook=basic_featuremaps_caching_fwd_hook):
    '''
    Collect pairs of input/output feature-maps.
    '''
    from functools import partial

    def install_io_collectors(m, intermediate_fms):
        if module_filter_fn(m):
            intermediate_fms['output_fms'][m.distiller_name] = []
            intermediate_fms['input_fms'][m.distiller_name] = []
            hook_handles.append(m.register_forward_hook(partial(fm_caching_fwd_hook, 
                                                                intermediate_fms=intermediate_fms)))

    # Register to the forward hooks, then run the forward-pass and collect the data
    msglogger.warning("==> Collecting input/ouptput feature-map pairs")
    distiller.assign_layer_fq_names(model)
    hook_handles = []
    intermediate_fms = {"output_fms": dict(), "input_fms": dict()}
    model.apply(partial(install_io_collectors, intermediate_fms=intermediate_fms))
    
    forward_fn()
    
    # Unregister from the forward hooks
    for handle in hook_handles:
        handle.remove()

    # We now need to concatenate the list of feature-maps to torch tensors.
    msglogger.info("Concatenating FMs...")
    model.intermediate_fms = {"output_fms": dict(), "input_fms": dict()}
    outputs = model.intermediate_fms['output_fms']
    inputs = model.intermediate_fms['input_fms']

    for (layer_name, X), Y in zip(intermediate_fms['input_fms'].items(), intermediate_fms['output_fms'].values()):                
        inputs[layer_name] = torch.cat(X, dim=0)
        outputs[layer_name] = torch.cat(Y, dim=0)

    msglogger.warning("<== Done.")
    del intermediate_fms 


dataset = 'imagenet'
arch = 'mobilenet_v1'
pretrained = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device ='cpu'
model_function_dict = mz.data_function_dict
model_function = model_function_dict[dataset]
model = model_function_dict[dataset]['mobilenet'](arch, pretrained)
model = Net()
model.to(device)
model = torch.nn.DataParallel(model).cuda()
prepare_pruning(model)

#pruner = automated_gradual_pruner.L1RankedStructureParameterPruner_AGP("L1RankedStructureParameterPruner_AGP", 0.15, 0.75, 'Rows', _prune_ratio.keys() )
#pruner = ranked_structures_pruner.LpRankedStructureParameterPruner("test", "Rows", 0.75, )
#pruner = admm_pruner.ADMMPruner("admm_pruner", "ADMM_puner", pruning_ratio=prune_ratios, rho=0.001, 
#                    sparsity_type="channel", masked_progressive=False, admm_epoch=10, 
#                    initial_lr=0.01, multi_rho=True, model=model)
zeros_mask_dict = create_model_masks_dict(model)

# Use this scrip check whether the channel purning reuslts are consistent with our ADMM pruning maks implemented by me.
# And the answer is true, with tuning different ratios, the results are all same.
for name, W in model.named_parameters():
    if name == "module.model.10.3.weight":
        binary_map = ranked_structures_pruner.LpRankedStructureParameterPruner.rank_and_prune_channels(0.75, W, 'module.fc.weight', zeros_mask_dict, model)
        #binary_map = ranked_structures_pruner.LpRankedStructureParameterPruner.rank_and_prune_filters(0.75, W, 'module.model.10.3.weight', zeros_mask_dict, model)
        #ranked_structures_pruner.FMReconstructionChannelPruner.cache_featuremaps_fwd_hook()
        #binary_map = ranked_structures_pruner.FMReconstructionChannelPruner.rank_and_prune_channels(0.75, W, name, zeros_mask_dict, model)
        #mask, _ = thresholding.expand_binary_map(W, 'Filters', binary_map)
        mask, _ = thresholding.expand_binary_map(W, 'Channels', binary_map)
        mask_numpy = mask.cpu().detach().numpy()

        weight = W.cpu().detach().numpy()
        shape = weight.shape
        percent = 0.75 * 100    
        #assert 1 == 2 
        
        if len(shape) == 2:
            weight_t = weight.transpose(1, 0)
        else: 
            weight_t = weight.transpose(1, 0 ,2, 3)
        weight2d = weight_t.reshape(shape[1], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 1, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm < percentile
        above_threshold = row_l2_norm > percentile
        print(above_threshold.shape)
        # Masked weight is the second output of this function.
        weight2d[under_threshold, :] = 0  
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i,:] = above_threshold[i]
        tensor_2d = torch.from_numpy(above_threshold)
        mask_admm, _ = thresholding.expand_binary_map(W, 'Channels', tensor_2d)
        mask_admm = mask_admm.cpu().detach().numpy()
        print(mask_numpy.shape)
        print(mask_admm.shape)
        print(np.sum(mask_admm == mask_numpy))
        if len(shape) == 2: 
            weight2d = weight2d.reshape(shape[1], shape[0])
            weight = weight2d.transpose(1, 0)
            expand_above_threshold = expand_above_threshold.reshape(shape[1], shape[0])
            expand_above_threshold = expand_above_threshold.transpose(1, 0)
        else:
            weight2d = weight2d.reshape(shape[1], shape[0], shape[2], shape[3])
            weight = weight2d.transpose(1, 0, 2, 3)
            expand_above_threshold = expand_above_threshold.reshape(shape[1], shape[0], shape[2], shape[3])
            expand_above_threshold = expand_above_threshold.transpose(1, 0, 2, 3)  
        """
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 1, axis = 1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm < percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0          
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)          
        for i in range(shape2d[0]):
            expand_above_threshold[i,:] = above_threshold[i]
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        """
        #print(expand_above_threshold)
        print(np.sum(expand_above_threshold == mask_numpy))
        print(np.sum(expand_above_threshold == mask_admm))
assert 1 == 2

for name, W in model.named_parameters():
    print(name)
    print(W.shape)

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
        #pruner.admm_update(model, epo, i, zeros_mask_dict)
        #ce_loss, admm_loss = pruner.append_admm_loss(model, ce_loss, pruner.sparsity_type) 
        #print(pruner.ADMM_Z["module.model.1.3.weight"][:,10,:,:])
        
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