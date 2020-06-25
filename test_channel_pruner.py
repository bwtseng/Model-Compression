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


# THis function includes in the pruner class, staticmethod. 
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

    # Should give another options that can use whole features map as the input/output of regression problem, no need it, since Prof. Han also adopt this implementations.
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


def assign_layer_fq_names(container, name=None):
    """Assign human-readable names to the modules (layers).

    Sometimes we need to access modules by their names, and we'd like to use
    fully-qualified names for convenience.
    """
    for name, module in container.named_modules():
        module.distiller_name = name

# Example of module_filter_fn used in the Distiller's AMC
def acceptance_criterion(m, mod_names):
    # Collect feature-maps only for Conv2d layers and fc layers, if they are in our modules list.
    return isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)) and m.distiller_name in mod_names

def collect_intermediate_featuremap_samples(model, forward_fn, module_filter_fn, 
                                            fm_caching_fwd_hook=basic_featuremaps_caching_fwd_hook): #cache_featuremaps_fwd_hook
    '''
    Collect pairs of input/output feature-maps.
    # Some Note here: 
    # 
        foward_fn: feed validation set and get the feature maps from hook, and services argument is just a nametuple which includes three callable function.
        module_filter_fn: check layer's property
        fm_caching_fwd_hood: your register function.
    # 
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

"""
# fIND THE ARGS IN CLASSIFIER amc.

    amc_cfg = distiller.utils.MutableNamedTuple({
            'modules_dict': compression_cfg["network"],  # dict of modules, indexed by arch name
            'save_chkpts': args.amc_save_chkpts,
            'protocol': args.amc_protocol,
            'agent_algo': args.amc_agent_algo,
            'num_ft_epochs': num_ft_epochs,
            'action_range': action_range,
            'reward_frequency': args.amc_reward_frequency,
            'ft_frequency': args.amc_ft_frequency,
            'pruning_pattern':  args.amc_prune_pattern,
            'pruning_method': args.amc_prune_method,
            'group_size': args.amc_group_size,
            'n_points_per_fm': args.amc_fm_reconstruction_n_pts,
            'ddpg_cfg': ddpg_cfg,
            'ranking_noise': args.amc_ranking_noise})
"""

# partial(acceptance_criterion, mod_names=modules_list) modules_dict defined in their yaml file, just copy whilst revsing to our favorite style.
# here comes the module_filter_fn, and it's also a partial function...


# here is the validation function, which comes from distiller.apptutils.classifier. Thus , we can further import our test function to test whether this class can be executable.
# validate_fn = partial(classifier.test, test_loader=val_loader, criterion=self.criterion,
#                              loggers=self.pylogger, args=self.args, activations_collectors=None)
model = Net()
collect_intermediate_featuremap_samples(model, validate_fn, acceptance_criterion, cache_featuremaps_fwd_hook)
