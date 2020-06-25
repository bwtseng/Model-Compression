import yaml
import numpy as np 
import distiller
import inspect 
import json
import logging
from torch.optim.lr_scheduler import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
from torch.optim.lr_scheduler import _LRScheduler

msglogger = logging.getLogger()
### **************************************
### Config other scheme schedule.
### **************************************
def get_dummy_input(dataset=None, device=None, input_shape=None):
    """Generate a representative dummy (random) input.
    If a device is specified, then the dummy_input is moved to that device.
    Args:
        dataset (str): Name of dataset from which to infer the shape
        device (str or torch.device): Device on which to create the input
        input_shape (tuple): Tuple of integers representing the input shape. Can also be a tuple of tuples, allowing
          arbitrarily complex collections of tensors. Used only if 'dataset' is None
    """
    def create_single(shape):
        t = torch.randn(shape)
        if device:
            t = t.to(device)
        return t

    def create_recurse(shape):
        if all(isinstance(x, int) for x in shape):
            return create_single(shape)
        return tuple(create_recurse(s) for s in shape)

    if input_shape is not None: 
        return create_recurse(input_shape)
    else:
        input_shape = _validate_input_shape(dataset, input_shape)
        return create_recurse(input_shape)
### **************************************
### Config other scheme schedule.
### **************************************



### ****************************************
### Used in compression_scheduler/policy/summary_graph file 
### ****************************************
def model_device(model):
    """Determine the device the model is allocated on."""
    # Source: https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
    if isinstance(model, nn.DataParallel):
        return model.src_device_obj
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        # Model has no parameters
        pass
    return 'cpu'

def normalize_module_name(layer_name):
    """Normalize a module's name.
    PyTorch let's you parallelize the computation of a model, by wrapping a model with a
    DataParallel module.  Unfortunately, this changs the fully-qualified name of a module,
    even though the actual functionality of the module doesn't change.
    Many time, when we search for modules by name, we are indifferent to the DataParallel
    module and want to use the same module name whether the module is parallel or not.
    We call this module name normalization, and this is implemented here.
    """
    modules = layer_name.split('.')
    try:
        idx = modules.index('module')
    except ValueError:
        return layer_name
    del modules[idx]
    return '.'.join(modules)

def denormalize_module_name(parallel_model, normalized_name):
    """Convert back from the normalized form of the layer name, to PyTorch's name
    which contains "artifacts" if DataParallel is used.
    """
    fully_qualified_name = [mod_name for mod_name, _ in parallel_model.named_modules() if
                            normalize_module_name(mod_name) == normalized_name]
    if len(fully_qualified_name) > 0:
        return fully_qualified_name[-1]
    else:
        return normalized_name   # Did not find a module with the name <normalized_name>

def model_find_param_name(model, param_to_find):
    """Look up the name of a model parameter.
    Arguments:
        model: the model to search
        param_to_find: the parameter whose name we want to look up
    Returns:
        The parameter name (string) or None, if the parameter was not found.
    """
    for name, param  in model.named_parameters():
        if param is param_to_find:
            return name
    return None


def model_find_module_name(model, module_to_find):
    """Look up the name of a module in a model.
    Arguments:
        model: the model to search
        module_to_find: the module whose name we want to look up
    Returns:
        The module name (string) or None, if the module was not found.
    """
    for name, m in model.named_modules():
        if m == module_to_find:
            return name
    return None


def model_find_param(model, param_to_find_name):
    """Look a model parameter by its name
    Arguments:
        model: the model to search
        param_to_find_name: the name of the parameter that we are searching for
    Returns:
        The parameter or None, if the paramter name was not found.
    """
    for name, param in model.named_parameters():
        if name == param_to_find_name:
            return param
    return None


def model_find_module(model, module_to_find):
    """Given a module name, find the module in the provided model.
    Arguments:
        model: the model to search
        module_to_find: the module whose name we want to look up
    Returns:
        The module or None, if the module was not found.
    """
    for name, m in model.named_modules():
        if name == module_to_find:
            return m
    return None

def non_zero_channels(tensor):
    """Returns the indices of non-zero channels.
    Non-zero channels are channels that have at least one coefficient that
    is not zero.  Counting non-zero channels involves some tensor acrobatics.
    """
    if tensor.dim() != 4:
        raise ValueError("Expecting a 4D tensor")

    norms = distiller.norms.channels_lp_norm(tensor, p=1)
    nonzero_channels = torch.nonzero(norms)
    return nonzero_channels

def density(tensor):
    """Computes the density of a tensor.
    Density is the fraction of non-zero elements in a tensor.
    If a tensor has a density of 1.0, then it has no zero elements.
    Args:
        tensor: the tensor for which we compute the density.
    Returns:
        density (float)
    """
    # Using torch.nonzero(tensor) can lead to memory exhaustion on
    # very large tensors, so we count zeros "manually".
    nonzero = tensor.abs().gt(0).sum()
    return float(nonzero.item()) / torch.numel(tensor)

def make_non_parallel_copy(model):
    """Make a non-data-parallel copy of the provided model.
    torch.nn.DataParallel instances are removed.
    """
    def replace_data_parallel(container):
        for name, module in container.named_children():
            if isinstance(module, nn.DataParallel):
                setattr(container, name, module.module)
            if has_children(module):
                replace_data_parallel(module)

    # Make a copy of the model, because we're going to change it
    new_model = deepcopy(model)
    if isinstance(new_model, nn.DataParallel):
        new_model = new_model.module
    replace_data_parallel(new_model)

    return new_model

def convert_tensors_recursively_to(val, *args, **kwargs):
    """ Applies `.to(*args, **kwargs)` to each tensor inside val tree. Other values remain the same."""
    if isinstance(val, torch.Tensor):
        return val.to(*args, **kwargs)

    if isinstance(val, (tuple, list)):
        return type(val)(convert_tensors_recursively_to(item, *args, **kwargs) for item in val)

    return val

def has_children(module):
    try:
        next(module.children())
        return True
    except StopIteration:
        return False

def param_name_2_module_name(param_name):
    return '.'.join(param_name.split('.')[:-1])

### ****************************************
### Used in compression_scheduler/policy/summary_graph file [end]
### ****************************************

def make_non_parallel_copy(model):
    """Make a non-data-parallel copy of the provided model.
    torch.nn.DataParallel instances are removed.
    """
    def replace_data_parallel(container):
        for name, module in container.named_children():
            if isinstance(module, nn.DataParallel):
                setattr(container, name, module.module)
            if has_children(module):
                replace_data_parallel(module)

    # Make a copy of the model, because we're going to change it
    new_model = deepcopy(model)
    if isinstance(new_model, nn.DataParallel):
        new_model = new_model.module
    replace_data_parallel(new_model)

    return new_model
    
### ****************************************
### Preprare for the quantization file [end]
### ****************************************



### **********************************
### Used for calling logger function. 
### **********************************
def density(tensor):
    """Computes the density of a tensor.
    Density is the fraction of non-zero elements in a tensor.
    If a tensor has a density of 1.0, then it has no zero elements.
    Args:
        tensor: the tensor for which we compute the density.
    Returns:
        density (float)
    """
    # Using torch.nonzero(tensor) can lead to memory exhaustion on
    # very large tensors, so we count zeros "manually".
    nonzero = tensor.abs().gt(0).sum()
    return float(nonzero.item()) / torch.numel(tensor)


def sparsity(tensor):
    """Computes the sparsity of a tensor.
    Sparsity is the fraction of zero elements in a tensor.
    If a tensor has a density of 0.0, then it has all zero elements.
    Sparsity and density are complementary.
    Args:
        tensor: the tensor for which we compute the density.
    Returns:
        sparsity (float)
    """
    return 1.0 - density(tensor)

def sparsity_2D(tensor):
    """Create a list of sparsity levels for each channel in the tensor 't'
    For 4D weight tensors (convolution weights), we flatten each kernel (channel)
    so it becomes a row in a 3D tensor in which each channel is a filter.
    So if the original 4D weights tensor is:
        #OFMs x #IFMs x K x K
    The flattened tensor is:
        #OFMS x #IFMs x K^2
    For 2D weight tensors (fully-connected weights), the tensors is shaped as
        #IFMs x #OFMs
    so we don't need to flatten anything.
    To measure 2D sparsity, we sum the absolute values of the elements in each row,
    and then count the number of rows having sum(abs(row values)) == 0.
    """
    if tensor.dim() == 4:
        # For 4D weights, 2D structures are channels (filter kernels)
        view_2d = tensor.view(-1, tensor.size(2) * tensor.size(3))
    elif tensor.dim() == 2:
        # For 2D weights, 2D structures are either columns or rows.
        # At the moment, we only support row structures
        view_2d = tensor
    else:
        return 0

    num_structs = view_2d.size()[0]
    nonzero_structs = len(torch.nonzero(view_2d.abs().sum(dim=1)))
    return 1 - nonzero_structs/num_structs


def density_2D(tensor):
    """Kernel-wise sparsity for 4D tensors"""
    return 1 - sparsity_2D(tensor)

def to_np(var):
    return var.data.cpu().numpy()


def size2str(torch_size):
    if isinstance(torch_size, torch.Size):
        return size_to_str(torch_size)
    if isinstance(torch_size, (torch.FloatTensor, torch.cuda.FloatTensor)):
        return size_to_str(torch_size.size())
    if isinstance(torch_size, torch.autograd.Variable):
        return size_to_str(torch_size.data.size())
    if isinstance(torch_size, tuple) or isinstance(torch_size, list):
        return size_to_str(torch_size)
    raise TypeError


def size_to_str(torch_size):
    """Convert a pytorch Size object to a string"""
    assert isinstance(torch_size, torch.Size) or isinstance(torch_size, tuple) or isinstance(torch_size, list)
    return '('+(', ').join(['%d' % v for v in torch_size])+')'


def norm_filters(weights, p=1):
    return distiller.norms.filters_lp_norm(weights, p)


def log_training_progress(stats_dict, params_dict, epoch, steps_completed, total_steps, log_freq, loggers):
    """Log information about the training progress, and the distribution of the weight tensors.
    Args:
        stats_dict: A tuple of (group_name, dict(var_to_log)).  Grouping statistics variables is useful for logger
          backends such as TensorBoard.  The dictionary of var_to_log has string key, and float values.
          For example:
              stats = ('Peformance/Validation/',
                       OrderedDict([('Loss', vloss),
                                    ('Top1', top1),
                                    ('Top5', top5)]))
        params_dict: A parameter dictionary, such as the one returned by model.named_parameters()
        epoch: The current epoch
        steps_completed: The current step in the epoch
        total_steps: The total number of training steps taken so far
        log_freq: The number of steps between logging records
        loggers: A list of loggers to send the log info to
    """
    if loggers is None:
        return
    if not isinstance(loggers, list):
        loggers = [loggers]
    for logger in loggers:
        logger.log_training_progress(stats_dict, epoch,
                                     steps_completed,
                                     total_steps, freq=log_freq)
        logger.log_weights_distribution(params_dict, steps_completed)


def log_activation_statistics(epoch, phase, loggers, collector):
    """Log information about the sparsity of the activations"""
    if collector is None:
        return
    if loggers is None:
        return
    for logger in loggers:
        logger.log_activation_statistic(phase, collector.stat_name, collector.value(), epoch)


def log_weights_sparsity(model, epoch, loggers):
    """Log information about the weights sparsity"""
    for logger in loggers:
        logger.log_weights_sparsity(model, epoch)


### **********************************
### Used for calling logger function. [end]
### **********************************



### ***************************
### ADMM utils function [start]
### ***************************
"""
source https://github.com/hongyi-zhang/mixup/blob/80000cea340bf829a52481ae45a317a487ce2deb/cifar/utils.py#L17
"""

def mixup_data(x, y, alpha=1.0):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
           (1 - lam) * criterion(pred, y_b, smooth=smooth)


class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    """
    Modify from https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/20f355eb655bad40195ae302b9d8036716be9a23/train.py#L33
    """
    # Calculate cross entropy loss, apply label smoothing if needed. '''
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

class GradualWarmupScheduler(_LRScheduler):
    # source: https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_iter, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
### ***************************
### ADMM utils function [end]
### ***************************