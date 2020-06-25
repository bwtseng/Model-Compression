from .pruner import _ParameterPruner
from .level_pruner import SparsityLevelParameterPruner
from .ranked_structures_pruner import *
from distiller.utils import *
from functools import partial
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
from collections import OrderedDict


class ChannelPrunerBase(_ParameterPruner):
    
    def __init__(self, name):
        super().__init__(name)

    def init(self, model):
        self.name_to_ind = OrderedDict()
        self.ind_to_name = OrderedDict()
        self.meta_data = OrderedDict()
        self.layers = nn.ModuleList()
        self.name_to_next_name = OrderedDict()

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        #if param_name in self.prune_ratios:           
        # Below line has a problem: if we 
        # Below is certainly masked the weights ....?
        #param.data = pruned_weight
        zeros_mask_dict[param_name].mask = above_threshold   
        zeros_mask_dict[param_name].mask_pruner = "Channel" 
        #print(zeros_mask_dict[param_name].mask_pruner)
        #self._set_param_mask_by_sparsity_target(param, param_name, zeros_mask_dict, target_sparsity, meta['model'])
        #zeros_mask_dict[param_name].mask = SparsityLevelParameterPruner.create_mask(param, desired_sparsity)
    # From AGP!
    #def _set_param_mask_by_sparsity_target(self, param, param_name, zeros_mask_dict, target_sparsity, model=None):
    #    """Set the parameter mask using a target sparsity. Override this in subclasses"""
    #    raise NotImplementedError

    def apply_layer_weight(self, model, keep_inds, layer_name):



class ADMMPruner(ADMMPrunerBase): 
    def __init__(self, name, class_name, pruning_ratio, sparsity_type, model):
        super().__init__(name)
        self.class_name = class_name
        self.prune_ratios = pruning_ratio
        self.sparsity_type = sparsity_type
        self.zero_masking = None
        self.masks = None
        

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        #if param_name not in self.prune_ratios.keys():
        if param_name not in self.prune_ratios:
            return
        super().set_param_mask(param, param_name, zeros_mask_dict, meta)


