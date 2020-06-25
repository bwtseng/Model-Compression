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

class ADMMPrunerBase(_ParameterPruner):
    def __init__(self, name):
        super().__init__(name)
        self.ADMM_U = {}
        self.ADMM_Z = {} 
        #self.final_sparsity = final_sparsity
        #assert final_sparsity > initial_sparsity


    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        #if param_name in self.prune_ratios:           
        # Below line has a problem: if we 
        above_threshold, pruned_weight = self.weight_pruning(param, self.prune_ratios[param_name])
        # Below is certainly masked the weights ....?
        #param.data = pruned_weight
        zeros_mask_dict[param_name].mask = above_threshold   
        zeros_mask_dict[param_name].mask_pruner = "ADMM" 
        #print(zeros_mask_dict[param_name].mask_pruner)
        #self._set_param_mask_by_sparsity_target(param, param_name, zeros_mask_dict, target_sparsity, meta['model'])
        #zeros_mask_dict[param_name].mask = SparsityLevelParameterPruner.create_mask(param, desired_sparsity)
    # From AGP!
    #def _set_param_mask_by_sparsity_target(self, param, param_name, zeros_mask_dict, target_sparsity, model=None):
    #    """Set the parameter mask using a target sparsity. Override this in subclasses"""
    #    raise NotImplementedError

    def init(self, model):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file          

        """          
        #self.prune_ratios = self.prune_ratios
        #self.rho = self.rho
        #self.sparsity_type = self.sparsity_type
        self.rhos = {}
        #self.model_named_parameters = model.named_parameters()
        #self.model_named_parameters = {}
        #for (name, W) in self.model_named_parameters:

        for (name, W) in model.named_parameters():
            #self.model_named_parameters[name] = W
            if name not in self.prune_ratios:
                continue
            self.ADMM_U[name] = torch.zeros(W.shape).cuda() # add U 
            self.ADMM_Z[name] = torch.Tensor(W.shape).cuda() # add Z

        for k in self.prune_ratios.keys():
             self.rhos[k] = self.rho  # this version we assume all rhos are equal    

        #if self.masked_progressive:
        #    self.masking()

    def admm_initialization(self, model, zeros_mask_dict):
        #if not config.admm:
        #    return
        for name, W in model.named_parameters():
            if name in self.prune_ratios:
                print(name)
                #print(True) make sure that name in pruning ratio
                mask, updated_Z = self.weight_pruning(W, self.prune_ratios[name]) # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
                self.ADMM_Z[name] = updated_Z
                zeros_mask_dict[name].mask = mask

    def admm_update(self, model, epoch, batch_idx, zeros_mask_dict):
        # This is the function for updation all the paramter rho and mu, which follows the 
        # lemma from original paper!
        #if not config.admm:
        #    return
        # sometimes the start epoch is not zero. It won't be valid if the start epoch is not 0
        
        if epoch == 0 and batch_idx == 0:
            self.admm_initialization(model, zeros_mask_dict)  # intialize Z, U variable # This line solves my problem

        if epoch != 0 and epoch % self.admm_epoch == 0 and batch_idx == 0:
            for name, W in model.named_parameters():
                if name not in self.prune_ratios:
                    continue
                if self.multi_rho:
                    self.admm_multi_rho_scheduler(name) # call multi rho scheduler every admm update
                self.ADMM_Z[name] = W + self.ADMM_U[name] # Z(k+1) = W(k+1)+U[k]
                mask, _Z = self.weight_pruning(self.ADMM_Z[name], self.prune_ratios[name]) #  equivalent to Euclidean Projection
                zeros_mask_dict[name].mask = mask
                self.ADMM_Z[name] = _Z
                self.ADMM_U[name] = W - self.ADMM_Z[name] + self.ADMM_U[name] # U(k+1) = W(k+1) - Z(k+1) +U(k)
    
    def append_admm_loss(self, model, loss, sparsity_type):
        '''
        append admm loss to cross_entropy loss
        Args:
            args: configuration parameters
            model: instance to the model class
            ce_loss: the cross entropy loss
        Returns:
            ce_loss(tensor scalar): original cross enropy loss
            admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
            ret_loss(scalar): the mixed overall loss
        '''

        # Below is the code follwing the fomula shown in lemma (5), this function means that each 
        # theta i should have it cost whist summation all of them.  
        
        admm_loss = {}
        if sparsity_type != "quantization":
            for name, W in model.named_parameters():  ## initialize Z (for both weights and bias)
                if name not in self.prune_ratios:
                    continue
                admm_loss[name] = 0.5 * self.rhos[name]*(torch.norm(W - self.ADMM_Z[name] + 
                                                                    self.ADMM_U[name], p=2)**2)

        else:
            for name, W in model.named_parameters():
                if name not in ADMM.number_bits:
                    continue
                admm_loss[name] = 0.5 * self.rhos[name]*(torch.norm(W-ADMM.alpha[name]*ADMM.ADMM_Q[name] + 
                                                        ADMM.ADMM_U[name], p=2)**2)
        mixed_loss = 0
        #mixed_loss += loss
        for name, ad_loss in admm_loss.items():
            mixed_loss += ad_loss
        return loss, mixed_loss #or reutrn admm loss function dict

    def admm_multi_rho_scheduler(self, name):
        """
        It works better to make rho monotonically increasing
        """
        self.rhos[name]*=1.3  # choose whatever you like

    def admm_adjust_learning_rate(self, optimizer, epoch):
        """ (The pytorch learning rate scheduler)
        Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        """
        For admm, the learning rate change is periodic.
        When epoch is dividable by admm_epoch, the learning rate is reset
        to the original one, and decay every 3 epoch (as the default 
        admm epoch is 9)

        """
        #admm_epoch = self.admm_epoch
        lr = None
        if epoch % self.admm_epoch == 0:
            lr = self.initial_lr 
        else:
            admm_epoch_offset = epoch % self.admm_epoch

            admm_step =  self.admm_epoch / 3  # roughly every 1/3 admm_epoch. 
            
            lr = self.initial_lr *(0.1 ** (admm_epoch_offset // admm_step))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class ADMMPruner(ADMMPrunerBase): 
    def __init__(self, name, class_name, pruning_ratio, rho, sparsity_type, masked_progressive, 
                 admm_epoch, initial_lr, multi_rho, model):
        
        super().__init__(name)
        self.class_name = class_name
        self.prune_ratios = pruning_ratio
        self.rho = rho
        self.sparsity_type = sparsity_type
        self.multi_rho = multi_rho
        self.zero_masking = None
        self.masks = None
        self.masked_progressive = masked_progressive
        self.masked_retrain = False
        self.admm_epoch = admm_epoch
        self.initial_lr = initial_lr 
        self.init(model)
        

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        #if param_name not in self.prune_ratios.keys():
        if param_name not in self.prune_ratios:
            return
        super().set_param_mask(param, param_name, zeros_mask_dict, meta)

    #def _set_param_mask_by_sparsity_target(self, param, param_name, zeros_mask_dict, target_sparsity, model=None):
    #    zeros_mask_dict[param_name].mask = SparsityLevelParameterPruner.create_mask(param, target_sparsity)
    

    def weight_pruning(self, weight, prune_ratio):
        """ 
        weight pruning [irregular,column,filter]
        Args: 
            weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
            prune_ratio (float between 0-1): target sparsity of weights
        
        Returns:
            mask for nonzero weights used for retraining
            a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero 

        """     
        # Important line, do not comment it!
        weight = weight.cpu().detach().numpy()            # convert cpu tensor to numpy     
        percent = prune_ratio * 100         
        
        if (self.sparsity_type == "irregular"):
            # Irregular is just the fine-grained pruning that can be applied to out current task.
            # weight = weight.cpu().detach().numpy() 
            weight_temp = np.abs(weight)   # a buffer that holds weights with absolute values     
            percentile = np.percentile(weight_temp, percent)   # get a value for this percentitle
            under_threshold = weight_temp < percentile     
            above_threshold = weight_temp > percentile     
            above_threshold = above_threshold.astype(np.float32) # has to convert bool to float32 for numpy-tensor conversion     
            weight[under_threshold] = 0     
            return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()

        elif (self.sparsity_type == "column"):
            #weight = weight.cpu().detach().numpy() 
            shape = weight.shape          
            weight2d = weight.reshape(shape[0], -1)
            shape2d = weight2d.shape
            column_l2_norm = LA.norm(weight2d, 2, axis = 0)
            percentile = np.percentile(column_l2_norm, percent)
            under_threshold = column_l2_norm < percentile
            above_threshold = column_l2_norm > percentile
            weight2d[:,under_threshold] = 0
            above_threshold = above_threshold.astype(np.float32)
            expand_above_threshold = np.zeros(shape2d, dtype=np.float32)          
            for i in range(shape2d[1]):
                expand_above_threshold[:,i] = above_threshold[i]
            expand_above_threshold = expand_above_threshold.reshape(shape)
            weight = weight2d.reshape(shape)          
            
            return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()

        elif (self.sparsity_type == "filter"):
            #weight = weight.cpu().detach().numpy() 
            shape = weight.shape
            weight2d = weight.reshape(shape[0], -1)
            shape2d = weight2d.shape
            row_l2_norm = LA.norm(weight2d, 2, axis = 1)
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
            return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()

        elif self.sparsity_type == "channel":
            # Remember not to prune the input channel of the first layer.
            shape = weight.shape
            # Fully connected layers only need to sample the norm from each column.
            # Convolution layer should reshape the matrix into the shape (C_in, C_out, W, H) at first,
            # then sample it norm according the axis=0, we can take distiller's function as the reference.
            # "norm.py"  channel norm computation for the module computation.
            if len(shape) == 2:
                weight_t = weight.transpose(1, 0)
            else: 
                weight_t = weight.transpose(1, 0 ,2, 3)
            weight2d = weight_t.reshape(shape[1], -1)
            shape2d = weight2d.shape
            row_l2_norm = LA.norm(weight2d, 2, axis=1)
            percentile = np.percentile(row_l2_norm, percent)
            under_threshold = row_l2_norm < percentile
            above_threshold = row_l2_norm > percentile
            # Masked weight is the second output of this function.
            weight2d[under_threshold, :] = 0  
            above_threshold = above_threshold.astype(np.float32)
            expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
            
            for i in range(shape2d[0]):
                expand_above_threshold[i,:] = above_threshold[i]
            
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
            return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()

        elif (self.sparsity_type == "bn_filter"):
            weight = weight.cpu().detach().numpy() 
            ## bn pruning is very similar to bias pruning
            weight_temp = np.abs(weight)
            percentile = np.percentile(weight_temp,percent)
            under_threshold = weight_temp < percentile     
            above_threshold = weight_temp > percentile     
            above_threshold = above_threshold.astype(np.float32) # has to convert bool to float32 for numpy-tensor conversion     
            weight[under_threshold] = 0     
            return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
        else:
            raise SyntaxError("Unknown sparsity type")
    
    # ****************
    # No longer used ?
    # ****************
    # For Progressive usage.
    def zero_masking(self):
        masks = {}
        for name, W in self.model_named_parameters.items():  ## no gradient for weights that are already zero (for progressive pruning and sequential pruning)
            if name in self.prune_ratios:
                w_temp = W.cpu().detach().numpy()
                indices = (w_temp != 0)
                indices = indices.astype(np.float32)            
                masks[name] = torch.from_numpy(indices).cuda()
        self.zero_masks = masks
    
    # For masking, use it before_retrain!!
    def masking(self, model, zeros_mask_dict):
        for name, W in model.named_parameters():
            if name in self.prune_ratios:           
                above_threshold, pruned_weight = self.weight_pruning(W, self.prune_ratios[name])
                W.data = pruned_weight
                zeros_mask_dict[name].mask = above_threshold        
        #self.masks = masks
        #return masks
    
