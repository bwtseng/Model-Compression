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
from sklearn.linear_model import Lasso, LassoLars, LinearRegression
from modules import eltwise
import math 
msglogger = logging.getLogger(__name__)


class Channel_pruner_FMR(_ParameterPruner):
    # Bowei convert the config format into ADMM's patter, while bulding a new policy to enable it.
    def __init__(self, name, pruning_ratio, use_lasso,  n_points_per_fm, #desired_sparsity, weights, # or weights?
                 model, arch, group_dependency=None, kwargs=None, 
                 magnitude_fn=distiller.norms.l1_norm, group_size=1, rounding_fn=math.floor):
        # Add forward_fn or dataloader to the channel pruner policy.
        assert magnitude_fn is not None
        #super().__init__(name, desired_sparsity, weights, group_dependency,
        #                 group_size=group_size, rounding_fn=rounding_fn, noise=ranking_noise)    
        super().__init__(name)   
        self.arch = arch 
        self.group_size = group_size
        self.rounding_fn = rounding_fn
        #self.noise = noise
        self.model = model 
        self.use_lasso = use_lasso
        self.pruning_ratios = pruning_ratio
        self.magnitude_fn = magnitude_fn    
        # Below list will be revised by local user if their designed model is with residual connections.
        #connection_exist_list = ['resnet', 'inception']  
        self.connection = False
        if arch == 'mobilenet_v2' or arch.startswith(('resnet', 'inception', 'efficient')):
            self.connection = True 
        #self.weights = weights
        #data_processing(dataset, data_dir, batch_size, workers=workers=, split_ratio=0.1)
        #self.desired_sparsity = desired_sparsity     
        # Indicate that the pruning is true applied in a certain layer in the DNN, 
        # thus we can start weights reconstruct to compensate the error among different layers.
        self.is_pruned = False
        self.assign_layer_fq_names(model)
        self.modules, self.non_pruned_modules = self.build_module_list(model)
        self.n_points_per_fm = n_points_per_fm
        self.layer_indice = OrderedDict()
        # TODO should think what this constraint is reasonable.  
        if self.connection:
            # Build dictionary and inverse-key dict for residual block.
            self.name_to_ind, self.ind_to_name, self.meta_data, \
                name_to_next_name, self.name_transfer_dict = self.build_dictionary(model, self.arch, self.modules)
            
            self.reverse_transfer_dict = OrderedDict()
            for module_name, local_name in self.name_transfer_dict.items():
                self.reverse_transfer_dict[local_name] = module_name

            self.hook_fn = partial(self.cache_featuremaps_fwd_hook, n_points_per_fm=self.n_points_per_fm, 
                                   layer_indice=self.layer_indice, arch=self.arch, 
                                   name_transfer_dict=self.name_transfer_dict,
                                   reverse_transfer_dict=self.reverse_transfer_dict)
        else: 
            self.hook_fn = partial(self.cache_featuremaps_fwd_hook, n_points_per_fm=self.n_points_per_fm, 
                                   layer_indice=self.layer_indice, arch=arch)

        self.acceptance_criterion = partial(self.acceptance_criterion, mod_names=self.modules)
        
        # import validate, how to incorporate it into our function?
        #x = torch.rand((2, 3, 224, 224))
        #self.forward_fn = partial(foward_fn, model=model, x=x)
        #forward_arg = partial(foward_fn, model=model, x=x)
        # First collect all the groud truth feature maps from original model. (Time-consuming?)
        #intermediate_fms = {"output_fms": dict(), "input_fms": dict()}
        """
        hook_handles = []
        m = list(model.modules())
        m = self.find_module_by_fq_name(model, 'layers.0.conv1')
        intermediate_fms['output_fms'][m.distiller_name] = []
        intermediate_fms['input_fms'][m.distiller_name] = []
        hook_handles.append(m.register_forward_hook(partial(self.cache_featuremaps_fwd_hook, 
                                                            intermediate_fms=intermediate_fms,
                                                            n_points_per_fm=5)))
        self.forward_fn()
        """
        #print(self.name_to_ind)
        #print(self.name_transfer_dict)
        
        # For testing useage:
        #self.collect_intermediate_featuremap_samples(model, self.forward_fn, 
        #                                             self.acceptance_criterion, self.hook_fn)
        #recollect_featuremap_samples(modules[5], forward_arg, acceptance_criterion, hook_fn)

    @staticmethod
    def assign_layer_fq_names(container, name=None):
        """
        Assign human-readable names to the modules (layers).

        Sometimes we need to access modules by their names, and we'd like to use
        fully-qualified names for convenience.
        """
        for name, module in container.named_modules():
            #print(name)
            module.distiller_name = name

    def build_module_list(self, model):
        # Refer to the distiller AMC script
        # We can load the module list using the config file! 
        module_list = []
        non_prune_module = []
        # how to add add module to it?
        for name, module in model.named_modules(): 
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, eltwise.EltwiseAdd)):
                module_list.append(name)
                name_weight = name + ".weight"
                if name_weight not in self.pruning_ratios:
                    non_prune_module.append(name)

        return module_list, non_prune_module

    # THis function includes in the pruner class, staticmethod. 
    #@staticmethod
    def cache_featuremaps_fwd_hook(self, module, input, output, intermediate_fms, n_points_per_fm, 
                                   layer_indice, arch, name_transfer_dict=None, reverse_transfer_dict=None):
        """
        Create a cached dictionary of each layer's input and output feature-maps.

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
            x_unfold = F.unfold(x, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding)
            return x_unfold

        # Sample random (uniform) points in each feature-map.
        # This method is biased toward small feature-maps.
        #print(module.distiller_name)
        if isinstance(module, (torch.nn.Conv2d, eltwise.EltwiseAdd)):
            # Only resnet needs this if condition:
            if module.distiller_name in layer_indice: 

                randx, randy = layer_indice[module.distiller_name]
            
            else:
                randx = np.random.randint(0, output.size(2), n_points_per_fm)
                randy = np.random.randint(0, output.size(3), n_points_per_fm)

        # Should give another options that can use whole features map as the input/output of regression problem, no need it, since Prof. Han also adopt this implementations.
        X = input[0]
        B = X.shape[0]
        C = X.shape[1]

        if isinstance(module, torch.nn.Linear):
            X = X.detach().cpu().clone()
            Y = output.detach().cpu().clone()

        elif isinstance(module, eltwise.EltwiseAdd):
            if self.connection:
                layer_name_with_indice = name_transfer_dict[module.distiller_name]
                conv_name = layer_name_with_indice.split("_")[0]

                # TODO we should consider more deeper residual network. the str("1") is no longer useful as layer_num >50
                randx, randy = layer_indice[reverse_transfer_dict[conv_name+"_1"]]

            #X = X.detach().cpu().clone()
            # Bowei revision, in which I follow the same computation as getting the feature maps Y.
            X = X[:, :, randx, randy]
            X = X.permute([0, 2, 1]).contiguous()
            X = X.view(-1, X.shape[-1]).cpu().clone()
            #Y = output.detach().cpu().clone()
            Y = output[:, :, randx, randy].detach().cpu().clone()

        elif module.kernel_size == (1, 1):
            if self.connection:
                layer_name_with_indice = name_transfer_dict[module.distiller_name]
                conv_name = layer_name_with_indice.split("_")[0]
                randx, randy = layer_indice[reverse_transfer_dict[conv_name+"_0"]]
            # Bowie revise the code here:
            w_out = output.size(2)
            pts = randx * w_out + randy
            X = X[:, :, randx, randy].detach().cpu().clone()
            # Form pytorch AMC:
            #X = im2col(X.detach().cpu().clone(), module)
            #X = X[:, :, pts].detach().cpu().clone()
            #X = X.view(B, C, 1, 1, -1)
            #X = X.permute([0, 4, 1, 2, 3]).contiguous()
            #X = X.view(-1, C, module.kernel_size[0], module.kernel_size[1])
            Y = output[:, :, randx, randy].detach().cpu().clone()
            layer_indice[module.distiller_name] = (randx, randy) 
        
        else:
            w, h = X.size(2), X.size(3)
            X = im2col(X.detach().cpu().clone(), module)#.squeeze()
            w_out = output.size(2)
            pts = randx * w_out + randy
            X = X[:, :, pts].detach().cpu().clone()
            Y = output[:, :, randx, randy].detach().cpu().clone()
            # Bowei adds the follow code from pytorch-AMC, more specifially, it's conv_input/output function.
            #X = X.view(B, C, module.kernel_size[0], module.kernel_size[1], -1)
            #X = X.permute([0, 4, 1, 2, 3]).contiguous()
            #X = X.view(-1, C, module.kernel_size[0], module.kernel_size[1])
            
            # We can save the indices in this way, since the output and input dimension are not changed during training.
            layer_indice[module.distiller_name] = (randx, randy) 

        # Preprocess the outputs: transpose the batch and channel dimensions, create a flattened view, and transpose.
        # The outputs originally have shape: (batch size, num channels, feature-map width, feature-map height).
        Y = Y.view(Y.size(0), Y.size(1), -1)
        Y = Y.transpose(2, 1)
        Y = Y.contiguous().view(-1, Y.size(2))

        # Transfer to numpy so as to get higher efficiency!
        Y = Y.cpu().numpy()
        X = X.cpu().numpy()
        #print(Y.shape)
        #print(X.shape)
        intermediate_fms['output_fms'][module.distiller_name].append(Y)
        intermediate_fms['input_fms'][module.distiller_name].append(X)
        
        # This will be used in recoolect function, since the smaple indice should keep the same.

        #layer_name_with_indice = self.name_transfer_dict[module.distiller_name]
        #self.layer_indice[module.distiller_name] = (randx, randy) 

        """
        DEBUG FUNCTION :

        X = X.view(X.size(0), -1, np.prod(module.kernel_size), X.size(2))
        print(X.shape)
        #X = X[:, binary_map, :, :]
        X = X.view(X.size(0), -1, X.size(3))
        X = X.transpose(1, 2)
        X = X.contiguous().view(-1, X.size(2))
        print(X.shape)
        new_w = _least_square_sklearn(X, Y)
        new_w = torch.from_numpy(new_w)
        print(new_w.shape)
        retained_channel = module.in_channels
        new_w = new_w.contiguous().view(-1, retained_channel, module.kernel_size[0], module.kernel_size[1])
        print(new_w.shape)
        print(module)
        """

    def basic_featuremaps_caching_fwd_hook(module, input, output, intermediate_fms):
        """
        A trivial function to cache input/output feature-maps
        
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

    @staticmethod
    def recollect_featuremap_samples(model, forward_fn, modules_filter_fn, fm_caching_fwd_hook=basic_featuremaps_caching_fwd_hook):
        # This function is designed for recollection the "input" features after last layer is already pruned.
        from functools import partial
        intermediate_fms = {"output_fms": dict(), "input_fms": dict()}
        hook_handles = []
        def install_io_collectors(m, intermediate_fms):
            if modules_filter_fn(m):
                intermediate_fms['output_fms'][m.distiller_name] = []
                intermediate_fms['input_fms'][m.distiller_name] = []
                hook_handles.append(m.register_forward_hook(partial(fm_caching_fwd_hook, 
                                                                    intermediate_fms=intermediate_fms)))
        """
        if modules_filter_fn(module):
            intermediate_fms['output_fms'][module.distiller_name] = []
            intermediate_fms['input_fms'][module.distiller_name] = []
            hook_handles.append(module.register_forward_hook(partial(fm_caching_fwd_hook, 
                                                                intermediate_fms=intermediate_fms)))
        """

        # Register to the forward hooks, then run the forward-pass and collect the data
        msglogger.warning("==> Collecting single input feature-map pairs")
        model.apply(partial(install_io_collectors, intermediate_fms=intermediate_fms))
        forward_fn()
        # Unregister from the forward hooks
        for handle in hook_handles:
            handle.remove()

        # We now need to concatenate the list of feature-maps to torch tensors.
        msglogger.info("Concatenating Input FMs...")
        #outputs = model.intermediate_fms['output_fms']
        inputs = model.intermediate_fms['input_fms']

        for (layer_name, X), Y in zip(intermediate_fms['input_fms'].items(), intermediate_fms['output_fms'].values()):                
            #inputs[layer_name] = np.concatenate(X, axis=0)
            inputs[layer_name] = torch.cat(X, dim=0)
            #outputs[layer_name] = torch.cat(Y, dim=0)

        msglogger.warning("<== Single collector is Done.")
        del intermediate_fms    

    @staticmethod
    def _param_name_2_layer_name(param_name):
        """
        Convert a weights tensor's name to the name of the layer using the tensor.   
        By convention, PyTorch modules name their weights parameters as self.weight
        (see for example: torch.nn.modules.conv) which means that their fully-qualified 
        name when enumerating a model's parameters is the modules name followed by '.weight'.
        We exploit this convention to convert a weights tensor name to the fully-qualified 
        module name.
        """
        return param_name[:-len('.weight')]


    # Example of module_filter_fn used in the Distiller's AMC
    @staticmethod
    def acceptance_criterion(m, mod_names):
        # Collect feature-maps only for Conv2d layers and fc layers, if they are in our modules list.
        return isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, eltwise.EltwiseAdd)) and m.distiller_name in mod_names

    @staticmethod
    def find_module_by_fq_name(model, fq_mod_name):
        """
        
        Given a module's fully-qualified name, find the module in the provided model.
        A fully-qualified name is assigned to modules in function assign_layer_fq_names.
        Arguments:
            model: the model to search
            fq_mod_name: the module whose name we want to look up
        Returns:
            The module or None, if the module was not found.
        
        """
        for module in model.modules():
            if hasattr(module, 'distiller_name') and fq_mod_name == module.distiller_name:
                return module
        return None


    @staticmethod
    def collect_intermediate_featuremap_samples(model, forward_fn, module_filter_fn, 
                                                fm_caching_fwd_hook=basic_featuremaps_caching_fwd_hook): #cache_featuremaps_fwd_hook
        '''
        Collect pairs of input/output feature-maps (all intermediate layers)!
        **
        Some Note here: 
            foward_fn: feed validation set and get the feature maps from hook, and services argument is just a nametuple which includes three callable function.
            module_filter_fn: check layer's property
            fm_caching_fwd_hood: your register function.
        **
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
        # *****
        # why ?
        # *****
        #self.assign_layer_fq_names(model)
        hook_handles = []
        intermediate_fms = {"output_fms": dict(), "input_fms": dict()}
        # Get all the feature maps at the same time.
        model.apply(partial(install_io_collectors, intermediate_fms=intermediate_fms))
        forward_fn() # In distiller's example, use partial to wrap this foward function

        # Unregister from the forward hooks
        for handle in hook_handles:
            handle.remove()

        # We now need to concatenate the list of feature-maps to torch tensors.
        msglogger.info("Concatenating FMs...")
        model.intermediate_fms = {"output_fms": dict(), "input_fms": dict()}
        outputs = model.intermediate_fms['output_fms']
        inputs = model.intermediate_fms['input_fms']

        for (layer_name, X), Y in zip(intermediate_fms['input_fms'].items(), intermediate_fms['output_fms'].values()):                
            #inputs[layer_name] = np.concatenate(X, axis=0)
            #outputs[layer_name] = np.concatenate(Y, axis=0)
            inputs[layer_name] = torch.cat(X, dim=0)
            outputs[layer_name] = torch.cat(Y, dim=0)

        msglogger.warning("<== Done.")
        del intermediate_fms 

    # example of forward_fn:
    #validate_fn = partial(_validate, test_loader=val_loader, criterion=self.criterion,
    #                              loggers=None, args=None, activations_collectors=None)
    @staticmethod
    def rank_and_prune_channels(fraction_to_prune, param, param_name=None, zeros_mask_dict=None, 
                                use_lasso=False, model=None, binary_map=None, 
                                magnitude_fn=distiller.norms.l1_norm, group_size=1, rounding_fn=math.floor,
                                noise=0):

        assert binary_map is None
        if binary_map is None:
            # Find the module representing this layer
            # I have already assined layer name in the __init__ function.
            # distiller.assign_layer_fq_names(model)
            layer_name = self._param_name_2_layer_name(param_name)
            # Can apply transfer dictionary here, if it is laster convolution layer,
            # Regenerate the input feature by finding the add module. 
            # find the new input here.
            conv = self.find_module_by_fq_name(model, layer_name)
            group_size = conv.group_size
            try:
                Y = model.intermediate_fms['output_fms'][layer_name]
                X = model.intermediate_fms['input_fms'][layer_name]
            
            except AttributeError:
                raise ValueError("To use FMReconstructionChannelPruner you must first collect input statistics")            
            
            # Two option, we can apply Lasso regresssion to determine the indice whilst converting them
            # to the binary map here. (May add the argument here.)
            if not use_lasso:
                bottomk_channels, channel_mags = distiller.norms.rank_channels(param, group_size, magnitude_fn,
                                                                           fraction_to_prune, rounding_fn, noise)
                threshold = bottomk_channels[-1]
                binary_map = channel_mags.gt(threshold)

                # These are the indices of channels we want to keep
                indices = binary_map.nonzero().squeeze()

            # TODO: this little piece of code can be refactored
            if bottomk_channels is None:
                # Empty list means that fraction_to_prune is too low to prune anything
                return

            else: 
                #input_fms, output_fms, weight, desired_sparsity, 
                #                alpha=1e-4, tolerance=0.02, debug=False):
                binary_map = self.lasso_regression_solver(X, Y, param, fraction_to_prune)

            if len(indices.shape) == 0:
                indices = indices.expand(1)

            op_type = 'conv' if param.dim() == 4 else 'fc'
            # We need to remove the chosen weights channels.  Because we are using 
            # min(MSE) to compute the weights, we need to start by removing feature-map 
            # channels from the input.  Then we perform the MSE regression to generate
            # a smaller weights tensor.
            if op_type == 'fc':
                X = X[:, binary_map]
                
            elif conv.kernel_size == (1, 1):
                X = X[:, binary_map, :]
                X = X.transpose(1, 2)
                X = X.contiguous().view(-1, X.size(2))

            else:
                # X is (batch, ck^2, num_pts)
                # we want:   (batch, c, k^2, num_pts)
                X = X.view(X.size(0), -1, np.prod(conv.kernel_size), X.size(2))
                X = X[:, binary_map, :, :]
                X = X.view(X.size(0), -1, X.size(3))
                X = X.transpose(1, 2)
                X = X.contiguous().view(-1, X.size(2))

            # Approximate the weights given input-FMs and output-FMs
            new_w = _least_square_sklearn(X, Y)
            new_w = torch.from_numpy(new_w) # shape: (num_filters, num_non_masked_channels * k^2)
            cnt_retained_channels = binary_map.sum()

            if op_type == 'conv':
                # Expand the weights back to their original size,
                new_w = new_w.contiguous().view(param.size(0), cnt_retained_channels, param.size(2), param.size(3))

                # Copy the weights that we learned from minimizing the feature-maps least squares error,
                # to our actual weights tensor.
                # This is really smart....
                param.detach()[:, indices, :,   :] = new_w.type(param.type())
            else:
                param.detach()[:, indices] = new_w.type(param.type())

        if zeros_mask_dict is not None:
            binary_map = binary_map.type(param.type())
            if op_type == 'conv':
                zeros_mask_dict[param_name].mask, _ = distiller.thresholding.expand_binary_map(param,
                                                                                               'Channels', binary_map)
                msglogger.info("FMReconstructionChannelPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                               param_name,
                               distiller.sparsity_ch(zeros_mask_dict[param_name].mask),
                               fraction_to_prune, binary_map.sum().item(), param.size(1))
            else:
                msglogger.error("fc sparsity = %.2f" % (1 - binary_map.sum().item() / binary_map.size(0)))
                zeros_mask_dict[param_name].mask = binary_map.expand(param.size(0), param.size(1))
                msglogger.info("FMReconstructionChannelPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                               param_name,
                               distiller.sparsity_cols(zeros_mask_dict[param_name].mask),
                               fraction_to_prune, binary_map.sum().item(), param.size(1))
        return binary_map


    def prune_kernel(self, fraction_to_prune, param, param_name=None, zeros_mask_dict=None, 
                     model=None, binary_map=None, magnitude_fn=distiller.norms.l1_norm, 
                     group_size=1, rounding_fn=math.floor):

        # sequentiallly prune the convolution layer!
        # Recollect the feature map of previous layer which is pruned at first.. 
        layer_name = self._param_name_2_layer_name(param_name)
        print("Start layer name {}".format(layer_name))
        conv = self.find_module_by_fq_name(model, layer_name)
        group = conv.groups

        if group != 1:
            # TODO I need to understand how to apply the weights to group convolution module.
            return 

        if not self.is_pruned: 
            if fraction_to_prune == 1: 
                # We are able to get more efficient in case that the first pruned layer doesn't need to collect feature maps. 
                return 
            else: 
                self.is_pruned = True

        # TODO below line should be more flexible and well defined.
        # Our scope aims to reconstruct downsample layer and conv1 layer by specifying hte convolution layer 0 in yaml file.
        # Finally, we register all the foward hook to the module so as to recollect all the input features in the pruning stage.
        self.recollect_featuremap_samples(model, self.forward_fn, self.acceptance_criterion, self.hook_fn)
        # Note we can not modify output_fms from each layer, it's the ground truth of least square solution.
        X = model.intermediate_fms['input_fms'][layer_name]
        Y = model.intermediate_fms['output_fms'][layer_name]
        # It's designed for residual block.
        consider_downsample = False
        if self.connection:
            if layer_name in self.name_transfer_dict.keys():
                layer_name_with_indice = self.name_transfer_dict[layer_name]
                layer_indice = self.name_to_ind[layer_name_with_indice]
                print("There is a downsample layer here.")
                if layer_indice[1] == 0:
                    # May consider shortcut layer: DO NOT prune, just reconstruct the layer since previous is also pruned.
                    conv_num = layer_name_with_indice.split('_')[0]
                    if conv_num + '_downsample' in self.reverse_transfer_dict.keys(): 
                        layer_name_downsample = self.reverse_transfer_dict[conv_num + '_downsample']
                        conv_downsample = self.find_module_by_fq_name(model, layer_name_downsample)
                        #self.recollect_featuremap_samples(conv_downsample, self.forward_fn, self.acceptance_criterion, self.hook_fn)
                        #downsample_out_ch = self.meta_data[conv_num + '_downsample']["n"]
                        #downsample_in_ch = self.meta_data[conv_num + '_downsample']["c"]
                        downsample_out_ch = conv_downsample.in_channels
                        downsample_in_ch = conv_downsample.out_channels
                        Y_downsample =  model.intermediate_fms['output_fms'][layer_name_downsample]
                        X_downsample =  model.intermediate_fms['input_fms'][layer_name_downsample]
                        #conv_downsample = self.find_module_by_fq_name(model, layer_name) # From this, we attain the identity output and its input from residual connection
                        consider_downsample = True 
                    #prunable = False 

                elif layer_indice[1] == 1:
                    # Identity mapping will conduct here. We need change the module to the output of add module, 
                    # from original last layer of the convolution. 
                    # Change key and value pair to get the module name [Reverse dictionary]:
                    # Formulation:
                    # a': original indentity branch
                    # b': original conv output
                    # a : new identity branch
                    # b : new conv ouput
                    # we want a + b = a' + b' , but we can only control b,
                    # so we need to find b such that b = a' + b' - a
                    # From this, we need to find new pruned input from identity mapping and pruned input 
                    # from conv2 module. My idea is that the input of conv0 is same as identity branch under our implementation.
                    conv_num = layer_name_with_indice.split('_')[0]
                    if conv_num + '_add' in self.reverse_transfer_dict.keys(): 
                        layer_name_identity = self.reverse_transfer_dict[conv_num + '_add']
                        res_ind = self.find_module_by_fq_name(model, layer_name) # From this, we attain the identity output and its input from residual connection        
                        #self.recollect_featuremap_samples(res_ind, self.forward_fn, self.acceptance_criterion, self.hook_fn)
                        Y = model.intermediate_fms['output_fms'][layer_name_identity]
                        # subtract the input from downsample or conv 0, which is already pruned in the previous iterations
                        #pruned_input = model.intermediate_fms['input_fms'][self.reverse_transfer_dict[conv_num+"_0"]]
                        pruned_input = model.intermediate_fms['input_fms'][self.reverse_transfer_dict[conv_num+"_add"]]
                        #Y -= model.intermediate_fms['input_fms'][self.reverse_transfer_dict[conv_num+"_0"]]
                        Y -= pruned_input
                        Y = Y.detach().cpu().clone()
                        #self.recollect_featuremap_samples(conv, forward_fn, modules_filter_fn, fm_caching_fwd_hook)
                else:
                    pass
        
        X = torch.from_numpy(X).float().cpu()
        Y = torch.from_numpy(Y).float().cpu()
        def convert_ls_format(X, op_type, module=None):
            if op_type == 'fc':
                X = X
            elif module.kernel_size == (1,1):
                X = X.transpose(1, 2)
                X = X.contiguous().view(-1, X.size(2))
            
            else:
                print("Input size: {}".format(X.shape))
                # Critertion X is the tesnorflow with shape (batch, ck^2, num_pts)
                X = X.view(X.size(0), -1, np.prod(module.kernel_size), X.size(2))
                #X = X[:, binary_map, :, :]
                X = X.view(X.size(0), -1, X.size(3))
                X = X.transpose(1, 2)
                X = X.contiguous().view(-1, X.size(2))
            return X

        # stage 2, find the pruned index:
        if fraction_to_prune != 1:
            # Get index_from_channel
            if not self.use_lasso:
                bottomk_channels, channel_mags = distiller.norms.rank_channels(param, group_size, magnitude_fn,
                                                                            fraction_to_prune, rounding_fn, noise)
                threshold = bottomk_channels[-1]
                binary_map = channel_mags.gt(threshold)
                # These are the indices of channels we want to keep
                indices = binary_map.nonzero().squeeze()

            else: 
                #input_fms, output_fms, weight, desired_sparsity, 
                #                alpha=1e-4, tolerance=0.02, debug=False):
                # TODO Check wether the format is consistent with distiller's channel pruning output.
                indices = self.lasso_regression_solver(X, Y, param, fraction_to_prune)
                binary_map = torch.zeros(param.shape[1], dtype=bool)
                binary_map[indices] = True 
                bottomk_channels = True 


            # TODO: this little piece of code can be refactored
            if bottomk_channels is None:
                # Empty list means that fraction_to_prune is too low to prune anything
                return

            # weight_reconstruction
            if len(indices.shape) == 0:
                indices = indices.expand(1)
        
            op_type = 'conv' if param.dim() == 4 else 'fc'
            # We need to remove the chosen weights channels.  Because we are using 
            # min(MSE) to compute the weights, we need to start by removing feature-map 
            # channels from the input.  Then we perform the MSE regression to generate
            # a smaller weights tensor.
            if op_type == 'fc':
                X = X[:, binary_map]
                
            elif conv.kernel_size == (1, 1):
                X = X[:, binary_map, :]
                X = X.transpose(1, 2)
                X = X.contiguous().view(-1, X.size(2))

            else:
                # X is (batch, ck^2, num_pts)
                # we want:   (batch, c, k^2, num_pts)
                X = X.view(X.size(0), -1, np.prod(conv.kernel_size), X.size(2))
                X = X[:, binary_map, :, :]
                X = X.view(X.size(0), -1, X.size(3))
                X = X.transpose(1, 2)
                X = X.contiguous().view(-1, X.size(2))
                
            cnt_retained_channels = binary_map.sum()
        
        else:
            op_type = 'conv' if param.dim() == 4 else 'fc'
            cnt_retained_channels = param.shape[1] 
            # Compute downsample layer here.
            X = convert_ls_format(X, op_type, conv)
            if consider_downsample:
                X_downsample = convert_ls_format(X_downsample, op_type, conv_downsample)

                

        # Approximate the weights given input-FMs and output-FMs
        new_w = _least_square_sklearn(X, Y)
        new_w = torch.from_numpy(new_w) # shape: (num_filters, num_non_masked_channels * k^2)

        if op_type == 'conv':
            # Expand the weights back to their original size,
            # TODO　Should revise here.
            new_w = new_w.contiguous().view(param.size(0), cnt_retained_channels, param.size(2), param.size(3))
            print("new_w_shape {}.".format(new_w.shape))
            if consider_downsample: 
                new_w_downsample = _least_square_sklearn(X_downsample, Y_downsample)
                new_w_downsample = torch.from_numpy(new_w_downsample)
                new_w_downsample = new_w_downsample.contiguous().view(downsample_out_ch, downsample_in_ch, 1, 1)
                # how to apply this new weights to the model? use module.weight to assign?
                conv_downsample = self.find_module_by_fq_name(model, layer_name_downsample)
                conv_downsample.weight = torch.nn.Parameter(new_w_downsample)
            # Copy the weights that we learned from minimizing the feature-maps least squares error,
            # to our actual weights tensor.
            # This is really smart....
            if fraction_to_prune != 1:
                param.detach()[:, indices, :,   :] = new_w.type(param.type())
            else:
                param.detach()[:, :, :, :]= new_w.type(param.type())
                #pass
        else:
            if fraction_to_prune != 1:
                param.detach()[:, indices] = new_w.type(param.type())
            else: 
                param.detach()[:, :, :, :] = new_w.type(param.type())
                #pass
        
        if fraction_to_prune != 1:
            if zeros_mask_dict is not None:
                binary_map = binary_map.type(param.type())
                if op_type == 'conv':
                    zeros_mask_dict[param_name].mask, _ = distiller.thresholding.expand_binary_map(param,
                                                                                                    'Channels', binary_map)
                    msglogger.info("FMReconstructionChannelPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                                    param_name,
                                    distiller.sparsity_ch(zeros_mask_dict[param_name].mask),
                                    fraction_to_prune, binary_map.sum().item(), param.size(1))
                else:
                    msglogger.error("fc sparsity = %.2f" % (1 - binary_map.sum().item() / binary_map.size(0)))
                    zeros_mask_dict[param_name].mask = binary_map.expand(param.size(0), param.size(1))
                    msglogger.info("FMReconstructionChannelPruner - param: %s pruned=%.3f goal=%.3f (%d/%d)",
                                    param_name,
                                    distiller.sparsity_cols(zeros_mask_dict[param_name].mask),
                                    fraction_to_prune, binary_map.sum().item(), param.size(1))

        return binary_map
    # Copy from rank prune channel 
    # whilst modify the proup group function.


    def lasso_regression_solver(self, input_fms, output_fms, weight, desired_sparsity, 
                                alpha=1e-4, tolerance=0.02, debug=False):
        # We can do it follow ADMM's numpy strategy.
        # X shape: [B, c_in, 3, 3]
        # Y shape: [B, c_out]
        # W shape: [c_out, c_in, 3, 3]
        X = input_fms
        W = weight

        X = X.view(input_fms.shape[0], input_fms.shape[1], weight.shape[2], weight.shape[3], -1)
        X = X.permute([0, 4, 1, 2, 3]).contiguous()
        X = X.view(-1, input_fms.shape[1], weight.shape[2], weight.shape[3])

        X = X.cpu().detach().numpy()
        Y = output_fms.cpu().detach().numpy()
        W = weight.cpu().detach().numpy()

        num_samples = X.shape[0]  # num of training samples
        c_in = W.shape[1]  # num of input channels
        c_out = W.shape[0]  # num of output channels
        # select subset of training samples
        # subset_inds = np.random.choice(num_samples, min(400, num_samples // 20))

        subset_inds = np.random.choice(num_samples, 1)
        # sample and reshape X to [c_in, subset_size, 9]
        reshape_X = X.reshape([num_samples, c_in, -1])
        reshape_X = reshape_X[subset_inds].transpose([1, 0, 2])

        # reshape W to [c_in, 9, c_out]
        reshape_W = W.reshape((c_out, c_in, -1)).transpose([1, 2, 0])
        # reshape Y to [subset_size x c_out]
        reshape_Y = Y[subset_inds].reshape(-1)
        # product has size [subset_size x c_out, c_in]
        product = np.matmul(reshape_X, reshape_W).reshape((c_in, -1)).T

        # use LassoLars because it's more robust than Lasso
        solver = LassoLars(alpha=alpha, fit_intercept=False, max_iter=3000)
        
        def solve(alpha):
            """ Solve binary indices using Lasso regression"""
            solver.alpha = alpha
            solver.fit(product, reshape_Y)
            nonzero_inds = np.where(solver.coef_ != 0.)[0]
            nonzero_num = sum(solver.coef_ != 0.)
            return nonzero_inds, nonzero_num, solver.coef_

        tic = time.perf_counter()
        c_new = int(c_in * (1 - desired_sparsity))

        if c_new == c_in:
            keep_inds = np.arange(c_new)
            keep_num = c_new

        elif c_new == 0:
            keep_inds = np.array([0])
            keep_num = 1

        else:
            left = 0  # minimum alpha is 0, which means don't use lasso regularizer at all
            right = alpha
            # the left bound of num of selected channels
            lbound = np.clip(c_new - tolerance * c_in / 2, 1, None)
            # the right bound of num of selected channels
            rbound = c_new + tolerance * c_in / 2

            # increase alpha until the lasso can find a selection with size < c_new
            while True:
                _, keep_num, coef = solve(right)
                if keep_num < c_new:
                    break
                else:
                    right *= 2
                    if debug:
                        print("relax right to %.3f" % right)
                        print("expected %d channels, but got %d channels" % (c_new, keep_num))

            # shrink the alpha for less aggressive lasso regularization
            # if the selected num of channels is less than the lbound
            while True:
                keep_inds, keep_num, coef = solve(alpha)
                # print loss
                loss = 1 / (2 * float(product.shape[0])) * \
                    np.sqrt(np.sum((reshape_Y - np.matmul(product, coef)) ** 2, axis=0)) + \
                    alpha * np.sum(np.fabs(coef))

                if debug:
                    print('loss: %.3f, alpha: %.3f, feature nums: %d, '
                          'left: %.3f, right: %.3f, left_bound: %.3f, right_bound: %.3f' %
                          (loss, alpha, keep_num, left, right, lbound, rbound))
                
                # We can further control the pruning channel here.
                # and revise the algorthim to meet the sparsity requirement that user specify.
                if lbound <= keep_num <= rbound:
                    break
                elif abs(left - right) <= right * 0.1:
                    if lbound > 1:
                        lbound = lbound - 1
                    if rbound < c_in:
                        rbound = rbound + 1
                    left = left / 1.2
                    right = right * 1.2
            
                elif keep_num > rbound:
                    left = left + (alpha - left) / 2
            
                else:
                    right = right - (right - alpha) / 2
            
                if alpha < 1e-10:
                    break
                alpha = (left + right) / 2

        if debug:
            print('LASSO Regression time: %.2f s' % (time.perf_counter() - tic))
            print(c_new, keep_num)

        return keep_inds #, [keep_num / c_in] 

    def build_dictionary(self, net, arch, module_list):
        # If bottlenect (resnet 50, mobilenet v2, inception v4), residual_layer = 3
        # Different arch may use different dictionary function!
        
        # ********
        # should be more clear for the following code:
        # ********
        if arch.startswith('resnet'):
            # Should consider resenet 152....
            layer_num = int(arch[::-1][:2][::-1])
            print(layer_num)
            block_layer_num = 2 if layer_num < 50 else 3 
            block_layer_num += 1 # Module number, which excludes the skip_conv/shortcup layer.
            #net_name = []
            #for name, module in net.named_modules():
            #    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, eltwise.EltwiseAdd)):
            #        net_name.append(name)
            name_to_ind = OrderedDict()
            ind_to_name = OrderedDict()
            meta_data = OrderedDict()
            name_to_next_name = OrderedDict()
            modules = list(net.modules())
            # if residual net, we can not prune the fc layer (output layer).
            f_size = 224 
            last_layer = 'conv0'
            layer_index = 0 
            conv_list_record = []
            add_index = 0
            linear_index = 0
            skip_conv = 0
            name_transfer_dict = OrderedDict()
            for ind, module in enumerate(modules):
                #print(type(module))
                if type(module) == torch.nn.modules.conv.Conv2d:
                    stride = module.stride[0]
                    kernel_size = module.kernel_size[0]
                    if stride == 2: 
                        f_size /= 2
                    if layer_index == 0 : 
                        name =  "input_conv"
                        layer_index +=1
                
                    else:
                        if kernel_size == 1: # also called shorcut
                            print(layer_index)
                            #i = (layer_index-1) // block_layer_num
                            j = 2 if layer_num < 50 else 3 
                            name = 'conv'+str(i)+'_downsample'
                            name_to_ind[name] = (i, j)
                            ind_to_name[(i, j)] = name
                            name_transfer_dict[module_list[layer_index]] = name
                            layer_index += 1
                            skip_conv += 1
                        
                        else:
                            i = (layer_index - 1 - skip_conv) // block_layer_num 
                            # Formalize the 
                            #j = (block_layer_num - 2) - (layer_index % block_layer_num) 
                            j = (layer_index - 1 - skip_conv) % block_layer_num
                            name = 'conv%d_%d' %(i, j)
                            name_to_ind[name] = (i, j)
                            ind_to_name[(i, j)] = name
                            name_transfer_dict[module_list[layer_index]] = name
                            layer_index += 1

                        name_to_next_name[last_layer] = name
                        last_layer = name
                        meta_data[name] = { 'n': module.out_channels,
                                            'c': module.in_channels,
                                            'ksize': kernel_size,
                                            'padding': module.padding,
                                            'fsize': f_size,
                                            'stride':stride
                                            }
                        
                elif type(module) == eltwise.EltwiseAdd:
                    #i = layer_index // (block_layer_num + 1)
                    #i = (layer_index - 1 - skip_conv) // block_layer_num 
                    name = "conv"+str(i)+"_add"  
                    #print(name)
                    name_transfer_dict[module_list[layer_index]] = name
                    #print(module)
                    layer_index += 1

                elif type(module) == torch.nn.Linear: 
                    name = "linear" + str(linear_index)
                    name_transfer_dict[module_list[layer_index]] = name
                    #print(module)
                    #layer_index += 1
                    #linear_index += 1 

                #print(module)
            #print(name_transfer_dict)
            return name_to_ind, ind_to_name, meta_data, name_to_next_name, name_transfer_dict
        
        elif "inception":
            # The architecture doesn't have connection block.
            # TODO　Don't need dictionary..?
            return 
        
        elif "mobilenet_v2":
            # TODO 
            return 
        
        else:
            raise KeyError("This architecture don't need build module transfer dictionary")


    #def leader(self):
        # The "leader" is the first weights-tensor in the list
    #    return self.params_names[0]

    def is_supported(self, param_name):
        return param_name in self.pruning_ratios.keys()

    def fraction_to_prune(self, param_name):
        return self.pruning_ratios[param_name]

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if not self.is_supported(param_name):
            return
        fraction_to_prune = self.fraction_to_prune(param_name)
        try:
            model = meta['model']
        except TypeError:
            model = None
        return self._set_param_mask_by_sparsity_target(param, param_name, zeros_mask_dict, fraction_to_prune, model)

    def _set_param_mask_by_sparsity_target(self, param, param_name, zeros_mask_dict, target_sparsity, model):
        if not self.is_supported(param_name):
            return

        binary_map = None
        """
        ***
        It's hard for me to understand what is the need of this group dependency here, as a consequence, we drop this condition here.
        But if it's useful, we will remove the comment here.
        ***
        if self.group_dependency == "Leader":
            if target_sparsity != self.last_target_sparsity:
                # Each time we change the target sparsity we need to compute and cache the leader's binary-map.
                # We don't have control over the order that this function is invoked, so the only indication that
                # we need to compute a new leader binary-map is the change of the target_sparsity.
                self.last_target_sparsity = target_sparsity
                self.leader_binary_map = self.prune_group(target_sparsity, model.state_dict()[self.leader()],
                                                          self.leader(), zeros_mask_dict=None)
            assert self.leader_binary_map is not None
            binary_map = self.leader_binary_map
        """
        # Delegate the actual pruning to a sub-class
        self.prune_group(target_sparsity, param, param_name, zeros_mask_dict, model, binary_map)


    def prune_group(self, fraction_to_prune, param, param_name, zeros_mask_dict, model=None, binary_map=None):
        if fraction_to_prune == 0:
            return

        # Input argument :
        # fraction_to_prune, param, param_name=None, zeros_mask_dict=None, 
        # model=None, binary_map=None,
        binary_map = self.prune_kernel(fraction_to_prune, param, param_name,
                                       zeros_mask_dict, model, binary_map,
                                       group_size=self.group_size,
                                       rounding_fn=self.rounding_fn)
        return binary_map
