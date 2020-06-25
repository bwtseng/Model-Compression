import pickle
import copy
import logging
import utility
import math
import statistics
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
from decimal import Decimal
from collections import OrderedDict
from functools import reduce
from typing import List, Tuple, Union, Dict, Any, Optional
import distiller
from scheduler import CompressionScheduler
import torch
import logging
import pickle
from pruning import * 
import utility
from summary_graph import SummaryGraph
import torch.optim as optim
from scheduler import ParameterMasker, create_model_masks_dict
msglogger = logging.getLogger()

#Original Model Cost: (Cost: memory=3881040, mac=21906240)
# Perform all layers selection. Why? the reason behind this is that as we get higher compression rate (whole model is pruned), the users may be more flexible to 
# decide other layers not to be pruned without any concerns of dropping utility accuracy.

class Conv2dTypeSpecificParams:
    """
    Holds layer parameters specific to Conv2D layers
    """

    def __init__(self, stride: Tuple[int, int], padding: Tuple[int, int], groups: int):
        """
        :param stride: Stride
        :param padding: Padding
        :param groups: Groups
        """
        self.stride = stride
        self.padding = padding
        self.groups = groups

class Layer_base:
    """
    Holds attributes for a given layer. This is a training-framework-agnostic abstraction for a layer
    """

    def __init__(self, module, name, weight_shape, output_shape):
        """
        Constructor
        :param module: Reference to the layer
        :param name: Unique name of the layer
        :param weight_shape: Shape of the weight tensor
        :param output_shape: Shape of the output activations
        """
        self.module = module
        self.name = str(name)
        self.weight_shape = weight_shape
        self.output_shape = output_shape

        self.picked_for_compression = False
        self.type_specific_params = None

        self._set_type_specific_params(module)

    #@abc.abstractmethod
    def _set_type_specific_params(self, module):
        """
        Using the provided module set type-specific-params
        :param module: Training-extension specific module
        :return:
        """
'''
class Layer(Layer_base):
    """ Holds attributes for a given layer """

    def _set_type_specific_params(self, module):

        if isinstance(module, torch.nn.Conv2d):
            params = Conv2dTypeSpecificParams(module.stride, module.padding, module.groups)
            self.type_specific_params = params

    def __init__(self, module: torch.nn.Module, name, output_shape):
        """
        Constructor
        :param module: Reference to the layer
        :param name: Unique name of the layer
        :param output_shape: Shape of the output activations
        
        # We add this to simplify our code.
        :param module: Reference to the layer
        :param name: Unique name of the layer
        :param weight_shape: Shape of the weight tensor
        :param output_shape: Shape of the output activations
        """
        if isinstance(module, torch.nn.Conv2d):
            if module.groups > 1:
                assert module.groups == module.in_channels
                assert module.in_channels == module.out_channels

                weight_shape = (module.out_channels, 1, module.kernel_size[0], module.kernel_size[1])
            else:
                weight_shape = (module.out_channels, module.in_channels, module.kernel_size[0], module.kernel_size[1])

        elif isinstance(module, torch.nn.Linear):
            weight_shape = (module.out_features, module.in_features, 1, 1)
        else:
            raise AssertionError("Layer currently supports only Conv2D and Linear")

        self.module = module
        self.name = str(name)
        self.weight_shape = weight_shape
        self.output_shape = output_shape

        self.picked_for_compression = False
        self.type_specific_params = None

        self._set_type_specific_params(module)
        
        self.var_name_of_module_in_parent = None
        self.parent_module = None
'''

class Layer(Layer_base):
    """ Holds attributes for a given layer """

    def _set_type_specific_params(self, module):

        if isinstance(module, torch.nn.Conv2d):
            params = Conv2dTypeSpecificParams(module.stride, module.padding, module.groups)
            self.type_specific_params = params

    def __init__(self, module: torch.nn.Module, name, output_shape):
        """
        Constructor
        :param module: Reference to the layer
        :param name: Unique name of the layer
        :param output_shape: Shape of the output activations
        """
        if isinstance(module, torch.nn.Conv2d):
            if module.groups > 1:
                assert module.groups == module.in_channels
                assert module.in_channels == module.out_channels

                weight_shape = (module.out_channels, 1, module.kernel_size[0], module.kernel_size[1])
            else:
                weight_shape = (module.out_channels, module.in_channels, module.kernel_size[0], module.kernel_size[1])

        elif isinstance(module, torch.nn.Linear):
            weight_shape = (module.out_features, module.in_features, 1, 1)
        else:
            raise AssertionError("Layer currently supports only Conv2D and Linear")

        Layer_base.__init__(self, module, name, weight_shape, output_shape)

        self.var_name_of_module_in_parent = None
        self.parent_module = None

class LayerDatabase:
    """
    Stores, creates and updates the Layer database
    Also stores compressible layers to model optimization
    """
    def __init__(self, model):
        self._model = model
        self._compressible_layers = OrderedDict()

    @property
    def model(self):
        """ Property to expose the underlying model """
        return self._model

    def __iter__(self):
        """
        Expose the underlying compressible_layers dictionary as an iterable
        :return:
        """
        return iter(self._compressible_layers.values())

    def find_layer_by_name(self, layer_name: str) -> Layer:
        """
        Find a layer in the database given the name of the layer
        :param layer_name: Name of the layer
        :return: Layer reference
        :raises KeyError if layer_name does not correspond to any layer in the database
        """

        for layer in self._compressible_layers.values():
            if layer.name == layer_name:
                return layer

        raise KeyError("Layer name %s does not exist in layer database" % layer_name)

    def find_layer_by_module(self, module) -> Layer:
        """
        Find a layer in the database given the name of the layer
        :param module: Module to find
        :return: Layer reference
        :raises KeyError if layer_name does not correspond to any layer in the database
        """
        return self._compressible_layers[id(module)]

    def mark_picked_layers(self, selected_layers):
        """
        Marks layers which are selected in the database
        :param selected_layers: layers which are selected for compression
        """
        for layer in self._compressible_layers.values():
            if layer in selected_layers:
                layer.picked_for_compression = True

    def get_selected_layers(self):
        """
        :return: Returns selected layers
        """
        selected_layers = [layer for layer in self._compressible_layers.values()
                           if layer.picked_for_compression is True]
        return selected_layers

    #@abc.abstractmethod
    def destroy(self):
        """
        Destroys the layer database
        """

# ***********************
# following code is the open source code from Qualcomm's repository, and used in the class Layerbase.
# ***********************
def create_rand_tensors_given_shapes(input_shape: Union[Tuple, List[Tuple]]) -> List[torch.Tensor]:
    """
    Given shapes of some tensors, create one or more random tensors and return them as a list of tensors
    :param input_shape: Shapes of tensors to create
    :return: Created list of tensors
    """
    if isinstance(input_shape, List):
        input_shapes = input_shape
    else:
        input_shapes = [input_shape]

    rand_tensors = []
    for shape in input_shapes:
        rand_tensors.append(torch.rand(shape))

    return rand_tensors


def is_leaf_module(module):

    """Utility function to determine if the given module is a leaf module - that is, does not have children modules
    :return:
        True if the module is a leaf, False otherwise
    """
    module_list = list(module.modules())

    return bool(len(module_list) == 1)

def get_device(model):
    """
    Function to find which device is model on
    Assumption : model is on single device
    :param model:
    :return: Device on which model is present
    """
    return next(model.parameters()).device

def run_hook_for_layers(model: torch.nn.Module, input_shapes: Union[Tuple, List[Tuple]], hook,
                        module_type_for_attaching_hook=None):
    """
    Register the given hook function for all layers in the model
    :param model: Model
    :param input_shapes: Shape of inputs to pass to the model
    :param hook: Hook function to register
    :param module_type_for_attaching_hook: Tuple of torch.nn module types for which hook has to be attached
    :return: None
    """

    # ------------------------
    # Register hook function
    # ------------------------
    hooks = []
    # All leaf modules
    modules = [module for module in model.modules() if is_leaf_module(module)]
    if module_type_for_attaching_hook:
        # if needed, filter by module types specified by caller
        modules = [module for module in modules if isinstance(module, module_type_for_attaching_hook)]
    for module in modules:
        hooks.append(module.register_forward_hook(hook))

    # ------------------------------------------------
    # Run forward pass to execute the hook functions
    # ------------------------------------------------
    device = get_device(model)
    dummy_tensors = create_rand_tensors_given_shapes(input_shapes)
    dummy_tensors = [tensor.to(device) for tensor in dummy_tensors]
    with torch.no_grad():
        _ = model(*dummy_tensors)

    # --------------------------
    # Remove all hooks we added
    # --------------------------
    for h in hooks:
        h.remove()

class LayerDatabase_torch(LayerDatabase):
    """
    Stores, creates and updates the Layer database
    Also stores compressible layers to model optimization
    """

    def __init__(self, model, input_shape):
        LayerDatabase.__init__(self, model)
        self._create_database(model, input_shape)

    def __deepcopy__(self, memodict):

        # pylint: disable=protected-access

        # Allocate a new LayerDatabase
        layer_db = copy.copy(self)
        memodict[id(self)] = layer_db

        # Create a deep copy of the model
        layer_db._model = copy.deepcopy(self._model, memodict)

        # Re-create the compressible layers dict
        layer_db._compressible_layers = {}

        modules_in_copy = list(layer_db._model.modules())

        # For all modules in the current model
        for index, module in enumerate(self._model.modules()):

            # If this module is in the current layer database
            if id(module) in self._compressible_layers:
                existing_layer = self._compressible_layers[id(module)]
                new_layer = Layer(modules_in_copy[index], existing_layer.name,
                                  existing_layer.output_shape)
                new_layer.picked_for_compression = existing_layer.picked_for_compression
                layer_db._compressible_layers[id(modules_in_copy[index])] = new_layer

        # Now we need to set parent references
        layer_db.set_reference_to_parent_module(layer_db._model, layer_db._compressible_layers)
        return layer_db

    def replace_layer(self, old_layer: Layer, new_layer: Layer):
        """
        Replace given layer with a new layer in the LayerDatabase
        :param old_layer: Existing Layer
        :param new_layer: New Layer
        :return: None
        """

        del self._compressible_layers[id(old_layer.module)]

        # set parent ref
        new_layer.parent_module = old_layer.parent_module
        new_layer.var_name_of_module_in_parent = old_layer.var_name_of_module_in_parent

        self._compressible_layers[id(new_layer.module)] = new_layer

    def replace_layer_with_sequential_of_two_layers(self, layer_to_replace: Layer,
                                                    layer_a: Layer, layer_b: Layer):
        """
        Replaces a layer with a sequential of layer in the database

        :param layer_to_replace: Layer to be replaced
        :param seq: Sequential of modules in layer_a and layer_b
        :param layer_a: 1st new layer
        :param layer_b: 2nd new layer
        :return: Nothing
        """

        # Create a sequential of these modules
        seq = torch.nn.Sequential(layer_a.module, layer_b.module)

        # Replace the original layer_to_replace in the model with this sequential
        setattr(layer_to_replace.parent_module, layer_to_replace.var_name_of_module_in_parent, seq)

        # Set parent correctly
        layer_a.parent_module = seq
        layer_a.var_name_of_module_in_parent = '0'

        layer_b.parent_module = seq
        layer_b.var_name_of_module_in_parent = '1'

        # Add the new layer to the database
        self._compressible_layers[id(layer_a.module)] = layer_a
        self._compressible_layers[id(layer_b.module)] = layer_b

        # Remove the the layer being replaced from the database
        del self._compressible_layers[id(layer_to_replace.module)]

    def update_layer_with_module_in_sequential(self, layer_to_update: Layer, seq: torch.nn.Sequential):
        """
        Update layer attributes with sequential in database

        :param layer_to_update: Layer to be updated
        :param seq: Sequential of modules in DownSample and layer
        :return: Nothing
        """
        # Remove the layer being updated from the database
        del self._compressible_layers[id(layer_to_update.module)]

        # Find the first conv2d within the sequential
        index, new_module = next((index, module) for (index, module) in enumerate(seq)
                                 if isinstance(module, torch.nn.Conv2d))

        # Determine new output shape
        new_output_shape = [new_module.in_channels, new_module.out_channels,
                            layer_to_update.output_shape[2], layer_to_update.output_shape[3]]
        new_module_name = layer_to_update.name + '.' + str(index)

        # Create a new layer
        new_layer = Layer(new_module, new_module_name, new_output_shape)

        # Set parent correctly
        new_layer.parent_module = seq
        new_layer.var_name_of_module_in_parent = str(index)

        # Add the updated layer to the database
        self._compressible_layers[id(new_layer.module)] = new_layer

    def _custom_hook_to_collect_layer_attributes(self, module, _, output):
        """
        Custom hook function which will be applied to all the layers in the model and store following
        information :
        model name (which will be removed), model reference, input shape and output shape
        """
        output_activation_shape = list(output.size())
        # activation dimension for FC layer is (1,1)
        if isinstance(module, torch.nn.Linear):
            output_activation_shape.extend([1, 1])

        module_name = None
        for name, module_ref in self._model.named_modules():
            if module is module_ref:
                #print(name)
                module_name = name

        self._compressible_layers[id(module)] = Layer(module, module_name, output_activation_shape)

    @staticmethod
    def _is_leaf_module(module):
        """Utility function to determine if the given module is a leaf module - that is, does not have children modules
        :return:
            True if the module is a leaf, False otherwise
        """
        module_list = list(module.modules())

        return bool(len(module_list) == 1)

    @classmethod
    def set_reference_to_parent_module(cls, module, layers):
        """
        Recursive function to set the parent references for each layer in the database
        :param module: Reference to the parent module
        :param layers: Layers to set reference for
        :return:
        """

        for module_name, module_ref in module.named_children():
            # first check if the module is leaf module or not
            if cls._is_leaf_module(module_ref):
                # iterate over all the layer attributes and if the match is found
                # then set the parent class and module name for that module
                if id(module_ref) in layers:
                    layer = layers[id(module_ref)]
                    layer.parent_module = module
                    layer.var_name_of_module_in_parent = module_name

            # if module is not leaf, call recursively
            else:
                cls.set_reference_to_parent_module(module_ref, layers)

    def _create_database(self, model, input_shape):
        # register custom hook for the model with run_graph provided by user
        # if the user wants to experiment with custom hook, we can support that option by
        # exposing the hook parameter to compress_net method
        run_hook_for_layers(model, input_shape, hook=self._custom_hook_to_collect_layer_attributes,
                                  module_type_for_attaching_hook=(torch.nn.Conv2d, torch.nn.Linear))

        # set the parent_class reference
        self.set_reference_to_parent_module(self._model, self._compressible_layers)

    def get_compressible_layers(self):
        """
        :return: Returns compressible layers
        """
        return self._compressible_layers

    def destroy(self):
        """
        Destroys the layer database
        """
        # clear the dictionary
        self._compressible_layers.clear()
        self._model = None

# **************************
# End here!
# **************************

class Rounder:
    """
    Rounds input channels to be kept and finds the corresponding updated compression ratio for Channel Pruning
    """
    def __init__(self, multiplicity: int):
        """
        :param multiplicity: Multiplicity to which rank is rounded-up
        """
        self._multiplicity = multiplicity

    def round(self, layer: Layer, comp_ratio: Decimal) -> Decimal:

        if self._multiplicity == 1:
            updated_comp_ratio = comp_ratio
        else:
            # get number of input channels to keep
            in_channels = layer.weight_shape[1]
            keep_inp_channels = in_channels * comp_ratio
            # Round input channels
            keep_inp_channels = round_up_to_multiplicity(self._multiplicity, keep_inp_channels, in_channels)

            # Find updated compression ratio
            updated_comp_ratio = Decimal(keep_inp_channels) / Decimal(in_channels)
            print("Comp_ratio: {}    Update_compr_ratio: {}".format(comp_ratio, updated_comp_ratio))
            assert comp_ratio <= updated_comp_ratio
            assert 0 <= updated_comp_ratio <= 1

        return updated_comp_ratio

    @staticmethod
    def round_up_to_multiplicity(multiplicity: int, num: int, max_allowable_num: int):
        """
        Function to round a number to the nearest multiplicity given the multiplicity
        :param multiplicity: multiplicity for rounding
        :param num: input number to be rounded
        :param max_allowable_num: maximum value for num allowed
        :return: number rounded up to nearest multiplicity
        """
        larger_multiple = math.ceil(float(num) / float(multiplicity)) * multiplicity
        if larger_multiple >= max_allowable_num:
            return max_allowable_num
        return int(larger_multiple)

class Cost:
    """
    Models cost of a layer or a collection of layers
    """

    def __init__(self, mem_cost: int, mac_cost: int):
        self.memory = mem_cost
        self.mac = mac_cost

    def __str__(self):
        return '(Cost: memory={}, mac={})'.format(self.memory, self.mac)

    def __add__(self, another_cost):
        return Cost(self.memory + another_cost.memory,
                    self.mac + another_cost.mac)

    def __sub__(self, another_cost):
        return Cost(self.memory - another_cost.memory,
                    self.mac - another_cost.mac)

class LayerCompRatioPair:
    """
    Models a pair of (layer: nn.Module, CompRatio: Decimal)
    """

    def __init__(self, layer: Layer, comp_ratio: Union[Decimal, None]):
        """
        Constructor
        :param layer: Reference to layer
        :param comp_ratio: Comp-ratio as a floating point number between 0 and 1
        """
        self.layer = layer
        self.comp_ratio = comp_ratio

    def __str__(self):
        return 'LayerCompRatioPair: layer={}, comp-ratio={}'.format(self.layer.name, self.comp_ratio)

def compute_layer_cost(layer: Layer):
    """
    Computes per layer cost
    :param layer: Attributes for a layer
    :return: Cost of the layer
    """
    weight_dim = list(layer.weight_shape)

    additional_act_dim = [layer.output_shape[2], layer.output_shape[3]]
    mem_cost = reduce(lambda x, y: x*y, weight_dim)
    mac_dim = weight_dim + additional_act_dim
    mac_cost = reduce(lambda x, y: x*y, mac_dim)
    return Cost(mem_cost, mac_cost)

def compute_model_cost(layer_db: LayerDatabase_torch):
    """
    Function to get the total cost of the model in terms of Memory and MAC metric
    :return: total cost (Memory), total cost (MAC)
    """

    network_cost_memory = 0
    network_cost_mac = 0
    for layer in layer_db:
        cost = compute_layer_cost(layer)
        network_cost_memory += cost.memory
        network_cost_mac += cost.mac
    return Cost(network_cost_memory, network_cost_mac)


class CompRatioSelectAlgo:

    def __init__(self, model, eval_dict_path:str, metric:str, group: str, target_comp_ratio:int,  
                 input_shape:Tuple, modules_to_ignore: List, multiplicity=1):

        self.model = model 
        self.target_comp_ratio = target_comp_ratio
        self.layer_db = LayerDatabase_torch(model, input_shape=input_shape)
        for layer in self.layer_db:
            print(layer.name)
        print(compute_model_cost(self.layer_db))
        self.input_shape = input_shape
        self.metric = metric
        self.group = group
        self.select(modules_to_ignore)
        self.rounding_algo = Rounder(multiplicity)
        self.eval_dict = self._unpickle_eval_scores_dict(eval_dict_path)
        self.updated_eval_scores_dict, self.commute_dict = self._update_eval_dict_with_rounding(self.eval_dict, self.rounding_algo)

    @staticmethod
    def _unpickle_eval_scores_dict(saved_eval_scores_dict_path: str):

        with open(saved_eval_scores_dict_path, 'rb') as f:
            eval_dict = pickle.load(f)
        msglogger.info("Greedy selection: Read eval dict from %s", saved_eval_scores_dict_path)
        return eval_dict

    def _update_eval_dict_with_rounding(self, eval_scores_dict, rounding_algo):
        updated_eval_dict = {}
        commute_dict = {}
        for layer_name in eval_scores_dict:
            # If use AttackPGD to wrap your model, the name of weight will change.
            # In this case, your model are in the moulde mode or wrapped by our PGD attacker.
            # Thus, the model name smapled from our sensitivity analysis should be moer careful.
            # To control it, we first apply the worst method to make these code executable.
            layer_name_qualcomm = layer_name[:-7]
            # ***
            # To apply the ratio list into our pruner, we need some transformation.
            # ***
            commute_dict[layer_name_qualcomm] = layer_name
            layer_eval_dict = eval_scores_dict[layer_name]
            eval_dict_per_layer = {}
            layer = self.layer_db.find_layer_by_name(layer_name_qualcomm)
            comp_ratio_list = sorted(list(layer_eval_dict.keys()), key=float)
            for comp_ratio in layer_eval_dict:
                rounded_comp_ratio = rounding_algo.round(layer, comp_ratio)

                eval_score = self._calculate_function_value_by_interpolation(rounded_comp_ratio, layer_eval_dict,
                                                                                comp_ratio_list)
                eval_dict_per_layer[Decimal(rounded_comp_ratio)] = eval_score
            updated_eval_dict[layer_name_qualcomm] = eval_dict_per_layer
        return updated_eval_dict, commute_dict
        
    def select(self, modules_to_ignore=[]):
        selected_layers = []
        
        for layer in self.layer_db:
            if layer.module in modules_to_ignore:
                continue

            if isinstance(layer.module, torch.nn.Linear):
                selected_layers.append(layer)

            elif isinstance(layer.module, torch.nn.Conv2d) and (layer.module.groups == 1):
                selected_layers.append(layer)
        self.layer_db.mark_picked_layers(selected_layers)

    @staticmethod
    def _calculate_function_value_by_interpolation(comp_ratio: Decimal, layer_eval_score_dict: dict,
                                                    comp_ratio_list: List):
        """
        Calculates eval score for a comp ratio by interpolation
        :param comp_ratio:
        :param layer_eval_score_dict:
        :param comp_ratio_list:
        :return:
        """
        if comp_ratio in comp_ratio_list:
            eval_score = layer_eval_score_dict[comp_ratio]
        else:
            ind = 0
            for ind, _ in enumerate(comp_ratio_list, start=0):
                if comp_ratio < comp_ratio_list[ind]:
                    break
            if ind == len(comp_ratio_list) - 1:
                eval_score = layer_eval_score_dict[comp_ratio_list[-1]]
                
            else:
                x1 = comp_ratio_list[ind]
                y1 = layer_eval_score_dict[comp_ratio_list[ind]]
                x2 = comp_ratio_list[ind - 1]
                y2 = layer_eval_score_dict[comp_ratio_list[ind - 1]]
                eval_score = (float(comp_ratio) - float(x1)) * (y1 - y2) / (float(x1) - float(x2)) + y1
        return eval_score
        
    @staticmethod
    def _evaluate_exit_condition(min_score, max_score, exit_threshold, current_comp_ratio, target_comp_ratio):

        if math.isclose(min_score, max_score, abs_tol=exit_threshold):
            return True, min_score

        if math.isclose(current_comp_ratio, target_comp_ratio, abs_tol=0.001):
            return True, statistics.mean([max_score, min_score])

        return False, None

    @staticmethod
    def _find_min_max_eval_scores(eval_scores_dict: Dict[str, Dict[Decimal, float]]):
        first_layer_scores = list(eval_scores_dict.values())[0]
        first_score = list(first_layer_scores.values())[0]
        min_score = first_score
        max_score = first_score
        for layer_scores in eval_scores_dict.values():
            for eval_score in layer_scores.values():
                if eval_score < min_score:
                    min_score = eval_score

                if eval_score > max_score:
                    max_score = eval_score
        return min_score, max_score

    @staticmethod
    def _find_layer_comp_ratio_given_eval_score(eval_scores_dict: Dict[str, Dict[Decimal, float]],
                                                given_eval_score, layer: Layer):

        # Find the closest comp ratio candidate for the current eval score
        eval_scores_for_layer = eval_scores_dict[layer.name]

        # Sort the eval scores by increasing order of compression
        comp_ratios = list(eval_scores_for_layer.keys())
        sorted_comp_ratios = sorted(comp_ratios, reverse=True)
        #sorted_comp_ratios = sorted(comp_ratios)

        # Special cases
        # Case1: Eval score is higher than even our most conservative comp ratio: then no compression
        if given_eval_score > eval_scores_for_layer[sorted_comp_ratios[0]]:
            return None

        if given_eval_score < eval_scores_for_layer[sorted_comp_ratios[-1]]:
            return sorted_comp_ratios[-1]

        # Start with a default of no compression
        selected_comp_ratio = None
        for index, comp_ratio in enumerate(sorted_comp_ratios[1:]):
            if given_eval_score > eval_scores_for_layer[comp_ratio]:
                selected_comp_ratio = sorted_comp_ratios[index]
                break

        print("{} selection ratio: {}, given eval score {}".format(layer.name, selected_comp_ratio, given_eval_score))

        # Remember to convert to sparsity lever(i.e 1-selected_comp_ratio [Sparsity level])
        # Actually, our pruner need sparsity criterion rather than compression ratio.
        
        if selected_comp_ratio:
            selected_sparsity_ratio = 1 - selected_comp_ratio
        else: 
            selected_sparsity_ratio = selected_comp_ratio

        return selected_sparsity_ratio

    def _find_all_comp_ratios_given_eval_score(self, given_eval_score, eval_scores_dict):
        layer_ratio_list = []
        for layer in self.layer_db.get_selected_layers():
            sparsity_ratio = self._find_layer_comp_ratio_given_eval_score(eval_scores_dict,
                                                                     given_eval_score, 
                                                                     layer)
            layer_ratio_list.append(LayerCompRatioPair(layer, sparsity_ratio))
        return layer_ratio_list

    # ******
    # In the while loop, caculate compressed model cost whilst takeing it as new compression ratio. 
    # ******
    def _calculate_model_comp_ratio_for_given_eval_score(self, model_cpy, eval_score:int, commute_dict,
                                                        eval_scores_dict, original_model_cost):
        # Calculate the compression ratios for each layer based on this score
        # *************
        # Our strategy is to select all layers to be compressed, you can also specify
        # the layer not to be pruned.
        # *************
        layer_ratio_list = self._find_all_comp_ratios_given_eval_score(eval_score, eval_scores_dict)
        for layer in self.layer_db:
            # If you wish this weight not to be pruned, set sparsity level to zero.
            if layer not in self.layer_db.get_selected_layers():
                layer_ratio_list.append(LayerCompRatioPair(layer, 0))

        # Build a dictionary to map each ratio to the original name of this model.
        sparsity_level = {}
        for layer_ratio_pair in layer_ratio_list:
            name = layer_ratio_pair.layer.name
            sparsity_level[commute_dict[name]] = layer_ratio_pair.comp_ratio

        self.perform_model_pruning(model_cpy, sparsity_level, group=self.group)
        compressed_layer_db = LayerDatabase_torch(model_cpy, input_shape=self.input_shape)
        compressed_model_cost = compute_model_cost(compressed_layer_db)
        print("Compressed Cost: {}".format(compressed_model_cost))
        if self.metric == 'memory':
            current_comp_ratio = Decimal(compressed_model_cost.memory / original_model_cost.memory)
        elif self.metric == 'mac':
            current_comp_ratio = Decimal(compressed_model_cost.mac / original_model_cost.mac)
        else:
            raise ValueError("No such this evaluation metric to proceed the greedy algorithm.")
        return current_comp_ratio

    @staticmethod
    def perform_model_pruning(model, sparsity_level, group):
        all_params = [param_name for param_name, param in model.named_parameters()]
        pruner_list = []
        first_layer_name = all_params[0]
        last_layer_name = all_params[-2]
        if group == 'element':
            sparsity_level = float(sparsity_level)
            # Element-wise sparasity
            # This can be well deployed in every layer of the deep neural network.
            sparsity_level = {}
            for param_name in all_params:
                if model.state_dict()[param_name].dim() not in [2,4]:
                    continue
                sparsity_levels = {param_name: sparsity_level[param_name]}
            pruner = distiller.pruning.SparsityLevelParameterPruner(name="sensitivity", levels=sparsity_levels)
            pruner_list.append(pruner)

        elif group == 'filter':
            # One thing you should keep in mind, in this case, the output layer will be executed by channel-wise analysis rather than filter.
            for param_name in all_params:
                if model.state_dict()[param_name].dim() not in [2,4]:
                    continue
                #sparsity_levels = {param_name: sparsity_level[param_name]}
                desired_sparsity = sparsity_level[param_name]
                if not desired_sparsity:
                    continue 

                if param_name == last_layer_name:
                    pruner = L1RankedStructureParameterPruner("sensitivity",
                                                            group_type="Channels",
                                                            desired_sparsity=desired_sparsity,
                                                            weights=last_layer_name)
                else:
                    pruner = L1RankedStructureParameterPruner("sensitivity",
                                                            group_type="Filters",
                                                            desired_sparsity=desired_sparsity,
                                                            weights=param_name)
                pruner_list.append(pruner)

        elif group == 'channel':
            for param_name in all_params:
                if param_name == first_layer_name:
                    continue
                
                if model.state_dict()[param_name].dim() not in [2,4] or param_name == first_layer_name:
                    continue
                desired_sparsity = sparsity_level[param_name]
                
                if not desired_sparsity:
                    continue
                pruner = distiller.pruning.L1RankedStructureParameterPruner("sensitivity",
                                                                            group_type="Channels",
                                                                            desired_sparsity=desired_sparsity,
                                                                            weights=param_name)
                pruner_list.append(pruner)

        # Build scheduler to set zero mask dictionary.
        scheduler = CompressionScheduler(model)
        for pruner in pruner_list:
            policy = distiller.PruningPolicy(pruner, pruner_args=None)
            scheduler.add_policy(policy, epochs=[0])

        # Compute the pruning mask per the pruner and apply the mask on the weights
        scheduler.on_epoch_begin(0)
        scheduler.mask_all_weights(epoch=0)
        # Build fake optimizer, the reason we call if fake is that there are no training loops here.
        optimizer = optim.SGD(model.parameters(), lr=0.001, 
                            momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss().to('cpu')
        model.to(device)
        input_shape = (1, 3, 32, 32)
        #print(compress_scheduler.zeros_mask_dict['basic_model.fc2.weight'].mask)
        dummy_input = utility.get_dummy_input('cifar10',  # Dataset should be specified.
                                            utility.model_device(model), 
                                            input_shape=input_shape)    
        sgraph = SummaryGraph(model, dummy_input)
        if group == 'filter':
            # First remove filter 
            thinning_recipe = thinning.create_thinning_recipe_filters(sgraph, model, scheduler.zeros_mask_dict, prune_output_layer=None)
            thinning.apply_and_save_recipe(model, scheduler.zeros_mask_dict, thinning_recipe, optimizer)
            # Second remove channel from last layer.
            zeros_mask_dict = create_model_masks_dict(model)
            zeros_mask_dict[last_layer_name].mask = scheduler.zeros_mask_dict[last_layer_name].mask
            #print(zeros_mask_dict[last_layer_name].mask)    
            thinning_recipe = thinning.create_thinning_recipe_channels(sgraph, model, zeros_mask_dict)
            thinning.apply_and_save_recipe(model, scheduler.zeros_mask_dict, thinning_recipe, optimizer) 
        elif group == 'channel':
            thinning_recipe = thinning.create_thinning_recipe_channels(sgraph, model, zeros_mask_dict)
            thinning.apply_and_save_recipe(model, scheduler.zeros_mask_dict, thinning_recipe, optimizer) 
        else: 
            raise ValueError("Can not execute thinning recipe with the model pruned by element-wise mode.")
        return model 

    def select_per_layer_comp_ratio(self):
        updated_eval_scores_dict, commute_dict = self._update_eval_dict_with_rounding(self.eval_dict, self.rounding_algo)
        #print(updated_eval_scores_dict)
        current_min_score, current_max_score = self._find_min_max_eval_scores(self.eval_dict)
        exit_threshold = (current_max_score - current_min_score)*0.0001
        msglogger.info("Greedy selection: overall_min_score=%f, overall_max_score=%f",
                        current_min_score, current_max_score)
        original_model_cost = compute_model_cost(self.layer_db)
        print("Original model cost: {}".format(original_model_cost))
        msglogger.info("Greedy selection: Original model cost=%s", original_model_cost)

        while True:
            current_mid_score = statistics.mean([current_max_score, current_min_score])
            model_cpy = copy.deepcopy(self.model)
            current_comp_ratio = self._calculate_model_comp_ratio_for_given_eval_score(model_cpy, current_mid_score, 
                                                                                      commute_dict, updated_eval_scores_dict, original_model_cost)
            
            msglogger.debug("Greedy selection: current candidate - comp_ratio=%f, score=%f, search-window=[%f:%f]",
                         current_comp_ratio, current_mid_score, current_min_score, current_max_score)

            should_exit, selected_score = self._evaluate_exit_condition(current_min_score, current_max_score,
                                                                        exit_threshold,
                                                                        current_comp_ratio, self.target_comp_ratio)
            if should_exit:
                print("Final evaluation score: {}".format(selected_score))
                break

            if current_comp_ratio > self.target_comp_ratio:
                # Not enough compression: Binary search the lower half of the scores
                current_max_score = current_mid_score
            else:
                # Too much compression: Binary search the upper half of the scores
                current_min_score = current_mid_score
        layer_ratio_list = self._find_all_comp_ratios_given_eval_score(selected_score, updated_eval_scores_dict)
        
        # Convert to sprarsit level dictionary! 
        for layer in self.layer_db:
            if layer not in self.layer_db.get_selected_layers():
                layer_ratio_list.append(LayerCompRatioPair(layer, None))
        # Build a dictionary and map each ratio to the original name of this model.
        sparsity_level = {}
        for layer_ratio_pair in layer_ratio_list:
            name = layer_ratio_pair.layer.name
            sparsity_level[commute_dict[name]] = layer_ratio_pair.comp_ratio

        model_cpy = copy.deepcopy(self.model)
        selected_comp_ratio = self._calculate_model_comp_ratio_for_given_eval_score(model_cpy, selected_score, 
                                                                                    commute_dict, updated_eval_scores_dict, original_model_cost) 
        
        print("Final compression ratio: {}".format(selected_comp_ratio))
        msglogger.info("Greedy selection: final choice - comp_ratio=%f, score=%f",
                    selected_comp_ratio, selected_score)
        return sparsity_level

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

input_shape = (1, 3, 32, 32)
pickle_path = 'model_save/sensitivity_analysis_lenet_cifar10_log_2020.05.29-012347/greedy_selection_eval_scores_dict.pkl'
# Your model setting should be consistent with your sensitivity analysis.
# More specifically, we hope each name of the module should be the same as the name listed in the csv file.
device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_function_dict = mz.data_function_dict
model = model_function_dict['cifar10']['lenet']('lenet', False, device=device)  
model = AttackPGD(model)
model.to(device)
model = torch.nn.DataParallel(model).cuda()
comp_algo = CompRatioSelectAlgo(model, pickle_path, metric='mac', group='filter', target_comp_ratio=0.125,
                                input_shape=input_shape, modules_to_ignore=[])
sparsity_level = comp_algo.select_per_layer_comp_ratio()
print("Final sparsity level dictionary: {}".format(sparsity_level))
print(CompRatioSelectAlgo.perform_model_pruning(model, sparsity_level, group='filter'))
layer_db = LayerDatabase_torch(model, input_shape=input_shape)
print(compute_model_cost(layer_db))