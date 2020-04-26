
from collections import OrderedDict
import yaml
import distiller
import policy as ply
from pruning import *
from thinning import *
from quantization import *
import inspect 
import json
from torch.optim.lr_scheduler import *
import scheduler as cs 
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
import logging
### **************************************
### Configure YAML file and its generation
### **************************************
msglogger = logging.getLogger()
app_cfg_logger = logging.getLogger("app_cfg")

def filter_kwargs(dict_to_filter, function_to_call):
    """Utility to check which arguments in the passed dictionary exist in a function's signature
    The function returns two dicts, one with just the valid args from the input and one with the invalid args.
    The caller can then decide to ignore the existence of invalid args, depending on context.
    """

    sig = inspect.signature(function_to_call)
    filter_keys = [param.name for param in sig.parameters.values() if (param.kind == param.POSITIONAL_OR_KEYWORD)]
    valid_args = {}
    invalid_args = {}

    for key in dict_to_filter:
        if key in filter_keys:
            valid_args[key] = dict_to_filter[key]
        else:
            invalid_args[key] = dict_to_filter[key]
    return valid_args, invalid_args

def build_component(model, name, user_args, **extra_args):
    # Instantiate component using the 'class' argument
    class_name = user_args.pop('class')
    #print(class_name)
    try:
        class_ = globals()[class_name]
    except KeyError as ex:
        raise ValueError("Class named '{0}' does not exist".format(class_name)) from ex

    # First we check that the user defined dict itself does not contain invalid args
    valid_args, invalid_args = filter_kwargs(user_args, class_.__init__)
    if invalid_args:
        raise ValueError(
            '{0} does not accept the following arguments: {1}'.format(class_name, list(invalid_args.keys())))

    # Now we add some "hard-coded" args, which some classes may accept and some may not
    # So then we filter again, this time ignoring any invalid args
    valid_args.update(extra_args)
    valid_args['model'] = model
    valid_args['name'] = name
    final_valid_args, _ = filter_kwargs(valid_args, class_.__init__)
    instance = class_(**final_valid_args)
    return instance

### Return a dict-type object remaining the information of this configuration.
def __factory(container_type, model, sched_dict, **extra_args):
    container = {}
    if container_type in sched_dict:
        for name, user_args in sched_dict[container_type].items():
            try:
                #print(name)
                instance = build_component(model, name, user_args, **extra_args)
                container[name] = instance
            except Exception as exception:
                print("\nFatal error while parsing [section: %s] [item: %s]" % (container_type, name))
                print("Exception: %s %s" % (type(exception), exception))
                raise

    return container

def __policy_params(policy_def, type):
    name = policy_def[type]['instance_name']
    args = policy_def[type].get('args', None)
    return name, args

def dict_config(model, optimizer, sched_dict, scheduler=None, resumed_epoch=None):
    app_cfg_logger.debug('Schedule contents:\n' + json.dumps(sched_dict, indent=2))
    #print('Schedule contents:\n' + json.dumps(sched_dict, indent=2))
    if scheduler is None:
        scheduler = cs.CompressionScheduler(model)

    pruners = __factory('pruners', model, sched_dict)
    regularizers = __factory('regularizers', model, sched_dict)
    quantizers = __factory('quantizers', model, sched_dict, optimizer=optimizer)
    if len(quantizers) > 1:
        raise ValueError("\nError: Multiple Quantizers not supported")
    extensions = __factory('extensions', model, sched_dict)

    try:
        lr_policies = []
        for policy_def in sched_dict['policies']:
            policy = None
            if 'pruner' in policy_def:
                try:
                    instance_name, args = __policy_params(policy_def, 'pruner')
                except TypeError as e:
                    print('\n\nFatal Error: a policy is defined with a null pruner')
                    print('Here\'s the policy definition for your reference:\n{}'.format(json.dumps(policy_def, indent=1)))
                    raise
                assert instance_name in pruners, "Pruner {} was not defined in the list of pruners".format(instance_name)
                pruner = pruners[instance_name]
                policy = ply.PruningPolicy(pruner, args)
                #print("I Love Purner!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            elif 'regularizer' in policy_def:
                instance_name, args = __policy_params(policy_def, 'regularizer')
                assert instance_name in regularizers, "Regularizer {} was not defined in the list of regularizers".format(instance_name)
                regularizer = regularizers[instance_name]
                if args is None:
                    policy = ply.RegularizationPolicy(regularizer)
                else:
                    policy = ply.RegularizationPolicy(regularizer, **args)

            elif 'quantizer' in policy_def:
                instance_name, args = __policy_params(policy_def, 'quantizer')
                assert instance_name in quantizers, "Quantizer {} was not defined in the list of quantizers".format(instance_name)
                quantizer = quantizers[instance_name]
                policy = ply.QuantizationPolicy(quantizer)

            elif 'lr_scheduler' in policy_def:
                # LR schedulers take an optimizer in their constructor, so postpone handling until we're certain
                # a quantization policy was initialized (if exists)
                lr_policies.append(policy_def)
                continue

            elif 'extension' in policy_def:
                instance_name, args = __policy_params(policy_def, 'extension')
                
                assert instance_name in extensions, "Extension {} was not defined in the list of extensions".format(instance_name)
                extension = extensions[instance_name]
                policy = extension

            else:
                raise ValueError("\nFATAL Parsing error while parsing the pruning schedule - unknown policy [%s]".format(policy_def))

            add_policy_to_scheduler(policy, policy_def, scheduler)

        # Any changes to the optimizer caused by a quantizer have occurred by now, so safe to create LR schedulers
        lr_schedulers = __factory('lr_schedulers', model, sched_dict, optimizer=optimizer,
                                  last_epoch=(resumed_epoch if resumed_epoch is not None else -1))
        for policy_def in lr_policies:
            instance_name, args = __policy_params(policy_def, 'lr_scheduler')
            assert instance_name in lr_schedulers, "LR-scheduler {} was not defined in the list of lr-schedulers".format(
                instance_name)
            lr_scheduler = lr_schedulers[instance_name]
            policy = ply.LRPolicy(lr_scheduler)
            add_policy_to_scheduler(policy, policy_def, scheduler)

    except AssertionError:
        # propagate the assertion information
        raise
    except Exception as exception:
        print("\nFATAL Parsing error!\n%s" % json.dumps(policy_def, indent=1))
        print("Exception: %s %s" % (type(exception), exception))
        raise
    return scheduler

def add_policy_to_scheduler(policy, policy_def, scheduler):
    if 'epochs' in policy_def:
        scheduler.add_policy(policy, epochs=policy_def['epochs'])
    else:
        scheduler.add_policy(policy, starting_epoch=policy_def['starting_epoch'],
                            ending_epoch=policy_def['ending_epoch'],
                            frequency=policy_def['frequency'])



def yaml_ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """Function to load YAML file using an OrderedDict
    See: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    """
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    return yaml.load(stream, OrderedLoader)



def file_config(model, optimizer, filename, scheduler=None, resumed_epoch=None):
    """Read the schedule from file"""
    with open(filename, 'r') as stream:
       msglogger.info('Reading compression schedule from: %s', filename)
       #print("Reading compression schedule from: {}".format(filename))
       try:
            sched_dict = yaml_ordered_load(stream)
            return dict_config(model, optimizer, sched_dict, scheduler, resumed_epoch)
       except yaml.YAMLError as exc:
            print("\nFATAL parsing error while parsing the schedule configuration file %s" % filename)
            raise
### *******************************************
### Configure YAML file and its generation [end] 
### *******************************************



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


### ****************************************
### Preprare for the quantization file 
### ****************************************
def config_component_from_file_by_class(model, filename, class_name, **extra_args):
    with open(filename, 'r') as stream:
        msglogger.info('Reading configuration from: %s', filename)
        print('Reading configuration from: {}.'.format(filename))
        try:
            config_dict = yaml_ordered_load(stream)
            config_dict.pop('policies', None)
            for section_name, components in config_dict.items():
                for component_name, user_args in components.items():
                    ### ONLY FIND THE POST LINEAR QUANTIZATION CLASS!!!!!
                    if user_args['class'] == class_name:
                        #print('Found component of class {}: Name: {} ; Section: {}'.format(class_name, component_name,
                        #                                                                    section_name))
                        msglogger.info( 'Found component of class {0}: Name: {1} ; Section: {2}'.format(class_name, component_name,
                                                                                            section_name))
                        user_args.update(extra_args)
                        return build_component(model, component_name, user_args)
            raise ValueError(
                'Component of class {0} does not exist in configuration file {1}'.format(class_name, filename))
        except yaml.YAMLError:
            print("\nFATAL parsing error while parsing the configuration file %s" % filename)
            raise

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


if __name__ == "__main__":
    #stream = open("yaml_file/resnet20_filters.schedule_agp.yaml", "r")
    #sched_dict = yaml_ordered_load(stream)
    import model_zoo as mz
    model_dict= mz.data_function_dict
    model = model_dict['cifar10']['vgg']('vgg13_bn', pretrained=False)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    # Setting Learning decay scheduler, that is, decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    # Set up compreesed config file.
    compress_scheduler = file_config(model, optimizer, "yaml_file/vgg13.schedule_agp_cifar10.yaml", None, None)
    """
    extensions = __factory('extensions', model, sched_dict)
    for policy_def in sched_dict['policies']:
        if 'extension' in policy_def:
            instance_name, args = __policy_params(policy_def, 'extension')
            assert instance_name in extensions, "Extension {} was not defined in the list of extensions".format(instance_name)
            extension = extensions[instance_name]
            policy = extension
    """
    epoch = 5
    agg_loss = compress_scheduler.policies[5][0].before_backward_pass(model, 5, 1, 100,  criterion, compress_scheduler.zeros_mask_dict)
    print(compress_scheduler.policies)
    print(compress_scheduler.policies[5][0].mask_gradients)
    print(compress_scheduler.policies[5][0].mask_on_forward_only)
    if epoch in compress_scheduler.policies:
        print(True)
    #compress_scheduler.before_parameter_optimization(30, 1, 3, optimizer)
    print(compress_scheduler)
    #compress_scheduler.on_minibatch_begin(15, 1, 3, optimizer)
    #compress_scheduler.on_minibatch_end(15, 1, 3, optimizer)
    #print(compress_scheduler.policies) TODO: can works without any bug here, try filter pruning now!!!!