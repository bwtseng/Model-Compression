
from collections import OrderedDict
import yaml
import distiller
import os 
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Access root folder
sys.path.append(BASE_DIR)
from thinning import *
from pruning import *
from quantization import *
from utility import GradualWarmupScheduler
import scheduler as cs 
import policy as ply
import inspect 
import json
from torch.optim.lr_scheduler import *
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
import logging
import torch.nn.functional as F
### **************************************
### Configure YAML file and its generation
### **************************************
msglogger = logging.getLogger()
app_cfg_logger = logging.getLogger("app_cfg")

__all__ = ['filter_kwargs', 'build_component', 
           '__factory', '__policy_params', 'dict_config', 'NullLogger']

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
    valid_args['class_name'] = class_name
    final_valid_args, _ = filter_kwargs(valid_args, class_.__init__)
    instance = class_(**final_valid_args)
    return instance

### Return a dict-type object remaining the information of this configuration.
def __factory(container_type, model, sched_dict, **extra_args):
    container = {}
    if container_type in sched_dict:
        for name, user_args in sched_dict[container_type].items():
            try:
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
                if instance_name.startswith('admm'):
                    policy = ply.ADMMPolicy(pruner, args)
                else:
                    policy = ply.PruningPolicy(pruner, args)
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
        #print(lr_policies)
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

### *******************************************
### Configure YAML file and its generation [end] 
### *******************************************
if __name__ == "__main__":
    #stream = open("yaml_file/resnet20_filters.schedule_agp.yaml", "r")
    #sched_dict = yaml_ordered_load(stream)
    import model_zoo as mz
    model_dict= mz.data_function_dict
    model = model_dict['cifar10']['vgg']('vgg13_bn', pretrained=False)
    model = torch.nn.DataParallel(model).cuda()
    #print(model.features)
    #model.named_parameters()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    input = torch.rand([1, 3, 32, 32])
    output = model(input)
    target = torch.ones([1], dtype=torch.int64).to('cuda:0')
    loss = criterion(output, target)    
    # Setting Learning decay scheduler, that is, decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    # Set up compreesed config file.
    compress_scheduler = file_config(model, optimizer, "/home/bwtseng/Downloads/model_compression/yaml_file/admm_scheduler_cifar.yaml", None, None)
    # ***********************************************
    # Please refer to pytorch optimization source code.
    # ***********************************************
    #loss.backward()
    #group = optimizer.param_groups[0]
    #p = group['params'][0]
    #print(p)
    #print(optimizer.state[p])
    #print(compress_scheduler)
    """
    extensions = __factory('extensions', model, sched_dict)
    for policy_def in sched_dict['policies']:
        if 'extension' in policy_def:
            instance_name, args = __policy_params(policy_def, 'extension')
            assert instance_name in extensions, "Extension {} was not defined in the list of extensions".format(instance_name)
            extension = extensions[instance_name]
            policy = extension
    """
    print(compress_scheduler.policies[0])
    #epoch = 5
    #agg_loss = compress_scheduler.policies[5][0].before_backward_pass(model, 1, 0, 100,  loss, compress_scheduler.zeros_mask_dict)
    
    for epoch in range(10):
        for minibatch_id in range(10):
            output = model(input)
            target = torch.ones([1], dtype=torch.int64).to('cuda:0')
            loss = criterion(output, target)    
            policy = compress_scheduler.policies.get(epoch, list())[0]
            meta = compress_scheduler.sched_metadata[policy]
            meta['current_epoch'] = epoch
            compress_scheduler.on_epoch_begin(epoch, optimizer, mask_gradients=True)
            compress_scheduler.on_minibatch_begin(epoch, minibatch_id, 10, optimizer)
            compress_scheduler.on_minibatch_end(epoch, minibatch_id, 10, optimizer)
            agg_loss = compress_scheduler.before_backward_pass(epoch, minibatch_id, 10,  loss, 
                                                               optimizer, return_loss_components=True)
            optimizer.zero_grad()
            if agg_loss is not None:
                agg_loss.overall_loss.backward()
            else: 
                loss.backward()
            compress_scheduler.before_parameter_optimization(epoch, minibatch_id, 10, optimizer)
            compress_scheduler.on_epoch_end(epoch, optimizer=optimizer)
    #compress_scheduler.on_minibatch_end(1, 0, 100)
    print(agg_loss)
    #print(compress_scheduler.policies)
    #print(compress_scheduler.policies[5][0].mask_gradients)
    #print(compress_scheduler.policies[5][0].mask_on_forward_only)
    #if epoch in compress_scheduler.policies:
    #    print(True)
    #compress_scheduler.before_parameter_optimization(30, 1, 3, optimizer)
    #print(compress_scheduler)
    #compress_scheduler.on_minibatch_begin(15, 1, 3, optimizer)
    #compress_scheduler.on_minibatch_end(15, 1, 3, optimizer)
    #print(compress_scheduler.policies) TODO: can works without any bug here, try filter pruning now!!!!