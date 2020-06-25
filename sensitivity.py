
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Perform sensitivity tests on layers and whole networks.
Construct a schedule for experimenting with network and layer sensitivity
to pruning.
The idea is to set the pruning level (percentage) of specific layers (or the
entire network), and then to prune once, run an evaluation on the test dataset,
and exit.  This should teach us about the "sensitivity" of the network/layers
to pruning.
This concept is discussed in "Learning both Weights and Connections for
Efficient Neural Networks" - https://arxiv.org/pdf/1506.02626v3.pdf
"""

from copy import deepcopy
from collections import OrderedDict
#import logging
import csv
import distiller
from scheduler import CompressionScheduler
import torch
import logging
import pickle
from pruning import * 

msglogger = logging.getLogger()
eval_scores_dict = {}
def perform_sensitivity_analysis(model, net_params, sparsities, test_func, group):
    """Perform a sensitivity test for a model's weights parameters.
    The model should be trained to maximum accuracy, because we aim to understand
    the behavior of the model's performance in relation to pruning of a specific
    weights tensor.
    By default this function will test all of the model's parameters.
    The return value is a complex sensitivities dictionary: the dictionary's
    key is the name (string) of the weights tensor.  The value is another dictionary,
    where the tested sparsity-level is the key, and a (top1, top5, loss) tuple
    is the value.
    Below is an example of such a dictionary:
    .. code-block:: python
    {'features.module.6.weight':    {0.0:  (56.518, 79.07,  1.9159),
                                     0.05: (56.492, 79.1,   1.9161),
                                     0.10: (56.212, 78.854, 1.9315),
                                     0.15: (35.424, 60.3,   3.0866)},
     'classifier.module.1.weight':  {0.0:  (56.518, 79.07,  1.9159),
                                     0.05: (56.514, 79.07,  1.9159),
                                     0.10: (56.434, 79.074, 1.9138),
                                     0.15: (54.454, 77.854, 2.3127)} }
    The test_func is expected to execute the model on a test/validation dataset,
    and return the results for top1 and top5 accuracies, and the loss value.
    """
    if group not in ['element', 'filter', 'channel']:
        raise ValueError("group parameter contains an illegal value: {}".format(group))
    sensitivities = OrderedDict()

    # Terminated layer for channel pruning
    last_layer_name = net_params[-2][0]
    first_layer_name = net_params[0][0]
    for param_name, param_var in net_params:

        # Ignore bias analysis.
        if model.state_dict()[param_name].dim() not in [2,4]:
            continue
        # Make a copy of the model, because when we apply the zeros mask (i.e.
        # perform pruning), the model's weights are altered
        model_cpy = deepcopy(model)
        layer_wise_eval_scores_dict = {}
        sensitivity = OrderedDict()
        for sparsity_level in sparsities:
            sparsity_level = float(sparsity_level)
            msglogger.info("Testing sensitivity of %s [%0.1f%% sparsity]" % (param_name, sparsity_level*100))

            # **************
            # Should I specify the pruner or not? Qualcomm directly use the SVD pruner to get the optimial permutation of filter, and the sensitivity table could also be optimal? 
            # **************

            # Create the pruner (a level pruner)
            # pruning schedule.
            if group == 'element':
                # Element-wise sparasity
                # This can be well deployed in every layer of the deep neural network.
                sparsity_levels = {param_name: sparsity_level}
                pruner = distiller.pruning.SparsityLevelParameterPruner(name="sensitivity", levels=sparsity_levels)

            elif group == 'filter':
                # Output filter ranking
                # But so far it's only restricted to convolution layers.  

                # *********************
                # My idea: (How about adding new )
                # The last layer (i.e output layer) can not be examined and pruned in this case, distiller's authors thus droped the analysis process to avoid trivial code writing.
                # How to define the termination condition? and apply this mechanism to other fully connected layers. I think this is really important, since there is too many weights here.
                # *********************
                
                if param_name == last_layer_name:
                    # One thing you should keep in mind, in this case, the output layer will be executed by channel-wise analysis rather than filter.
                    pruner = L1RankedStructureParameterPruner("sensitivity",
                                                            group_type="Channels",
                                                            desired_sparsity=sparsity_level,
                                                            weights=param_name)
                else: 
                    pruner = L1RankedStructureParameterPruner("sensitivity",
                                                             group_type="Filters",
                                                             desired_sparsity=sparsity_level,
                                                             weights=param_name)

                """
                if model.state_dict()[param_name].dim() != 4:
                    continue
                pruner = distiller.pruning.L1RankedStructureParameterPruner("sensitivity",
                                                                            group_type="Filters",
                                                                            desired_sparsity=sparsity_level,
                                                                            weights=param_name)
                """
            elif group == 'channel':
                # Input channel ranking
                # But so far it's only restricted to convolution layers.
                # *********************
                # My idea:
                # Channel pruning is very applicable and more suitable to deploy (no change the size of output logit), but distiiler not support analyze, either.
                # The reason is that pruner should be specified the group type by YAML file, so they decide dropping this code to avoid trivial code gerneration.
                # *Should I add new sensitivity analysis policy or pruner*?
                # *********************
                """
                if model.state_dict()[param_name].dim() != 4:
                    continue
                """
                if param_name == first_layer_name:
                    continue
                pruner = distiller.pruning.L1RankedStructureParameterPruner("sensitivity",
                                                                            group_type="Channels",
                                                                            desired_sparsity=sparsity_level,
                                                                            weights=param_name)
                
            policy = distiller.PruningPolicy(pruner, pruner_args=None)
            scheduler = CompressionScheduler(model_cpy)
            scheduler.add_policy(policy, epochs=[0])

            # Compute the pruning mask per the pruner and apply the mask on the weights
            scheduler.on_epoch_begin(0)
            scheduler.mask_all_weights(epoch=0)

            # Test and record the performance of the pruned model
            record = test_func(model=model_cpy, parameter_name=param_name)
            layer_wise_eval_scores_dict[1-sparsity_level] = record[0]
            sensitivity[sparsity_level] = record+(param_var,)
            sensitivities[param_name] = sensitivity
            #Our testing function output is consisted of nat_top1, nat_top5, nat_loss, adv_top1, adv_top5, adv_loss.
        eval_scores_dict[param_name] = layer_wise_eval_scores_dict

    return sensitivities, eval_scores_dict


def sensitivities_to_png(sensitivities, fname):
    """Create a mulitplot of the sensitivities.
    The 'sensitivities' argument is expected to have the dict-of-dict structure
    described in the documentation of perform_sensitivity_test.
    """
    try:
        # sudo apt-get install python3-tk
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: Function plot_sensitivity requires package matplotlib which"
              "is not installed in your execution environment.\n"
              "Skipping the PNG file generation")
        return

    msglogger.info("Generating sensitivity graph")

    for param_name, sensitivity in sensitivities.items():
        sense = [values[1] for sparsity, values in sensitivity.items()]
        sparsities = [sparsity for sparsity, values in sensitivity.items()]
        plt.plot(sparsities, sense, label=param_name)

    plt.ylabel('top5')
    plt.xlabel('sparsity')
    plt.title('Pruning Sensitivity')
    plt.legend(loc='lower center',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(fname, format='png')


def sensitivities_to_csv(sensitivities, fname):
    """Create a CSV file listing from the sensitivities dictionary.
    The 'sensitivities' argument is expected to have the dict-of-dict structure
    described in the documentation of perform_sensitivity_test.
    """
    with open(fname, 'w') as csv_file:
        writer = csv.writer(csv_file)
        # write the header
        # nat_top1, nat_top5, nat_loss, adv_top1, adv_top5, adv_loss for the class number higher than ten
        if len(list(list(sensitivities.values())[0].values())[0]) == 5:
            writer.writerow(['parameter', 'sparsity', 'nat_top1', 'nat_loss', 'adv_top1', 'adv_loss', 'variance'])
        
        elif len(list(list(sensitivities.values())[0].values())[0]) == 7:
            writer.writerow(['parameter', 'sparsity', 'nat_top1', 'nat_top5', 'nat_loss', 
                             'adv_top1', 'adv_top5','adv_loss', 'variance'])

        elif len(list(list(sensitivities.values())[0].values())[0]) == 3:
            writer.writerow(['parameter', 'sparsity', 'top1', 'loss', 'variance'])

        elif len(list(list(sensitivities.values())[0].values())[0]) == 4:
            writer.writerow(['parameter', 'sparsity', 'top1', 'top5', 'loss', 'variance'])

        else:
            raise ValueError("Please revise your test function output.")

        for param_name, sensitivity in sensitivities.items():
            for sparsity, values in sensitivity.items():
                writer.writerow([param_name] + [sparsity] + list(values))


def _pickle_eval_scores_dict(eval_scores_dict, fname):

    with open(fname, 'wb') as file:
        pickle.dump(eval_scores_dict, file)

    #logger.info("Greedy selection: Saved eval dict to %s", self.PICKLE_FILE_EVAL_DICT)
