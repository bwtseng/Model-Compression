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

from .pruner import _ParameterPruner
import distiller
import torch


class SensitivityPruner(_ParameterPruner):
    """Use algorithm from "Learning both Weights and Connections for Efficient
    Neural Networks" - https://arxiv.org/pdf/1506.02626v3.pdf

    I.e.: "The pruning threshold is chosen as a quality parameter multiplied
    by the standard deviation of a layers weights."
    In this code, the "quality parameter" is referred to as "sensitivity" and
    is based on the values learned from performing sensitivity analysis.

    Note that this implementation deviates slightly from the algorithm Song Han
    describes in his PhD dissertation, in that the threshold value is set only
    once.  In his PhD dissertation, Song Han describes a growing threshold, at
    each iteration.  This requires n+1 hyper-parameters (n being the number of
    pruning iterations we use): the threshold and the threshold increase (delta)
    at each pruning iteration.
    The implementation that follows, takes advantage of the fact that as pruning
    progresses, more weights are pulled toward zero, and therefore the threshold
    "traps" more weights.  Thus, we can use less hyper-parameters and achieve the
    same results.
    """

    def __init__(self, name, sensitivities, **kwargs):
        super(SensitivityPruner, self).__init__(name)
        self.sensitivities = sensitivities

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.sensitivities:
            if '*' not in self.sensitivities:
                return
            else:
                sensitivity = self.sensitivities['*']
        else:
            sensitivity = self.sensitivities[param_name]

        zeros_mask_dict[param_name].mask = self.create_mask(param, sensitivity)

    @staticmethod
    def create_mask(param, sensitivity):
        if not hasattr(param, 'stddev'):
            param.stddev = torch.std(param).item()
        with torch.no_grad():
            threshold = param.stddev * sensitivity
            mask = distiller.threshold_mask(param.data, threshold)
            return mask

'''
def threshold_mask(param, threshold):
    """Create a threshold mask for the provided parameter tensor using
    magnitude thresholding.

    Arguments:
        param: a parameter tensor which should be pruned.
        threshold: the pruning threshold.
    Returns:
        prune_mask: The pruning mask.
    """
    return torch.gt(torch.abs(param), threshold).type(param.type())
'''