import contextlib
import logging
import torch
from quantization.quantizer import FP_BKP_PREFIX
#from policy import PolicyLoss, LossComponent
#from utility import model_device, normalize_module_name
import policy as ply
import utility
import torch.nn as nn

__all__ = ["CompressionScheduler", "ParameterMasker", "create_model_masks_dict"]
msglogger = logging.getLogger()
class CompressionScheduler(object):
    """Responsible for scheduling pruning and masking parameters.
    """
    def __init__(self, model, zeros_mask_dict=None, device=torch.device("cuda")):
        self.model = model
        self.device = device
        self.pruner_epoch = []
        self.pruner_info = {}
        self.policies = {}
        self.sched_metadata = {}
        self.prune_mechanism = False
        self.retrain_phase = False
        self.admm_prune = False
        self.thinning = False
        # Create the masker objects and place them in a dictionary indexed by the parameter name
        self.zeros_mask_dict = zeros_mask_dict or create_model_masks_dict(model)

    # My idea is that use the sched_metadata to collect the information about the
    # starting epoch of retrain phase, and mix-up and ward-up criterion can be applied in the
    # main function 
    def collect_starting_epoch(self):
        #self.sched_metadata[policy] = {}
        pass

    def add_policy(self, policy, epochs=None, starting_epoch=0, ending_epoch=1, frequency=1):
        """Add a new policy to the schedule.
        Args:
            epochs (list): A list, or range, of epochs in which to apply the policy
        """

        if epochs is None:
            epochs = list(range(starting_epoch, ending_epoch, frequency))

        for epoch in epochs:
            if epoch not in self.policies:
                self.policies[epoch] = [policy]
            else:
                self.policies[epoch].append(policy)
            assert len(self.policies[epoch]) > 0
            
        self.sched_metadata[policy] = {'starting_epoch': starting_epoch,
                                       'ending_epoch': ending_epoch,
                                       'frequency': frequency}

        class_name = policy.__class__.__name__.split("Policy")[0]
        
        if "Remover" in class_name:
            self.thinning = True
            self.thinning_epoch = epochs

        # In the following code, we save the maximum and minimum epochs withing all pruners.
        # This is designed for distingushing the "pretrain", "ADMM pruning" and "retrain" phase. 
        # Toward this end, we are able to tune the initial learning rate in an automative way.
        if class_name in ['ADMM', "Pruning"]:
            self.prune_mechanism = True
            if 'max_epoch' in self.pruner_info:
                if ending_epoch > self.pruner_info['max_epoch']:
                    self.pruner_info['max_epoch'] = ending_epoch
            else:
                self.pruner_info['max_epoch'] = ending_epoch
            
            if class_name == 'ADMM':
                self.admm_prune = True
                # Can not deal with seperate ADMM pruner.
                self.pruner_info["ADMM_epoch"] = ending_epoch

            if 'min_epoch' in self.pruner_info:
                if starting_epoch < self.pruner_info['min_epoch']:
                    self.pruner_info['min_epoch'] = starting_epoch
            else:
                self.pruner_info['min_epoch'] = starting_epoch
        
    def on_epoch_begin(self, epoch, optimizer=None, **kwargs):

        for policy in self.policies.get(epoch, list()):
            meta = self.sched_metadata[policy]
            meta['current_epoch'] = epoch
            # *****************************************
            # Reset learning rate in pretraining/retraining phase!
            # *****************************************
            """
            if epoch == (self.pruner_info['max_epoch']):
                for group in optimizer.param_groups:
                    group['lr'] = kwargs['initial_learning_rate']

            if epoch == (self.pruner_info['min_epoch']):
                for group in optimizer.param_groups:
                    group['lr'] = kwargs['initial_learning_rate']
            """

            # I believe that gradient masking should be the must do thing.
            if self.prune_mechanism: 
                if epoch >= self.pruner_info['max_epoch'] and not self.thinning:
                    kwargs['mask_gradients'] = True
                    policy.on_epoch_begin(self.model, self.zeros_mask_dict, meta, **kwargs)
                else:
                    kwargs['mask_gradients'] = False
                    policy.on_epoch_begin(self.model, self.zeros_mask_dict, meta, **kwargs)
            else: 
                if self.retrain_phase: 
                    kwargs['mask_gradients'] = True
                    policy.on_epoch_begin(self.model, self.zeros_mask_dict, meta, **kwargs)
                else: 
                    kwargs['mask_gradients'] = False
                    policy.on_epoch_begin(self.model, self.zeros_mask_dict, meta, **kwargs)
            """
            if 'max_epoch' in self.pruner_info and epoch >= self.pruner_info['max_epoch']:    
                policy.on_epoch_begin(self.model, self.zeros_mask_dict, meta, mask_gradients=True)

            else:
                kwargs['mask_gradients'] = False
                policy.on_epoch_begin(self.model, self.zeros_mask_dict, meta,
                                  **kwargs)
            """

    def on_minibatch_begin(self, epoch, minibatch_id, minibatches_per_epoch, optimizer=None):
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                meta = self.sched_metadata[policy]
                meta['current_epoch'] = epoch
                policy.on_minibatch_begin(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                          self.zeros_mask_dict, meta, optimizer)

    def before_backward_pass(self, epoch, minibatch_id, minibatches_per_epoch, loss, optimizer=None,
                             return_loss_components=False):
        # We pass the loss to the policies, which may override it
        overall_loss = loss
        loss_components = []
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                policy_loss = policy.before_backward_pass(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                                          overall_loss, self.zeros_mask_dict)
                if policy_loss is not None:
                    curr_loss_components = self.verify_policy_loss(policy_loss)
                    overall_loss = policy_loss.overall_loss
                    loss_components += curr_loss_components
        if return_loss_components:
            return ply.PolicyLoss(overall_loss, loss_components)

        return overall_loss


    def before_parameter_optimization(self, epoch, minibatch_id, minibatches_per_epoch, optimizer):
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                #class_name = policy.__class__.__name__.
                meta = self.sched_metadata[policy]
                meta['current_epoch'] = epoch
                # *******************************
                # Should be modified.
                # *******************************
                if self.prune_mechanism:
                    if epoch > self.pruner_info['max_epoch'] and not not self.thinning:
                        policy.before_parameter_optimization(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                                             self.zeros_mask_dict, meta, optimizer, apply_gradient_mask=False)
                    else:
                        policy.before_parameter_optimization(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                                             self.zeros_mask_dict, meta, optimizer)
                else:
                    if self.retrain_phase:
                        policy.before_parameter_optimization(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                                             self.zeros_mask_dict, meta, optimizer, apply_gradient_mask=False)
                    else:
                        policy.before_parameter_optimization(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                                             self.zeros_mask_dict, meta, optimizer)
                
                #if epoch > self.pruner_info['max_epoch']:
                    # Retrain stage, apply Apply mask gradient mechanism.

    def on_minibatch_end(self, epoch, minibatch_id, minibatches_per_epoch, optimizer=None) :
        # When we get to this point, the weights are no longer masked.  This is because during the backward
        # pass, the weights may have been updated.  This is true even when the gradients are zero, for some
        # optimization algorithms such as SGD with momentum.  See the Note in PyTorch's SGD documentation:
        # https://pytorch.org/docs/stable/optim.html#torch.optim.SGD.
        #
        # Therefore we choose to always apply the pruning mask.  In the future we may optimize this by applying
        # the mask only if the some policy is actually using the mask.
        

        # ************************************
        # This may be dropped in the future, the reason is that ADMM optimize only when
        # user prints the target sparsity.  Re-train phase thus need it!!!!
        # ************************************

        self.mask_all_weights(epoch, is_forward=False)
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                policy.on_minibatch_end(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                        self.zeros_mask_dict, optimizer)

    def on_epoch_end(self, epoch, optimizer=None, **kwargs):
        for policy in self.policies.get(epoch, list()):
            meta = self.sched_metadata[policy]
            meta['current_epoch'] = epoch
            meta['optimizer'] = optimizer
            kwargs['optimizer'] = optimizer
            policy.on_epoch_end(self.model, self.zeros_mask_dict, meta,
                                **kwargs) #{optimzer:}

    def mask_all_weights(self, epoch, is_forward=True):
        for name, param in self.model.named_parameters():
            try:
                masker = self.zeros_mask_dict[name]
                if is_forward or not masker.mask_on_forward_only:
                    # When we mask on forward-pass only, we allow the gradients to change
                    # the weights.
                    if masker.mask_pruner is not None:
                        # When purners are implemented by our policy, we will give them
                        # specified name of this mask object. Furthermore, training 
                        # mask is no longer needed in some case, for instance: ADMM. 
                        # Thus, we mask these weights only when retraining this model. 
                        if epoch >= self.pruner_info['ADMM_epoch']:
                            #if epoch >= self.pruner_info['ADMM_epoch']: 
                            masker.mask_tensor(param)
                    else: 
                        masker.mask_tensor(param)
            except KeyError:
                # Quantizers for training might modify model parameters in a couple of ways:
                #   1. By adding a prefix to the parameter tensor name
                #   2. By wrapping the module holding the parameter in a wrapper module
                # If the source of the error is one of the above, workaround and move on
                #
                # Quantizers might also add new learnable parameters (e.g. the clip value in PACT quantization)
                # These parameters will also be missing from the masks mapping. For now, we'll assume that we're
                # not interested in pruning these parameters - and we just ignore them.
                #
                # TODO: This is not scalable at all. Find a solution that doesn't "hard-code" these conditions...
                
                name_parts = name.split('.')
                prefixed = name_parts[-1].startswith(FP_BKP_PREFIX)
                wrapped = name_parts[-2] == 'wrapped_module'
                if prefixed or wrapped:
                    if prefixed:
                        name_parts[-1] = name_parts[-1].replace(FP_BKP_PREFIX, '', 1)
                    if wrapped:
                        name_parts.pop(-2)
                    name = '.'.join(name_parts)
                    self.zeros_mask_dict[name].apply_mask(param)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        Currently it contains just the pruning mask.
        """
        masks = {}
        for name, masker in self.zeros_mask_dict.items():
            masks[name] = masker.mask
        state = {'masks_dict': masks}
        return state

    def load_state_dict(self, state, normalize_dataparallel_keys=False):
        """Loads the scheduler state.
        Currently the scheduler state is comprised only of the set of pruning masks.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`. It is a dictionary of parameter
                names (keys) and parameter masks (values).
            normalize_dataparallel_keys (bool): indicates if we should convert the keys from
                DataParallel format.  This should be set to True when loading a model
                from a GPU-checkpoint onto a CPU (because currently we don't use DataParallel
                on the CPU).
        """
        try:
            loaded_masks = state['masks_dict']
        except KeyError as exception:
            #print('could not load the CompressionScheduler state.'' masks_dict is missing from state')
            msglogger.error('Could not load the CompressionScheduler state.'' masks_dict is missing from state')
            with contextlib.suppress(TypeError):
                #print('Scheduler state keys are: {}'.format(', '.join(state)))
                msglogger.debug('Scheduler state keys are: {}'.format(', '.join(state)))
            raise

        if normalize_dataparallel_keys:
            loaded_masks = {normalize_module_name(k): v for k, v in loaded_masks.items()}
        device = model_device(self.model)
        for name, mask in self.zeros_mask_dict.items():
            masker = self.zeros_mask_dict[name]
            masker.mask = loaded_masks[name]
            if masker.mask is not None:
                masker.mask = masker.mask.to(device)

    def init_from_masks_dict(self, masks_dict, normalize_dataparallel_keys=False):
        """This is a convenience function to initialize a CompressionScheduler from a dictionary
        Args:
            masks_dict (list): A dictionary formatted as {parameter_name: 4D mask tensor}
            normalize_dataparallel_keys (bool): indicates if we should convert the keys from
                DataParallel format.
        """
        for name, mask in self.zeros_mask_dict.items():
            if name not in masks_dict:
                masks_dict[name] = None
        state = {'masks_dict': masks_dict}
        self.load_state_dict(state, normalize_dataparallel_keys)

    @staticmethod
    def verify_policy_loss(policy_loss):
        if not isinstance(policy_loss, ply.PolicyLoss):
            raise TypeError("A Policy's before_backward_pass must return either None or an instance of " +
                            PolicyLoss.__name__)
        curr_loss_components = policy_loss.loss_components
        if not isinstance(curr_loss_components, list):
            curr_loss_components = [curr_loss_components]
        if not all(isinstance(lc, ply.LossComponent) for lc in curr_loss_components):
            raise TypeError("Expected an instance of " + LossComponent.__name__ +
                            " or a list of such instances")
        return curr_loss_components

### Master deployed here @@@@@
class ParameterMasker(object):
    """A ParameterMasker can mask a parameter tensor or a gradients tensor.
    It is used when pruning DNN weights.
    """
    def __init__(self, param_name):
        self.mask = None                # Mask lazily initialized by pruners
        self.param_name = param_name    # For debug/logging purposes
        self.mask_pruner = None
        self.is_regularization_mask = False
        self.use_double_copies = False
        self.mask_on_forward_only = False
        self.unmasked_copy = None
        self.backward_hook_handle = None

    def apply_mask(self, parameter):
        """Apply a mask on the weights tensor (parameter)."""
        if self.mask is None:
            return
        if self.use_double_copies:
            self.unmasked_copy = parameter.clone().detach()
        self.mask_tensor(parameter)
        if self.is_regularization_mask:
            self.mask = None
        return parameter

    def mask_tensor(self, tensor):
        if self.mask is not None:
            tensor.data.mul_(self.mask)

    def mask_gradient(self, gradient):
        if self.mask is not None:
            return gradient.mul(self.mask)

    def revert_weights(self, parameter, train_step):
        if not self.use_double_copies or self.unmasked_copy is None:
            if train_step == 0:
                # msglogger.debug('Parameter {0} does not maintain double copies'.format(self.param_name))
                # print('Parameter {0} does not maintain double copies'.format(self.param_name))
                # Double copies issue is not a problem, confirmed in the original repo on April 20, 2020.
                pass
            return 
        parameter.data.copy_(self.unmasked_copy)
        self.unmasked_copy = None


def create_model_masks_dict(model):
    """A convenience function to create a dictionary of parameter maskers for a model"""
    zeros_mask_dict = {}
    for name, param in model.named_parameters():
        masker = ParameterMasker(name)
        zeros_mask_dict[name] = masker
    return zeros_mask_dict

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