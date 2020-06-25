#
# Copyright (c) 2019 Intel Corporation
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

"""
Performance trackers used to track the best performing epochs when training.
"""
import operator
import distiller

class MutableNamedTuple(dict):
    def __init__(self, init_dict):
        for k, v in init_dict.items():
            self[k] = v

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val

__all__ = ["TrainingPerformanceTracker",
           "SparsityAccuracyTracker"]


class TrainingPerformanceTracker(object):
    """Base class for performance trackers using Top1 and Top5 accuracy metrics"""
    def __init__(self, num_best_scores):
        self.perf_scores_history = []
        self.max_len = num_best_scores

    def reset(self):
        self.perf_scores_history = []

    def step(self, model, epoch, **kwargs):
        """Update the list of top training scores achieved so far"""
        raise NotImplementedError

    def best_scores(self, how_many=1):
        """Returns `how_many` best scores experienced so far"""
        if how_many < 1:
            how_many = self.max_len
        how_many = min(how_many, self.max_len)
        return self.perf_scores_history[:how_many]


class SparsityAccuracyTracker(TrainingPerformanceTracker):
    """A performance tracker which prioritizes non-zero parameters.
    Sort the performance history using the count of non-zero parameters
    as main sort key, then sort by top1, top5 and and finally epoch number.
    Expects 'top1' and 'top5' to appear in the kwargs.
    """
    def step(self, model, epoch, **kwargs):
        #assert all(score in kwargs.keys() for score in ('top1', 'top5'))
        model_sparsity, _, params_nnz_cnt = distiller.model_params_stats(model)
        #print(kwargs.keys())

        if kwargs['adv_train']:
            if 'adv_top5' in kwargs.keys(): 
                assert all(score in kwargs.keys() for score in ('adv_top1', 'adv_top5', 'nat_top1','nat_top5'))
                self.perf_scores_history.append(distiller.MutableNamedTuple({
                    'params_nnz_cnt': -params_nnz_cnt,
                    'sparsity': model_sparsity,
                    'nat_top1': kwargs['nat_top1'],
                    'nat_top5': kwargs['nat_top5'],
                    'adv_top1': kwargs['adv_top1'],
                    'adv_top5': kwargs['adv_top5'],
                    'epoch': epoch}))
                    
                # Keep perf_scores_history sorted from best to worst
                self.perf_scores_history.sort(
                    key=operator.attrgetter('params_nnz_cnt', 'nat_top1', 'nat_top5', 'adv_top1', 'adv_top5', 'epoch'), #top5'
                    reverse=True)
                    
            else: 
                assert all(score in kwargs.keys() for score in ['top1'])
                self.perf_scores_history.append(distiller.MutableNamedTuple({
                    'params_nnz_cnt': -params_nnz_cnt,
                    'sparsity': model_sparsity,
                    'nat_top1': kwargs['nat_top1'],
                    'adv_top1': kwargs['adv_top1'],
                    'epoch': epoch}))
                # Keep perf_scores_history sorted from best to worst
                self.perf_scores_history.sort(
                    key=operator.attrgetter('params_nnz_cnt', 'nat_top1', 'adv_top1', 'epoch'), #top5'
                    reverse=True)

        else: 
            if 'top5' in kwargs.keys(): 
                assert all(score in kwargs.keys() for score in ('top1', 'top5'))
                self.perf_scores_history.append(distiller.MutableNamedTuple({
                    'params_nnz_cnt': -params_nnz_cnt,
                    'sparsity': model_sparsity,
                    'top1': kwargs['top1'],
                    'top5': kwargs['top5'],
                    'epoch': epoch}))
                # Keep perf_scores_history sorted from best to worst
                self.perf_scores_history.sort(
                    key=operator.attrgetter('params_nnz_cnt', 'top1', 'top5', 'epoch'), #top5'
                    reverse=True)
            else: 
                assert all(score in kwargs.keys() for score in ['top1'])
                self.perf_scores_history.append(distiller.MutableNamedTuple({
                    'params_nnz_cnt': -params_nnz_cnt,
                    'sparsity': model_sparsity,
                    'top1': kwargs['top1'],
                    'epoch': epoch}))
                # Keep perf_scores_history sorted from best to worst
                self.perf_scores_history.sort(
                    key=operator.attrgetter('params_nnz_cnt', 'top1', 'epoch'), #top5'
                    reverse=True)


if __name__ == "__main__":
    #def step(epoch, **kwargs):
    #    print(kwargs.keys())
    import torchvision.models as models
    model = models.resnet18()
    tracker = SparsityAccuracyTracker(1)
    tracker.step(model, epoch=1, top1=1, top5=5, adv_train=False)
    tracker.step(model, epoch=2, top1=0, top5=6, adv_train=False)
    tracker.step(model, epoch=2, top1=0.2, top5=2, adv_train=False)
    tracker.step(model, epoch=2, top1=0.4, top5=3, adv_train=False)
    tracker.step(model, epoch=2, top1=5, top5=11, adv_train=False)
    print(tracker.best_scores(-1))
