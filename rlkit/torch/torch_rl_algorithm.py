import abc
from collections import OrderedDict
from typing import Iterable
import pickle

import numpy as np

from rlkit.core import logger
from rlkit.core.eval_util import dprint
from rlkit.core.rl_algorithm import MetaRLAlgorithm
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, np_ify, torch_ify
from rlkit.core import logger, eval_util


class MetaTorchRLAlgorithm(MetaRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(self, *args, render_eval_paths=False, plotter=None, dump_eval_paths=False, output_dir=None, recurrent=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.plotter = plotter
        self.dump_eval_paths = dump_eval_paths
        self.output_dir = output_dir
        self.recurrent = recurrent

    ###### Torch stuff #####
    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def get_batch(self, idx=None):
        ''' get a batch from replay buffer for input into net '''
        if idx is None:
            idx = self.task_idx
        batch = self.replay_buffer.random_batch(idx, self.batch_size)
        return np_to_pytorch_batch(batch)

    def get_encoding_batch(self, idx=None, batch_size=None, eval_task=False):
        ''' get a batch from the separate encoding replay buffer '''
        # n.b. if eval is online, training should match the distribution of context lengths
        # and should sample trajectories instead of unordered transitions
        if batch_size is None:
            batch_size = self.embedding_batch_size
        # n.b. if using sequence model for encoder, samples should be ordered
        is_seq = self.recurrent
        padded = False # TODO decide if appropriate for e.g. RNN training
        if idx is None:
            idx = self.task_idx
        if eval_task:
            batch = self.eval_enc_replay_buffer.random_batch(idx, batch_size=batch_size, sequence=is_seq, padded=padded)
        else:
            batch = self.enc_replay_buffer.random_batch(idx, batch_size=batch_size, sequence=is_seq, padded=padded)
        return np_to_pytorch_batch(batch)

    ##### Eval stuff #####
    def obtain_eval_paths(self, idx, deterministic=False, prior=False):
        '''
        collect paths with current policy
        each transition will update the running latent context
        the context used to condition the policy can be resampled at different intervals
        (to enable trajectory-level or transition-level adaptation, for example)
        '''
        self.reset_posterior()
        resample = 'never' if prior else self.resample_z
        test_paths = self.eval_sampler.obtain_samples(deterministic=deterministic, resample=resample)
        if self.sparse_rewards:
            for p in test_paths:
                p['rewards'] = ptu.sparsify_rewards(p['rewards'])
        return test_paths

    def collect_paths(self, idx, epoch):
        self.task_idx = idx
        dprint('Task:', idx)
        self.env.reset_task(idx)
        num_evals = self.num_evals
        paths = []
        for _ in range(num_evals):
            paths += self.obtain_eval_paths(idx, deterministic=True)
        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}'.format(idx, epoch))

        return paths

    def evaluate(self, epoch):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        self.eval_statistics = statistics

        ### sample trajectories from prior for vis
        prior_paths = []
        for _ in range(10):
            prior_paths += self.obtain_eval_paths(None, deterministic=True, prior=True)
        if self.dump_eval_paths:
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        train_avg_returns = []
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        dprint('evaluating on {} train tasks'.format(len(indices)))
        for idx in indices:
            self.eval_enc_replay_buffer.task_buffers[idx].clear()
            paths = self.collect_paths(idx, epoch)
            train_avg_returns.append(eval_util.get_average_returns(paths))

        ### test tasks
        dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_avg_returns = []
        for idx in self.eval_tasks:
            self.eval_enc_replay_buffer.task_buffers[idx].clear()
            paths = self.collect_paths(idx, epoch)
            test_avg_returns.append(eval_util.get_average_returns(paths))

            # save the final posterior
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.policy.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.policy.z_vars[0]))
                self.eval_statistics['Z mean eval'] = z_mean
                self.eval_statistics['Z variance eval'] = z_sig

            # TODO(KR) what does this do
            if hasattr(self.env, "log_diagnostics"):
                self.env.log_diagnostics(paths)

        avg_train_return = np.mean(train_avg_returns)
        avg_test_return = np.mean(test_avg_returns)
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return ptu.from_numpy(elem_or_tuple).float()


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }
