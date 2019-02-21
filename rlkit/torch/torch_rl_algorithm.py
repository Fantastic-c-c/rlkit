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
            if batch_size == -1:
                batch = self.eval_enc_replay_buffer.all_data(idx, starts=False)
            else:
                batch = self.eval_enc_replay_buffer.random_batch(idx, batch_size=batch_size, sequence=is_seq, padded=padded)
        else:
            if batch_size == -1:
                batch = self.enc_replay_buffer.all_data(idx, starts=False)
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
        test_paths = []
        self.reset_posterior()
        if prior:
            return self.eval_sampler.obtain_samples(num_samples = 1 * self.max_path_length + 1, deterministic=deterministic, resample='trajectory')

        # warm start buffer with some trajectories from the prior
        for _ in range(10):
            self.sample_z()
            paths = self.eval_sampler.obtain_samples(num_samples = 1 * self.max_path_length + 1, deterministic=deterministic, resample='never')
            self.eval_enc_replay_buffer.task_buffers[idx].add_path(paths[0])
            test_paths += paths

        eval_task = (idx in self.eval_tasks)

        print('z mean', np.mean(np.abs(ptu.get_numpy(self.policy.z_means)), axis=0))
        print('z sig', np.mean(ptu.get_numpy(self.policy.z_vars), axis=0))

        for _ in range(10):
            dprint('encoder buffer size task: {}'.format(idx), self.eval_enc_replay_buffer.task_buffers[idx].size())
            self.infer_posterior(idx, batch_size=-1, eval_task=True)
            print('z', self.policy.z.detach().cpu().numpy())
            paths = self.eval_sampler.obtain_samples(deterministic=deterministic, resample='never')
            self.eval_enc_replay_buffer.task_buffers[idx].add_path(paths[0])
            test_paths += paths

        print('z mean', np.mean(np.abs(ptu.get_numpy(self.policy.z_means)), axis=0))
        print('z sig', np.mean(ptu.get_numpy(self.policy.z_vars), axis=0))

        # collect multiple trajectories from final posterior to lower variance of result
        self.sample_z()
        paths = self.eval_sampler.obtain_samples(num_samples= 5 * self.max_path_length + 1, deterministic=deterministic, resample='never')
        test_paths += paths

        if self.sparse_rewards:
            for p in test_paths:
                p['rewards'] = self.env.sparsify_rewards(p['rewards'])
        return test_paths

    def collect_paths(self, idx, epoch, run):
        self.eval_enc_replay_buffer.task_buffers[idx].clear()
        self.task_idx = idx
        dprint('Task:', idx)
        self.env.reset_task(idx)
        paths = self.obtain_eval_paths(idx, deterministic=True)
        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            runs, all_rets = [], []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
                runs.append(paths)
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            final_returns.append(np.mean(all_rets[-5:])) # score last 5 rollouts
            online_returns.append(all_rets)
        return final_returns, online_returns

    def evaluate(self, epoch):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        self.eval_statistics = statistics

        ### sample trajectories from prior for vis
        prior_paths = []
        for _ in range(10):
            self.sample_z()
            prior_paths += self.obtain_eval_paths(None, deterministic=True, prior=True)
        if self.dump_eval_paths:
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        dprint('evaluating on {} train tasks'.format(len(indices)))
        train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        print('train online returns')
        print(train_online_returns)

        ### test tasks
        dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)
        print('test online returns')
        print(test_online_returns)

        # save the final posterior
        if self.use_information_bottleneck:
            z_mean = np.mean(np.abs(ptu.get_numpy(self.policy.z_means[0])))
            z_sig = np.mean(ptu.get_numpy(self.policy.z_vars[0]))
            self.eval_statistics['Z mean eval'] = z_mean
            self.eval_statistics['Z variance eval'] = z_sig

        # TODO(KR) what does this do
        #if hasattr(self.env, "log_diagnostics"):
            #self.env.log_diagnostics(paths)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

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
