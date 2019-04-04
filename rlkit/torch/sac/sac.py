from collections import OrderedDict
import numpy as np
import pickle
from typing import Iterable

import torch
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, np_ify, torch_ify
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import MetaRLAlgorithm
from rlkit.torch.sac.proto import ProtoAgent


class ProtoSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            reparameterize=True,
            recurrent=False,
            use_information_bottleneck=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            policy=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.eval_statistics = None
        self.kl_lambda = kl_lambda

        self.reparameterize = reparameterize
        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards

        # TODO consolidate optimizers!
        self.policy_optimizer = optimizer_class(
            self.policy.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.policy.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.policy.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.policy.vf.parameters(),
            lr=vf_lr,
        )
        self.context_optimizer = optimizer_class(
            self.policy.task_enc.parameters(),
            lr=context_lr,
        )

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

    def get_encoding_batch(self, idx=None, batch_size=None):
        ''' get a batch from the separate encoding replay buffer '''
        if batch_size is None:
            batch_size = self.embedding_batch_size
        # if using sequence model for encoder, samples should be ordered
        is_seq = self.recurrent
        if idx is None:
            idx = self.task_idx
        batch = self.enc_replay_buffer.random_batch(idx, batch_size=batch_size, sequence=is_seq)
        return np_to_pytorch_batch(batch)

    def sample_data(self, indices, encoder=False):
        # sample from replay buffer for each task
        # TODO(KR) this is ugly af
        obs, actions, rewards, next_obs, terms = [], [], [], [], []
        for idx in indices:
            if encoder:
                batch = self.get_encoding_batch(idx=idx)
            else:
                batch = self.get_batch(idx=idx)
            o = batch['observations'][None, ...]
            a = batch['actions'][None, ...]
            if encoder and self.sparse_rewards:
                # in sparse reward settings, only the encoder is trained with sparse reward
                r = batch['sparse_rewards'][None, ...]
            else:
                r = batch['rewards'][None, ...]
            no = batch['next_observations'][None, ...]
            t = batch['terminals'][None, ...]
            obs.append(o)
            actions.append(a)
            rewards.append(r)
            next_obs.append(no)
            terms.append(t)
        obs = torch.cat(obs, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        next_obs = torch.cat(next_obs, dim=0)
        terms = torch.cat(terms, dim=0)
        return [obs, actions, rewards, next_obs, terms]

    def prepare_encoder_data(self, obs, act, rewards):
        ''' prepare task data for encoding '''
        # for now we embed only observations and rewards
        # assume obs and rewards are (task, batch, feat)
        task_data = torch.cat([obs, act, rewards], dim=2)
        return task_data


    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        batch = self.sample_data(indices, encoder=True)

        # zero out context and hidden encoder state
        self.reset_posterior(num_tasks=len(indices))

        for i in range(num_updates):
            # TODO(KR) argh so ugly
            mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in batch]
            obs_enc, act_enc, rewards_enc, _, _ = mini_batch
            self._take_step(indices, obs_enc, act_enc, rewards_enc)

            # stop backprop
            self.policy.detach_z()

    def _take_step(self, indices, obs_enc, act_enc, rewards_enc):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)
        enc_data = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc)

        # run inference in networks
        q1_pred, q2_pred, v_pred, policy_outputs, target_v_values, task_z = self.policy(obs, actions, next_obs, enc_data, obs_enc, act_enc)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.policy.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self.policy.min_q(obs, new_actions, task_z)

        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()
        self.policy._update_target_network()

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        if self.reparameterize:
            policy_loss = (
                    log_pi - log_policy_target
            ).mean()
        else:
            policy_loss = (
                log_pi * (log_pi - log_policy_target + v_pred).detach()
            ).mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            # TODO this is kind of annoying and higher variance, why not just average
            # across all the train steps?
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                # TODO should average across tasks rather than tasking the first
                z_mean = np.mean(np.abs(ptu.get_numpy(self.policy.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.policy.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
    #####
    def reset_posterior(self, num_tasks=1):
        # reset to prior and sample z
        self.policy.clear_z(num_tasks=num_tasks)

    def infer_posterior(self, idx, batch_size=None):
        # infer q(z | c) given context
        if batch_size == None:
            batch_size = self.embedding_batch_size
        batch = self.get_encoding_batch(idx=idx, batch_size=batch_size)
        obs = batch['observations'][None, ...]
        act = batch['actions'][None, ...]
        rewards = batch['rewards'][None, ...]
        in_ = self.prepare_encoder_data(obs, act, rewards)
        self.policy.infer_posterior(in_)
        self.policy.sample_z()

    @property
    def networks(self):
        return self.policy.networks + [self.policy]

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.policy.qf1.state_dict(),
            qf2=self.policy.qf2.state_dict(),
            policy=self.policy.policy.state_dict(),
            vf=self.policy.vf.state_dict(),
            target_vf=self.policy.target_vf.state_dict(),
            task_enc=self.policy.task_enc.state_dict(),
        )
        return snapshot


# TODO: make a new file to place these utils, maybe torch/sac/sac_utils.py? alternatively just move them into pytorch utils
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
