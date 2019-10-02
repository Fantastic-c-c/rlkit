from collections import OrderedDict
import numpy as np
import pickle

import torch
import torch.optim as optim
from torch import nn as nn
from torch.autograd import Variable

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_ify, torch_ify
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import MetaTorchRLAlgorithm
from rlkit.torch.sac.proto import ProtoAgent


class ProtoSoftActorCritic(MetaTorchRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,

            class_lr=1e-1,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            rf_loss_scale=1.,
            policy_mean_reg_weight=0.,#1e-3,
            policy_std_reg_weight=0.,#1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            reparameterize=True,
            use_information_bottleneck=False,
            sparse_rewards=False,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
            use_alpha_network=False,
            sep_alpha=True,
            **kwargs
    ):
        super().__init__(
            env=env,
            policy=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            **kwargs
        )
        deterministic_embedding=False
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.latent_dim = latent_dim
        self.qf_criterion = nn.MSELoss()
        # self.vf_criterion = nn.MSELoss()
        self.rf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()
        self.eval_statistics = None
        self.kl_lambda = kl_lambda
        self.rf_loss_scale = rf_loss_scale

        self.reparameterize = reparameterize
        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        # self.alpha_network = nets[-1]
        self.use_alpha_network = use_alpha_network
        self.sep_alpha = sep_alpha
        if self.use_alpha_network:
            self.alpha_network = nets[-1]
            self.alpha_optimizer = optimizer_class(
                self.alpha_network.parameters(),
                lr=policy_lr,
            )
        else:
            self.alpha_network = None
            if self.sep_alpha:
                self.log_alpha = ptu.zeros(self.num_tasks, requires_grad=True)
            else:
                self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        # self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        # if self.use_automatic_entropy_tuning:
        if target_entropy:
            self.target_entropy = target_entropy
        else:
            self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
        # self.log_alpha = ptu.zeros(1, requires_grad=True)
        # self.alpha_optimizer = optimizer_class(
        #     self.alpha_network.parameters(),
        #     lr=policy_lr,
        # )

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
        # self.vf_optimizer = optimizer_class(
        #     self.policy.vf.parameters(),
        #     lr=vf_lr,
        # )
        self.context_optimizer = optimizer_class(
            self.policy.task_enc.parameters(),
            lr=context_lr,
        )
        self.rf_optimizer = optimizer_class(
            self.policy.rf.parameters(),
            lr=context_lr,
        )

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
        if self.sparse_rewards:
            rewards = ptu.sparsify_rewards(rewards)
        rewards = rewards / self.reward_scale
        task_data = torch.cat([obs, act, rewards], dim=2)
        return task_data

    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // mb_size

        batch = self.sample_data(indices, encoder=True)

        # zero out context and hidden encoder state
        self.policy.clear_z(num_tasks=len(indices))

        for i in range(num_updates):
            # TODO(KR) argh so ugly
            mini_batch = [x[:, i * mb_size: i * mb_size + mb_size, :] for x in batch]
            obs_enc, act_enc, rewards_enc, _, _ = mini_batch
            self._take_step(indices, obs_enc, act_enc, rewards_enc)

            # stop backprop
            self.policy.detach_z()

    def idx_to_one_hot(self, idx):
        # Returns a numpy array
        num_tasks = len(self.train_tasks) + len(self.eval_tasks)
        task_idx_one_hot = np.zeros(num_tasks)
        task_idx_one_hot[idx] = 1
        return task_idx_one_hot


    def _take_step(self, indices, obs_enc, act_enc, rewards_enc):

        num_tasks = len(indices)

        # data is (task, batch, feat)
        one_hot_task_idx = self.idx_to_one_hot(indices[0])
        obs, actions, rewards, next_obs, terms = self.sample_data(indices)
        enc_data = self.prepare_encoder_data(obs_enc, act_enc, rewards_enc)

        if self.use_alpha_network:
            log_alpha = self.alpha_network(torch.unsqueeze(ptu.from_numpy(one_hot_task_idx), dim=0))
        else:
            if self.sep_alpha:
                log_alpha = torch.matmul(torch.unsqueeze(ptu.from_numpy(one_hot_task_idx), dim=0), torch.unsqueeze(self.log_alpha, dim=1))
            else:
                log_alpha = self.log_alpha
        alpha = torch.exp(log_alpha)

        # run inference in networks
        r_pred, q1_pred, q2_pred, policy_outputs, target_q_values, task_z = self.policy(obs, actions, next_obs, enc_data, obs_enc, act_enc, alpha)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        # if self.use_automatic_entropy_tuning:
        #     """
        #     Alpha Loss
        #     """
        #     alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        #     self.alpha_optimizer.zero_grad()
        #     alpha_loss.backward()
        #     self.alpha_optimizer.step()
        #     alpha = self.log_alpha.exp()
        # else:
        #     alpha = 1
        #     alpha_loss = 0
        # log_alpha = self.alpha_network(torch.unsqueeze(ptu.from_numpy(one_hot_task_idx), dim=0))
        # alpha = torch.exp(log_alpha)
        alpha_loss = -(log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optimizer.step()

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.policy.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # auxiliary reward prediction from encoder states
        rewards_enc_flat = rewards_enc.contiguous().view(self.embedding_mini_batch_size * num_tasks, -1)
        rf_loss = self.rf_loss_scale * self.rf_criterion(r_pred, rewards_enc_flat)
        self.rf_optimizer.zero_grad()
        rf_loss.backward(retain_graph=True)
        self.rf_optimizer.step()

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_q_values
        # qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        # qf_loss.backward()
        # self.qf1_optimizer.step()
        # self.qf2_optimizer.step()
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()
        self.context_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = self.policy.min_q(obs, new_actions, task_z)

        # policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        if self.reparameterize:
            policy_loss = (
                    alpha*log_pi - log_policy_target
            ).mean()
        else:
            policy_loss = (
                alpha*log_pi * (log_pi - log_policy_target + v_pred).detach()
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
                z_mean = np.mean(np.abs(ptu.get_numpy(self.policy.z_dists[0].mean)))
                z_sig = np.mean(ptu.get_numpy(self.policy.z_dists[0].variance))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)
            self.eval_statistics['Alpha'] = alpha.item()
            self.eval_statistics['Alpha Loss'] = alpha_loss.item()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['RF Loss'] = np.mean(ptu.get_numpy(rf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'R Predictions',
                ptu.get_numpy(r_pred)
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

    def sample_z_from_prior(self):
        self.policy.clear_z()

    def sample_z_from_posterior(self, idx, eval_task=False):
        batch = self.get_encoding_batch(idx=idx, eval_task=eval_task)
        obs = batch['observations'][None, ...]
        act = batch['actions'][None, ...]
        rewards = batch['rewards'][None, ...]
        in_ = self.prepare_encoder_data(obs, act, rewards)
        self.policy.set_z(in_)

    @property
    def networks(self):
        if self.use_alpha_network:
            return self.policy.networks + [self.policy] + [self.alpha_network]
        else:
            return self.policy.networks + [self.policy]

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(
            qf1=self.policy.qf1,
            qf2=self.policy.qf2,
            policy=self.policy.policy,
            rf=self.policy.rf,
            target_qf1=self.policy.target_qf1,
            target_qf2=self.policy.target_qf2,
            task_enc=self.policy.task_enc,
        )
        return snapshot
