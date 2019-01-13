import numpy as np

import torch
from torch import Tensor
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_ify, torch_ify


class ProtoAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 nets,
                 **kwargs
    ):
        super().__init__()
        self.task_enc, self.policy, self.qf1, self.qf2, self.vf, self.rf = nets
        self.latent_dim = latent_dim
        self.target_vf = self.vf.copy()
        self.recurrent = kwargs['recurrent']
        self.reparam = kwargs['reparameterize']
        self.use_ib = kwargs['use_information_bottleneck']
        self.tau = kwargs['soft_target_tau']
        self.reward_scale = kwargs['reward_scale']
        self.sparse_rewards = kwargs['sparse_rewards']

        # initialize task embedding to zero
        # (task, latent dim)
        self.register_buffer('z', torch.zeros(1, latent_dim))
        # for incremental update, must keep track of number of datapoints accumulated
        self.register_buffer('num_z', torch.zeros(1))

    def clear_z(self, num_tasks=1):
        # TODO in IB case, should be set to prior
        self.z = self.z.new_full((num_tasks, self.latent_dim), 0)
        self.task_enc.reset(num_tasks) # clear hidden state in recurrent case

    def detach_z(self):
        self.z = self.z.detach()
        if self.recurrent:
            self.task_enc.hidden = self.task_enc.hidden.detach()

    def update_context(self, inputs):
        ''' update task embedding with a single transition '''
        # TODO there should be one generic method for preparing data for the encoder!!!
        o, a, r, no, d = inputs
        if self.sparse_rewards:
            r = (r < .2).astype(float)
        r = r / self.reward_scale
        o = ptu.from_numpy(o[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        data = torch.cat([o, r], dim=2)
        self.update_z(data)

    def set_z(self, in_):
        ''' compute latent task embedding only from this input data '''
        new_z = self.task_enc(in_)
        new_z = new_z.view(in_.size(0), -1, self.latent_dim)
        new_z = torch.mean(new_z, dim=1)
        self.z = new_z

    def update_z(self, in_):
        '''
        update current task embedding
         - by running mean for prototypical encoder
         - by updating hidden state for recurrent encoder
        '''
        z = self.z
        num_z = self.num_z

        # TODO this only works for single task (t == 1)
        new_z = self.task_enc(in_)
        if new_z.size(0) != 1:
            raise Exception('incremental update for more than 1 task not supported')
        if self.recurrent:
            z = new_z
        else:
            new_z = new_z[0] # batch x feat
            num_updates = new_z.size(0)
            for i in range(num_updates):
                num_z += 1
                z += (new_z[i][None] - z) / num_z

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def _product_of_gaussians(mus, sigmas):
        '''
        compute mu, sigma of product of gaussians
        '''
        sigmas_squared = sigmas ** 2
        sigma_squared = 1. / torch.sum(torch.reciprocol(sigmas_squared), dim=0)
        mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
        return mu, torch.sqrt(sigma_squared)

    def _mean_of_gaussians(mus, sigmas):
        '''
        compute mu, sigma of mean of gaussians
        '''
        mu = torch.mean(mus, dim=0)
        sigma = torch.sqrt(torch.mean(sigmas**2, dim=0))
        return mu, sigma

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.tau)

    def forward(self, obs, actions, next_obs, enc_data, obs_enc):
        self.set_z(enc_data)
        return self.infer(obs, actions, next_obs, obs_enc)


    def infer(self, obs, actions, next_obs, obs_enc):
        '''
        compute predictions of SAC networks for update

        regularize encoder with reward prediction from latent task embedding
        '''

        task_z = self.z

        # auxiliary reward regression
        rf_z = [z.repeat(obs_enc.size(1), 1) for z in task_z]
        rf_z = torch.cat(rf_z, dim=0)
        r = self.rf(obs_enc.view(obs_enc.size(0) * obs_enc.size(1), -1), rf_z)

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # Q and V networks
        # encoder will only get gradients from Q nets
        q1 = self.qf1(obs, actions, task_z)
        q2 = self.qf2(obs, actions, task_z)
        v = self.vf(obs, task_z.detach())

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=self.reparam, return_log_prob=True)

        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        return r, q1, q2, v, policy_outputs, target_v_values, task_z

    def min_q(self, obs, actions, task_z):
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)

        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    @property
    def networks(self):
        return [self.task_enc, self.policy, self.qf1, self.qf2, self.vf, self.target_vf, self.rf]




