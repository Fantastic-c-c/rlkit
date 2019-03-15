import numpy as np

import torch
from torch import Tensor
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_ify, torch_ify


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class ProtoAgent(nn.Module):

    def __init__(self,
                 env,
                 latent_dim,
                 nets,
                 **kwargs
    ):
        super().__init__()
        self.env = env
        self.latent_dim = latent_dim
        self.task_enc, self.policy, self.qf1, self.qf2, self.vf = nets
        self.target_vf = self.vf.copy()
        self.recurrent = kwargs['recurrent']
        self.reparam = kwargs['reparameterize']
        self.use_ib = kwargs['use_information_bottleneck']
        self.tau = kwargs['soft_target_tau']
        self.reward_scale = kwargs['reward_scale']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.det_z = False

        # initialize task embedding to zero
        # (task, latent dim)
        self.register_buffer('z', torch.zeros(1, latent_dim))
        # for incremental update, must keep track of number of datapoints accumulated
        self.register_buffer('num_z', torch.zeros(1))

        # initialize posterior to the prior
        mu = torch.zeros(1, latent_dim)
        if self.use_ib:
            sigma_squared = torch.ones(1, latent_dim)
        else:
            sigma_squared = torch.zeros(1, latent_dim)
        self.register_buffer('z_means', mu)
        self.register_buffer('z_vars', sigma_squared)

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        self.sample_z()
        self.task_enc.reset(num_tasks) # clear hidden state in recurrent case

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.task_enc.hidden = self.task_enc.hidden.detach()

    def update_context(self, inputs):
        ''' update q(z|c) with a single transition '''
        # TODO there should be one generic method for preparing data for the encoder!!!
        o, a, r, no, d = inputs
        if self.sparse_rewards:
            r = self.env.sparsify_rewards(r)
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        data = torch.cat([o, a, r], dim=2)
        self.update_posterior(data)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, in_):
        ''' compute q(z|c) as a function of input context '''
        params = self.task_enc(in_)
        params = params.view(in_.size(0), -1, self.task_enc.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)

    def sample_z(self):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def update_posterior(self, in_):
        '''
        update current q(z|c) with new data
         - by adding to sum of natural parameters for prototypical encoder
         - by updating hidden state for recurrent encoder
        '''
        num_z = self.num_z

        # TODO this only works for single task (t == 1)
        up_params = self.task_enc(in_) # task x batch x feat
        if up_params.size(0) != 1:
            raise Exception('incremental update for more than 1 task not supported')
        if self.recurrent:
            up_mu = up_params[..., :self.latent_dim]
            up_ss = F.softplus(up_params[..., self.latent_dim:])
            self.z_means = up_mu[None]
            self.z_vars = up_ss[None]
        else:
            num_updates = up_params.size(1)
            # only have one task here
            up_params = up_params[0] # batch x feat
            curr_mu = self.z_means[0]
            curr_ss = self.z_vars[0]
            if self.use_ib:
                natural_z = torch.cat(_canonical_to_natural(curr_mu, curr_ss)) #feat
                up_mu = up_params[..., :self.latent_dim]
                up_ss = F.softplus(up_params[..., self.latent_dim:])
                new_natural = torch.cat(_canonical_to_natural(up_mu, up_ss), dim=1) # batch x feat
                new_natural = torch.sum(new_natural, dim=0)
                natural_z += new_natural
                m, s  = _natural_to_canonical(natural_z[:self.latent_dim], natural_z[self.latent_dim:]) # feat
                self.z_means = m[None]
                self.z_vars = s[None]
            else:
                for i in range(num_updates):
                    num_z += 1
                    curr_mu += (up_params[i] - curr_mu) / num_z
                self.z_means = curr_mu[None]

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.tau)

    def forward(self, obs, actions, next_obs, enc_data, obs_enc, act_enc):
        self.infer_posterior(enc_data)
        self.sample_z()
        return self.infer_ac(obs, actions, next_obs, obs_enc, act_enc)

    def infer_ac(self, obs, actions, next_obs, obs_enc, act_enc):
        '''
        compute predictions of SAC networks for update

        regularize encoder with reward prediction from latent task embedding
        '''

        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.contiguous().view(t * b, -1)
        actions = actions.contiguous().view(t * b, -1)
        next_obs = next_obs.contiguous().view(t * b, -1)
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

        return q1, q2, v, policy_outputs, target_v_values, task_z

    def min_q(self, obs, actions, task_z):
        t, b, _ = obs.size()
        obs = obs.contiguous().view(t * b, -1)

        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    @property
    def networks(self):
        return [self.task_enc, self.policy, self.qf1, self.qf2, self.vf, self.target_vf]




