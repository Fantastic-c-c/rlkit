import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu


def _product_of_gaussians(mus, sigmas):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = sigmas ** 2
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, torch.sqrt(sigma_squared)


def _mean_of_gaussians(mus, sigmas):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma = torch.sqrt(torch.mean(sigmas**2, dim=0))
    return mu, sigma


class ProtoNet(nn.Module):

    def __init__(self,
                 latent_dim,
                 nets,
                 reparam=True,
                 use_ib=False,
                 det_z=False,
                 tau=1e-2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.task_enc, self.policy, self.qf1, self.qf2, self.vf, self.rf = nets
        self.target_vf = self.vf.copy()
        self.reparam = reparam
        self.use_ib = use_ib
        self.det_z = det_z
        self.tau = tau


    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.tau)

    def forward(self, obs, actions, next_obs, enc_data, obs_enc):
        if self.use_ib:
            task_z, kl_div = self.embed(enc_data, with_kl_div=True)
            return list(self.infer(obs, actions, next_obs, obs_enc, task_z)) + [kl_div]
        else:
            task_z = self.embed(enc_data)
            return self.infer(obs, actions, next_obs, obs_enc, task_z)

    def embed(self, in_, with_kl_div=False):
        '''
        compute latent task embedding from data

        if using info bottleneck, embedding is sampled from a
        gaussian distribution whose parameters are predicted
        '''
        # TODO: implement incremental task encoding by making task_z a class member
        # in_ should be (num_tasks x batch x feat)
        t, b, f = in_.size()
        in_ = in_.view(t * b, f)

        # compute embedding per task
        embeddings = self.task_enc(in_).view(t, b, -1)
        embeddings = torch.unbind(embeddings, dim=0)


        # TODO info bottleneck (WIP) need KL loss
        if self.use_ib:
            mus = [e[:, :self.latent_dim] for e in embeddings]
            sigmas = [F.softplus(e[:, self.latent_dim:]) for e in embeddings]
            z_params = [_product_of_gaussians(m, s) for m, s in zip(mus, sigmas)]
            if not self.det_z:
                z_dists = [torch.distributions.Normal(m, s) for m, s in z_params]
                task_z = [d.rsample() for d in z_dists]
                if with_kl_div:
                    prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
                    kl_divs = [torch.distributions.kl.kl_divergence(z_dist, prior) for z_dist in z_dists]
                    kl_div_sum = torch.sum(torch.stack(kl_divs))
            else:
                task_z = [p[0] for p in z_params]
            if any([torch.isnan(z).any() for z in task_z]):
                import ipdb; ipdb.set_trace()
        else:
            task_z = [torch.mean(e, dim=0) for e in embeddings]

        if with_kl_div:
            return task_z, kl_div_sum

        return task_z


    def infer(self, obs, actions, next_obs, obs_enc, task_z):
        '''
        compute predictions of SAC networks for update

        regularize encoder with reward prediction from latent task embedding
        '''

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




