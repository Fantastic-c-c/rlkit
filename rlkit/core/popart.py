import torch
import numpy as np
from torch import nn as nn
import pdb
from functools import reduce

class PopArtLayer(nn.Module):
    def __init__(
            self,
            in_size=1,
            output_size=1,
            beta=1e-4,
            epsilon=1e-4,
            stable_rate=0.1,
            min_steps=1000,
            use_gpu=False,
            **kwargs
    ):        
        super(PopArtLayer, self).__init__()
        # Popart params
        self.mean = 0
        self.mean_of_square = 0
        self.step = 0
        self.pop_is_active = 0
        if use_gpu:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        self.kernel = torch.randn(in_size, output_size, dtype=torch.float32, requires_grad=True, device=device)
        self.bias = torch.randn(output_size, dtype=torch.float32, requires_grad=True, device=device)

        self.beta = beta
        self.epsilon = epsilon
        self.stable_rate = stable_rate
        self.min_steps = min_steps

    def forward(self, h):
        return h*self.kernel + self.bias

    def pop_art_update(self, x):
        """
        Performs ART (Adaptively Rescaling Targets) update,
        adjusting normalization parameters with respect to new targets x.
        Updates running mean, mean of squares and returns
        new mean and standard deviation for later use.
        """
        # assert len(x.shape) == 2, "Must be 2D (batch_size, time_steps)"
        x = x[:, 0]
        beta = self.beta
        old_kernel = self.kernel
        old_bias = self.bias
        old_online_mean = self.mean
        old_online_mean_of_square = self.mean_of_square
        step = self.step
        pop_is_active = self.pop_is_active

        def update_rule(old, new):
            """
            Update rule for running estimations,
            dynamically adjusting sensitivity with every time step
            to new data (see Eq. 10 in the paper).
            """
            nonlocal step
            step += 1
            adj_beta = beta / (1 - (1 - beta)**step)
            return (1 - adj_beta) * old + adj_beta * new

        def update_rule_static_beta(old, new):
            return (1 - self.beta) * old + self.beta * new

        x_means = np.stack([x.mean(), np.square(x).mean()])

        # Updating normalization parameters (for ART)

        online_mean = update_rule_static_beta(old_online_mean, x.mean())
        online_mean_of_square = update_rule(old_online_mean_of_square, np.square(x).mean())
        
        # online_mean, online_mean_of_square = reduce(
        #     update_rule, x_means,
        #     np.array([old_online_mean, old_online_mean_of_square]))

        # pdb.set_trace()
        old_std_dev = np.sqrt(
            max(old_online_mean_of_square - np.square(old_online_mean), self.epsilon))
        std_dev = np.sqrt(max(online_mean_of_square - np.square(online_mean), self.epsilon))
        old_std_dev = old_std_dev if old_std_dev > 0 else std_dev
        # Performing POP (Preserve the Output Precisely) update
        # but only if we are not in the beginning of the training
        # when both mean and std_dev are close to zero or still
        # stabilizing. Otherwise POP kernel (W) and bias (b) can
        # become very large and cause numerical instability.
        std_is_stable = (
            step > self.min_steps
            and np.abs(1 - old_std_dev / std_dev) < self.stable_rate)
        if (int(pop_is_active) == 1 or
                (std_dev > self.epsilon and std_is_stable)):
            new_kernel = old_std_dev * old_kernel / std_dev
            new_bias = (
                (old_std_dev * old_bias + old_online_mean - online_mean)
                / std_dev)
            pop_is_active = 1
        else:
            new_kernel, new_bias = old_kernel, old_bias
        # Saving updated parameters into graph variables

        self.kernel = torch.Tensor(new_kernel)
        self.bias = torch.Tensor(new_bias)
        self.mean = online_mean
        self.mean_of_square = online_mean_of_square
        self.step = step
        self.pop_is_active = pop_is_active
        return online_mean, std_dev

    def de_normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Converts previously normalized data into original values.
        """
        std_dev = np.sqrt(self.mean_of_square - np.square(self.mean))
        return (x * (std_dev if std_dev > 0 else self.epsilon)
                + self.mean)

    def update_and_normalize(self, x: np.ndarray):
        """
        Normalizes given tensor `x` and updates parameters associated
        with PopArt: running means (art) and network's output scaling (pop).
        """
        mean, std_dev = self.pop_art_update(x)
        result = ((x - mean) / (std_dev if std_dev > 0 else self.epsilon))
        return result, mean, std_dev