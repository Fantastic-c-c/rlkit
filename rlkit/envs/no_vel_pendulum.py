import numpy as np
from gym.envs.classic_control import PendulumEnv
from rlkit.envs.light_dark_wrapper import LightDarkEnv, no_obs, gaussian_noise

from . import register_env

@register_env('no-velocity-pendulum')
class NoVelocityPendulumEnv(LightDarkEnv):
    def __init__(self):
        wrapped_env = PendulumEnv()
        super(NoVelocityPendulumEnv, self).__init__(wrapped_env,
                                                dark_cond=lambda x: True,
                                                noise_fun=no_obs,
                                                xpos_indices=(2,))
