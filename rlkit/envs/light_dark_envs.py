import numpy as np

from rlkit.envs.light_dark_wrapper import LightDarkEnv, no_obs, gaussian_noise
from rlkit.envs.point_robot import PointEnv

from . import register_env

@register_env('light-dark-point-robot')
class LightDarkPointEnv(LightDarkEnv):
    def __init__(self):
        wrapped_env = PointEnv()
        super(LightDarkPointEnv, self).__init__(wrapped_env,
                                                # dark_cond=lambda x: False,
                                                noise_fun=no_obs)
