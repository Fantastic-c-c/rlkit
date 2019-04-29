import numpy as np
from gym.envs.classic_control import PendulumEnv

@register_env('no-velocity-pendulum')
class NoVelocityPendulum(LightDarkEnv):
    def __init__(self):
        wrapped_env = PendulumEnv()
        super(LightDarkPointEnv, self).__init__(wrapped_env,
                                                dark_cond=lambda x: True,
                                                noise_fun=no_obs,
                                                xpos_indices=(2,))
