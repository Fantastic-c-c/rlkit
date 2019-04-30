import numpy as np
from rlkit.envs.wrappers import ProxyEnv

def gaussian_noise(xpos, scale):
    return xpos + np.random.normal(scale=scale, size=np.shape(xpos))

def no_obs(xpos):
    return np.zeros(np.shape(xpos))

# TODO: move this to wrappers.py and do serializable stuff?
class LightDarkEnv(ProxyEnv):
    def __init__(self, wrapped_env,
                 dark_cond=lambda xy: xy[0] < 0,
                 noise_fun=lambda x: gaussian_noise(x, 0.1),
                 xpos_indices=(0, 1)):
        self._wrapped_env = wrapped_env
        self._dark_cond = dark_cond
        self._noise_fun = noise_fun
        self._xpos_indices = np.array(xpos_indices)
        ProxyEnv.__init__(self, wrapped_env)

    def __getattr__(self, item):
        if item in self.__dict__:
            return getattr(self, item)

        return getattr(self._wrapped_env, item)

    def step(self, action):
        ob, reward, done, d = super().step(action)
        new_ob = self._get_obs()
        # assumes wrapped env's obs is the true state
        d['state'] = self._wrapped_env._get_obs()
        return new_ob, reward, done, d

    def _get_obs(self):
        obs = self._wrapped_env._get_obs()
        xpos = obs[self._xpos_indices] # gets xy as numpy array
        if self._dark_cond(xpos):
            obs_xpos = self._noise_fun(xpos)
            obs[self._xpos_indices] = obs_xpos
        return obs
