from rlkit.core.serializable import Serializable
import gym
import numpy as np
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_wall_6dof import SawyerReachPushPickPlaceWall6DOFEnv


class HardModeEnv(gym.Env, Serializable):
    def __init__(self, task_list):
        Serializable.quick_init(self, locals())
        self._task_envs = []
        for i, task in enumerate(task_list):
            if task is SawyerReachPushPickPlace6DOFEnv or task is SawyerReachPushPickPlaceWall6DOFEnv:
            # TODO: this could cause flaws in task_idx if SawyerReachPushPickPlace6DOFEnv/SawyerReachPushPickPlaceWall6DOFEnv is not the first environment
                self._task_envs.append(task(multitask=False, obs_type='with_goal', random_init=True, if_render=False, fix_task=True, task_idx=i%3))
            else:
                self._task_envs.append(task(multitask=False, obs_type='with_goal', if_render=False, random_init=True))
        self._active_task = None

    def reset(self, **kwargs):
        return self.active_env.reset(**kwargs)

    @property
    def action_space(self):
        return self.active_env.action_space

    @property
    def observation_space(self):
        return self.active_env.observation_space

    def step(self, action):
        obs, reward, done, info = self.active_env.step(action)
        info['task'] = self.active_task_one_hot
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.active_env.render(*args, **kwargs)

    def close(self):
        for env in self._task_envs:
            env.close()

    @property
    def task_space(self):
        n = len(self._task_envs)
        one_hot_ub = np.ones(n)
        one_hot_lb = np.zeros(n)
        return gym.spaces.Box(one_hot_lb, one_hot_ub, dtype=np.float32)

    @property
    def active_task(self):
        return self._active_task

    @property
    def active_task_one_hot(self):
        one_hot = np.zeros(self.task_space.shape)
        t = self.active_task or 0
        one_hot[t] = self.task_space.high[t]
        return one_hot

    @property
    def active_env(self):
        return self._task_envs[self.active_task or 0]

    @property
    def num_tasks(self):
        return len(self._task_envs)

    '''
    API's for MAML Sampler
    '''
    def sample_tasks(self, meta_batch_size):
        return np.random.randint(0, self.num_tasks, size=meta_batch_size)
    
    def set_task(self, task):
        self._active_task = task

    # def log_diagnostics(self, paths, prefix):
    #     pass

    '''
    API's for PEARL
    '''
    def reset_task(self, idx):
        self._active_task = idx
        self.reset()

    def get_all_task_idx(self):
        return range(len(self._task_envs))
