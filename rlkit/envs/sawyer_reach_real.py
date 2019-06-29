from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from rlkit.core.serializable import Serializable
import numpy as np
from gym.spaces import Box

from . import register_env


@register_env('sawyer-reach-real-3d')
class MultitaskSawyerReachEnv(Serializable, SawyerReachXYZEnv):
    def __init__(self,
            randomize_tasks=True,
            n_tasks=5,
            goal_thresh=-0.02,
            **kwargs):
        Serializable.quick_init(self, locals())
        SawyerReachXYZEnv.__init__(self, config_name='laudri_config', **kwargs)

        self.goal_thresh = goal_thresh
        # define a safe goal space, slighty smaller than the obs space to make sure the robot always has to move at least a little
        self.goal_low = self.observation_space.low + np.array([0.1, .05, .02])
        self.goal_high = self.observation_space.high + np.array([-0.02, -0.05, -0.1])
        self.goal_space = Box(self.goal_low, self.goal_high)

        # generate random goals
        if n_tasks == 1:
            self.goals = [self.pos_control_reset_position + np.array([.1, .1, -.1])]
        else:
            np.random.seed(1337)
            self.goals = [1 * np.random.uniform(self.goal_low, self.goal_high) for _ in range(n_tasks)]

        print('goals \n', self.goals)
        self.reset_task(0)

    def step(self, action):
        self._act(action)
        new_obs = self._get_obs()
        reward = self.compute_reward(action, new_obs)
        info = self._get_info()
        done = False
        if reward > self.goal_thresh:
            done = True
        return new_obs, reward, done, info

    def compute_reward(self, action, obs):
        return -np.linalg.norm(obs - self._goal)

    def reset_task(self, idx):
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return list(range(len(self.goals)))

    def get_tasks(self):
        return self.goals




