from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from rlkit.core.serializable import Serializable
import numpy as np
from gym.spaces import Box

from . import register_env


@register_env('sawyer_reach_real_3d')
class MultitaskSawyerReachEnv(Serializable, SawyerReachXYZEnv):
    def __init__(self,
            randomize_tasks=True,
            n_tasks=5,
            goal_thresh=-0.02,
            **kwargs):
        Serializable.quick_init(self, locals())
        SawyerReachXYZEnv.__init__(self, config_name='pearl_fjalar_config',
                                   **kwargs)

        self.goal_thresh = goal_thresh
        # define a safe goal space, slighty smaller than the obs space to make sure the robot always has to move at least a little
        self.goal_low = self.observation_space.low + np.array([0.1, .05, .02])
        self.goal_high = self.observation_space.high + np.array([-0.02, -0.05, -0.1])
        self.goal_space = Box(self.goal_low, self.goal_high)

        # generate random goals
        print("N TASKS: " + str(n_tasks))
        if n_tasks == 2:
            self.goals = [[0.6463305, 0.15157676, 0.27320544],
                     [0.60380304, 0.08447907, 0.24696944]]
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