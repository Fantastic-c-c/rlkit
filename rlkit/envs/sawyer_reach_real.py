from sawyer_control.envs.sawyer_reaching import SawyerReachEnv
import numpy as np

from . import register_env


@register_env('sawyer-reach-real-3d')
class MultitaskSawyerReachEnv(SawyerReachEnv):
    def __init__(self,
            *args,
            randomize_tasks=True,
            n_tasks=5,
            **kwargs):
        super().__init__(*args, **kwargs)

        # define a safe goal space, slighty smaller than the obs space to make sure the robot always has to move at least a little
        self.goal_low = self.observation_space.low + np.array([0.1, .05, .02])
        self.goal_high = self.observation_space.high + np.array([-0.02, -0.05, -0.1])

        # generate random goals
        if n_tasks == 1:
            self.goals = [self._reset_position + 0.15]
        else:
            np.random.seed(1337)
            self.goals = [1 * np.random.uniform(self.goal_low, self.goal_high) for _ in range(n_tasks)]

        print('goals \n', self.goals)
        self.reset_task(0)

    def reset_task(self, idx):
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return list(range(len(self.goals)))

    def get_tasks(self):
        return self.goals




