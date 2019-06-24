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

        # define a safe goal space
        self.goal_low = self.observation_space.low
        self.goal_high = self.observation_space.high

        # generate random goals
        if n_tasks == 1:
            self.goals = [self._reset_position + 0.15]
        else:
            self.goals = [1 * np.random.uniform(self.goal_low, self.goal_high) for _ in range(n_tasks)]

        print('goals \n', self.goals)
        self.reset_task(0)

    def reset_task(self, idx):
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return list(range(len(self.goals)))




