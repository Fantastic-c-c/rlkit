from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from sawyer_control.core.serializable import Serializable
import numpy as np


@register_env('sawyer-reach-real-3d')
class PearlSawyerReachXYZEnv(SawyerReachXYZEnv):
    def __init__(self, *args, randomize_tasks=True, n_tasks=5,
                 height_2d=None,
                 goal_thresh=-0.05,
                 **kwargs):
        Serializable.quick_init(self, locals())
        SawyerReachXYZEnv.__init__(
            self,
            *args,
            height_2d=height_2d,
            **kwargs
        )

        self.goal_thresh = goal_thresh
        # self.observation_space = self.hand_space
        init_task_idx = 0
        self._task = None  # This is set by reset_task down below

        directions = list(range(n_tasks))
        if randomize_tasks:
            goals = [1 * np.random.uniform(self.goal_low, self.goal_high) for _ in directions]
            if height_2d and height_2d >= 0:
                goals = [[g[0], g[1], height_2d] for g in goals]
        else:
            # add more goals if we want non-randomized tasks
            goals = [
                [0.6, 0.2, 0.25],
                [0.6, -0.2, 0.25]
                     ]
            # goals = [[g[0], g[1], hand_z_position] for g in goals]
            if (n_tasks > len(goals)):
                raise NotImplementedError("We don't have enough goals defined")
        self.goals = np.asarray(goals)
        self.tasks = [{'direction': direction} for direction in directions]

        # set the initial goal
        self.reset_task(init_task_idx)

    def step(self, action):
        observation, reward, _, info = super().step(action)
        done = reward > self.goal_thresh  # threshold is negative
        if done:
            print("Close enough to goal - done!")
        return observation, reward, done, info

    def get_all_goals(self):
        return self.goals

    def get_goal_at(self, idx):
        return self.goals[idx]

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_goal(self, direction):
        return self.goals[direction]

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self.set_goal(self.reset_goal(self._task['direction']))
        self._goal = self._task['direction']  # TODO: Is this necessary?
        self.reset()

    def reset(self):
        return self.reset_model()

    def reset_model(self):
        # We have redefined this not to reset the goal (as this is handled in reset_task)
        old_goal = self.get_goal()
        self._reset_robot()
        return self._get_obs()  # Redefine to just return state

