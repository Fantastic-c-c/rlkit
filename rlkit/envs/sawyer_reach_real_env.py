from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
import numpy as np


class PearlSawyerReachXYZEnv(SawyerReachXYZEnv):
    def __init__(self, *args, randomize_tasks=True, n_tasks=5,
                 reward_type='hand_distance',
                 **kwargs):
        SawyerReachXYZEnv.__init__(
            self,
            *args,
            reward_type,
            **kwargs
        )
        # self.observation_space = self.hand_space
        init_task_idx = 0
        self._task = None  # This is set by reset_task down below

        # hand_z_position = 0.055  # TODO: Remove this to go 3-dimensions

        directions = list(range(n_tasks))
        if randomize_tasks:
            goals = self.sample_goals(n_tasks)
            # goals = [[g[0], g[1], hand_z_position] for g in goals]
            # goals = [1 * np.random.uniform(-1., 1., 2) for _ in directions]
        else:
            # add more goals if we want non-randomized tasks
            goals = [
                     ]
            # goals = [[g[0], g[1], hand_z_position] for g in goals]
            if (n_tasks > len(goals)):
                raise NotImplementedError("We don't have enough goals defined")
        self.goals = np.asarray(goals)
        self.tasks = [{'direction': direction} for direction in directions]

        # set the initial goal
        self.reset_task(init_task_idx)

        self.reset()
        print("GOALS: " + str(goals))

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_goal(self, direction):
        return self.goals[direction]

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self.set_goal(self.reset_goal(self._task['direction']))
        self.reset()

    def reset(self):
        return self.reset_model()

    def reset_model(self):
        # We have redefined this not to reset the goal (as this is handled in reset_task)
        self._reset_robot()
        return self._get_obs()  # Redefine to just return state
