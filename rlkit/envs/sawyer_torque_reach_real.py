from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from sawyer_control.core.serializable import Serializable
import numpy as np

@register_env('sawyer-torque-reach-real')
class PearlSawyerReachXYZTorqueEnv(SawyerReachXYZEnv):
    def __init__(self,
                 *args,
                 randomize_tasks=True,
                 n_tasks=5,
                 height_2d=None,
                 goal_thresh=-0.05,
                 goal_low=np.array([0.7, 0.0, 0.22]),
                 goal_high=np.array([0.75, 0.06, 0.25]),
                 **kwargs):
        Serializable.quick_init(self, locals())
        SawyerReachXYZEnv.__init__(
            self,
            *args,
            config_name="pearl_fjalar_config",
            action_mode="torque",
            max_speed=0.01,  # default val
            position_action_scale=0.03,  # default val
            height_2d = height_2d,
            **kwargs
        )

        self.goal_thresh = goal_thresh
        # self.ee_angle_lows = np.array([-0.05, -0.05, -np.pi * .75])
        # self.ee_angle_highs = np.array([0.05, 0.05, np.pi * .75])
        # self.ee_pos_lows = np.array([.671, -0.038, 0.195])
        # self.ee_pos_highs = np.array([.787, .090, .270])

        init_task_idx = 0
        self._task = None  # This is set by reset_task down below

        directions = list(range(n_tasks))
        if randomize_tasks:
            goals = self.sample_goals(n_tasks)
            if height_2d and height_2d >= 0:
                goals = [[g[0], g[1], height_2d] for g in goals]
            # goals = [1 * np.random.uniform(-1., 1., 2) for _ in directions]
        else:
            raise NotImplementedError("We don't have enough goals defined")
        self.goals = np.asarray(goals)
        print("GOALS: " + str(self.goals))
        self.tasks = [{'direction': direction} for direction in directions]

        # set the initial goal
        self.reset_task(init_task_idx)

        self.reset()

    def sample_goals(self, n_tasks):
        # Taken from: https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
        vec = np.random.randn(3, n_tasks)  # 3 dimensional sphere
        vec /= np.linalg.norm(vec, axis=0)
        vec = vec.T
        widths = (self.goal_space.high - self.goal_space.low) / 2.0
        center = self.goal_space.low + widths
        scaled_vec = vec * widths
        goals = scaled_vec + center
        return goals

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

