# RLKit Imports
import numpy as np
from gym import spaces
from gym import Env
from . import register_env

# Multiworld / Mujoco imports
from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYEnv

import time


@register_env('sawyer-reach-sim-2d')
class PearlSawyerReachXYZEnv(SawyerReachXYEnv):
    def __init__(self, *args, randomize_tasks=True, n_tasks=5,
                 reward_type='hand_distance',
                 action_scale=0.02,
                 hand_z_position=0.35,
                 hand_low=(-0.17, 0.46, 0.18),
                 # NOTE: these coords are different from physical sawyer as (x, y) coords flipped
                 hand_high=(0.17, 0.8, 0.52),
                 **kwargs):
        self.quick_init(locals())
        SawyerReachXYEnv.__init__(
            self,
            *args,
            action_scale=action_scale,
            hand_low=hand_low,
            hand_high=hand_high,
            **kwargs
        )
        self.observation_space = self.hand_space  # now we just care about hand

        self.goal_low = np.array([-0.15, 0.48, self.hand_z_position])
        self.goal_high = np.array([0.15, 0.78, self.hand_z_position])
        self.goal_space = Box(self.goal_low, self.goal_high, dtype=np.float32)
        init_task_idx = 0

        directions = list(range(n_tasks))
        if randomize_tasks:
            goals = self.sample_goals(n_tasks)
            # goals = [1 * np.random.uniform(-1., 1., 2) for _ in directions]
        else:
            # add more goals in n_tasks > 7
            goals = [
            ]
            if (n_tasks > len(goals)):
                raise NotImplementedError("We don't have enough goals defined")
        self.goals = np.asarray(goals)
        self.tasks = [{'direction': direction} for direction in directions]

        # set the initial goal
        self.reset_task(init_task_idx)

        self.reset()

    def sample_goals(self, n_tasks):
        # Taken from: https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
        n_dim = 2
        vec = np.random.randn(n_dim, n_tasks)  # 2 dimensional circle
        vec /= np.linalg.norm(vec, axis=0)
        vec = vec.T
        vec = np.append(vec, 0, axis=1)

        widths = (self.goal_space.high - self.goal_space.low) / 2.0  # width of each dimension
        center = self.goal_space.low + widths
        scaled_vec = vec * widths
        goals = scaled_vec + center
        return goals

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def step(self, action):
        ob, reward, done, info = super().step(action)
        ob = ob['observation']  # just return the state
        return ob, reward, done, info

    def get_all_goals(self):
        return self.goals

    def reset_goal(self, direction):
        return self.goals[direction]

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self.reset_goal(self._task['direction'])
        goal_dict = {'desired_goal': None, 'state_desired_goal': self._goal}

        self.set_goal(goal_dict)
        self.reset()

    def reset(self):
        return self.reset_model()

    def _reset_hand(self):
        if hasattr(self, "goal_space"):
            widths = (self.goal_space.high - self.goal_space.low) / 2.0
            center = self.goal_space.low + widths
        else:
            center = np.array([0, 0.5, 0.02])
        for _ in range(10):
            self.data.set_mocap_pos('mocap', center)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 0, 0]))
            self.do_simulation(None, self.frame_skip)

    def reset_model(self):
        velocities = self.data.qvel.copy()
        angles = self.data.qpos.copy()
        angles[:7] = [1.7244448, -0.92036369, 0.10234232, 2.11178144, 2.97668632, -0.38664629, 0.54065733]
        self.set_state(angles.flatten(), velocities.flatten())
        self._reset_hand()
        # self.set_goal(self.sample_goal()) # We don't want to do this because we set our own goal
        self.sim.forward()
        return self._get_obs()['observation']  # Redefine to just return state


if __name__ == '__main__':
    env = PearlSawyerReachXYEnv(frame_skip=1)  # num_resets_before_puck_reset=int(1e6))
    for i in range(1000):
        if i % 150 == 0:
            env.reset_task(np.random.randint(0, 5))
            env.reset()
        env.step(np.asarray([0, 1, 0]))
        env.render()
        time.sleep(0.01)
