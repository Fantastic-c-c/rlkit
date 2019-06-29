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
class PearlSawyerReachXYSimEnv(SawyerReachXYEnv):
    def __init__(self, *args, randomize_tasks=True, n_tasks=5,
                 reward_type='hand_distance',
                 use_mocap=False,
                 action_scale=0.02,
                 hand_z_position=0.38,
                 hand_low=(-0.17, 0.46, 0.21),
                 # NOTE: these coords are different from physical sawyer as (x, y) coords flipped
                 hand_high=(0.17, 0.8, 0.55),
                 **kwargs):
        self.quick_init(locals())
        # NOTE: when true, prevents reset() from calling render()
        # to return an image observation, inadvertently also setting
        # camera parameters to default, which we do not want! Set to
        # False after __init__() is done.
        self.init = True
        SawyerReachXYEnv.__init__(
            self,
            *args,
            action_scale=action_scale,
            hand_low=hand_low,
            hand_high=hand_high,
            hand_z_position=hand_z_position,
            **kwargs
        )
        self.observation_space = self.hand_space  # now we just care about hand
        self.goal_low = np.array([0.05, 0.55, self.hand_z_position])
        self.goal_high = np.array([0.15, 0.78, self.hand_z_position])
        self.goal_space = Box(self.goal_low, self.goal_high, dtype=np.float32)
        init_task_idx = 0

        directions = list(range(n_tasks))
        if randomize_tasks:
            #goals = self.sample_goals(n_tasks)
            goals = [1 * np.random.uniform(self.goal_low, self.goal_high) for _ in directions]
        else:
            # add more goals in n_tasks > 7
            goals = [
            ]
            if (n_tasks > len(goals)):
                raise NotImplementedError("We don't have enough goals defined")
        self.goals = np.asarray(goals)
        self.tasks = [{'direction': direction} for direction in directions]

        self.image_dim = 84 # new, dim of image observation, set in mujoco_env.py

        # set the initial goal
        def cam_init(x):
            # TODO: other cam params here
            x.type = 0
            x.elevation = -20
            x.distance = self.model.stat.extent * 1.0
            x.azimuth = 250
        self.initialize_camera(cam_init)
        self.reset_task(init_task_idx
)
        self.init = False

    def get_goal(self):
        return self._state_goal

    def set_goal(self, goal):
        self._state_goal = goal
        self._set_goal_marker(self._state_goal)

    def set_to_goal(self, goal):
        state_goal = goal
        for _ in range(30):
            self.data.set_mocap_pos('mocap', state_goal)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            # keep gripper closed
            self.do_simulation(np.array([1]))


    def sample_goals(self, n_tasks):
        # Taken from: https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
        n_dim = 2
        np.random.seed(1337)
        vec = np.random.randn(n_dim, n_tasks)  # 2 dimensional circle
        vec /= np.linalg.norm(vec, axis=0)
        vec = vec.T
        vec = np.append(vec, np.zeros((n_tasks, 1)), axis=1)

        widths = (self.goal_space.high - self.goal_space.low) / 2.0  # width of each dimension
        center = self.goal_space.low + widths
        scaled_vec = vec * widths
        goals = scaled_vec + center
        print('goals', goals)
        return goals

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def step(self, action):
        ob, reward, done, info = super().step(action)
        image = self.get_image()
        ob = np.moveaxis(image, 2, 0)
        # ob = ob['observation']  # just return the state
        return ob, reward, done, info

    def debug(self, action):
        ob, reward, done, info = super().step(action)
        realstate = ob['observation']  # just return the state
        image = self.get_image()
        ob = np.moveaxis(image, 2, 0)
        # ob = np.moveaxis(image, 2, 0)

        return realstate, ob

    def get_all_goals(self):
        return self.goals

    def reset_goal(self, direction):
        return self.goals[direction]

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self.reset_goal(self._task['direction'])

        self.set_goal(self._goal)
        self.reset()

    def reset(self):
        if self.init:
            return None
        return self.reset_model()

    def _reset_hand(self):
        # 2-D reaching, so start with hand at same height as goals
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.38]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    def reset_model(self):
        velocities = self.data.qvel.copy()
        angles = self.data.qpos.copy()
        angles[:7] = [1.7244448, -0.92036369,  0.10234232,  2.11178144,  2.97668632, -0.38664629, 0.54065733]
        # angles[:7] = [0.165470703125, -0.785970703125, -0.7146943359375, 1.50058203125, 0.557935546875, 1.0604775390625,
        #               -2.0759697265625]
        self.set_state(angles.flatten(), velocities.flatten())
        self._reset_hand()
        # self.set_goal(self.sample_goal()) # We don't want to do this because we set our own goal
        self.sim.forward()

        #when reset model, also need to return an image observation
        image = self.get_image()
        ob = np.moveaxis(image, 2, 0)

        return ob
        # return self._get_obs()['observation']  # Redefine to just return state #original


if __name__ == '__main__':
    env = PearlSawyerReachXYSimEnv()  # num_resets_before_puck_reset=int(1e6))

    path = 15
    for i in range(3*path + 1):
        if i % path == 0:
            print("~~~~~~~~~~~~~~")
            print("DIFF: {}".format(env.data.mocap_pos - env._get_obs()['observation']))
            print("POS: {} | {}".format(env._get_obs()['observation'], env.data.mocap_pos))
            env.reset_task(np.random.randint(0, 5))
        curr = env.get_endeff_pos()
        env.step(np.asarray([1, 0]))
        print("{} DELTA: {}".format(i, env.get_endeff_pos() - curr))
        env.render()
