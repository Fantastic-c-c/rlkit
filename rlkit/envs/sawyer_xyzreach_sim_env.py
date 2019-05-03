# RLKit Imports
import numpy as np
from gym import spaces
from gym import Env
from . import register_env

# Multiworld / Mujoco imports
from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

import time

class SawyerReachXYZEnv(SawyerXYZEnv, MultitaskEnv):
    def __init__(
            self,
            reward_type='hand_distance',
            norm_order=1,
            indicator_threshold=0.06,

            fix_goal=False,
            fixed_goal=(0.15, 0.6, 0.3),
            hide_goal_markers=False,

            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(self, model_name=self.model_name, **kwargs)

        self.reward_type = reward_type
        self.norm_order = norm_order
        self.indicator_threshold = indicator_threshold

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._state_goal = None
        self.hide_goal_markers = hide_goal_markers
        self.action_space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]), dtype=np.float32)
        self.hand_space = Box(self.hand_low, self.hand_high, dtype=np.float32)
        self.observation_space = Dict([
            ('observation', self.hand_space),
            ('desired_goal', self.hand_space),
            ('achieved_goal', self.hand_space),
            ('state_observation', self.hand_space),
            ('state_desired_goal', self.hand_space),
            ('state_achieved_goal', self.hand_space),
            ('proprio_observation', self.hand_space),
            ('proprio_desired_goal', self.hand_space),
            ('proprio_achieved_goal', self.hand_space),
        ])
        self.reset()

    def step(self, action):
        self.set_xyz_action(action)
        # keep gripper closed
        self.do_simulation(np.array([1]))
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        info = self._get_info()
        done = False
        return ob, reward, done, info

    def _get_obs(self):
        flat_obs = self.get_endeff_pos()
        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
            proprio_observation=flat_obs,
            proprio_desired_goal=self._state_goal,
            proprio_achieved_goal=flat_obs,
        )

    def _get_info(self):
        hand_diff = self._state_goal - self.get_endeff_pos()
        hand_distance = np.linalg.norm(hand_diff, ord=self.norm_order)
        hand_distance_l1 = np.linalg.norm(hand_diff, ord=1)
        hand_distance_l2 = np.linalg.norm(hand_diff, ord=2)
        return dict(
            hand_distance=hand_distance,
            hand_distance_l1=hand_distance_l1,
            hand_distance_l2=hand_distance_l2,
            hand_success=float(hand_distance < self.indicator_threshold),
        )

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
            goal
        )
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('hand-goal-site'), 2] = (
                -1000
            )

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_reach.xml')

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 1.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 0.3
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = 270
        self.viewer.cam.trackbodyid = -1

    def reset_model(self):
        velocities = self.data.qvel.copy()
        angles = self.data.qpos.copy()
        angles[:7] = [1.7244448, -0.92036369,  0.10234232,  2.11178144,  2.97668632, -0.38664629, 0.54065733]
        self.set_state(angles.flatten(), velocities.flatten())
        self._reset_hand()
        self.set_goal(self.sample_goal())
        self.sim.forward()
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.02]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    """
    Multitask functions
    """
    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']
        self._set_goal_marker(self._state_goal)

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        for _ in range(30):
            self.data.set_mocap_pos('mocap', state_goal)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            # keep gripper closed
            self.do_simulation(np.array([1]))

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.hand_space.low,
                self.hand_space.high,
                size=(batch_size, self.hand_space.low.size),
            )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals
        goals = desired_goals
        hand_diff = hand_pos - goals
        if self.reward_type == 'hand_distance':
            r = -np.linalg.norm(hand_diff, ord=self.norm_order, axis=1)
        elif self.reward_type == 'vectorized_hand_distance':
            r = -np.abs(hand_diff)
        elif self.reward_type == 'hand_success':
            r = -(np.linalg.norm(hand_diff, ord=self.norm_order, axis=1)
                  > self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'hand_distance_l1',
            'hand_distance_l2',
            'hand_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
        self._set_goal_marker(goal)

@register_env('sawyer-reach-sim-3d')
class PearlSawyerReachXYZEnv(SawyerReachXYZEnv):
    def __init__(self, *args, randomize_tasks=True, n_tasks=5,
                 reward_type='hand_distance',
                 action_scale=0.02,
                 hand_low=(-0.17, 0.46, 0.18), # NOTE: these coords are different from physical sawyer as (x, y) coords flipped
                 hand_high=(0.17, 0.8, 0.52),
                 **kwargs):
        self.quick_init(locals())
        SawyerReachXYZEnv.__init__(
            self,
            *args,
            action_scale=action_scale,
            hand_low=hand_low,
            hand_high=hand_high,
            **kwargs
        )
        self.observation_space = self.hand_space # now we just care about hand

        self.goal_low = np.array([-0.15, 0.48, 0.20])
        self.goal_high = np.array([0.15, 0.78, 0.50])
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

    # def _get_obs(self): # Redefine to just return state for PEARL algorithm
    #     flat_obs = self.get_endeff_pos()
    #     return flat_obs
    # def sample_goals(self, batch_size):
    #     if self.fix_goal:
    #         goals = np.repeat(
    #             self.fixed_goal.copy()[None],
    #             batch_size,
    #             0
    #         )
    #     else:
    #         goals = np.random.uniform(
    #             self.goal_space.low,
    #             self.goal_space.high,
    #             size=(batch_size, self.goal_space.low.size),
    #         )
    #     return {
    #         'desired_goal': goals,
    #         'state_desired_goal': goals,
    #     }

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

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def step(self, action):
        ob, reward, done, info = super().step(action)
        ob = ob['observation'] # just return the state
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
        angles[:7] = [1.7244448, -0.92036369,  0.10234232,  2.11178144,  2.97668632, -0.38664629, 0.54065733]
        self.set_state(angles.flatten(), velocities.flatten())
        self._reset_hand()
        # self.set_goal(self.sample_goal()) # We don't want to do this because we set our own goal
        self.sim.forward()
        return self._get_obs()['observation'] # Redefine to just return state

if __name__ == '__main__':
    env = PearlSawyerReachXYZEnv(frame_skip=1)#num_resets_before_puck_reset=int(1e6))
    for i in range(1000):
        if i % 150 == 0:
            env.reset_task(np.random.randint(0, 5))
            env.reset()
        env.step(np.asarray([0, 1, 0]))
        env.render()
        time.sleep(0.01)
