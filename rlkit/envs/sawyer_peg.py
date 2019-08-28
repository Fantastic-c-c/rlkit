from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box

from rlkit.envs.mujoco_env import MujocoEnv
from . import register_env


@register_env('peg-insert')
class SawyerPegInsertionEnv(MujocoEnv):
    '''
    Top down peg insertion with 7DoF joint position control
    '''
    def __init__(self, max_path_length=30, n_tasks=1, randomize_tasks=False):
        self.max_path_length = max_path_length
        self.frame_skip = 5
        self.action_scale = 100

        xml_path = 'sawyer_peg_insertion.xml'
        self.prev_qpos = None
        super(SawyerPegInsertionEnv, self).__init__(
                xml_path,
                frame_skip=self.frame_skip, # sim rate / control rate ratio
                automatically_set_obs_and_action_space=True)
        # set the reset position as defined in XML
        self.init_qpos = self.sim.model.key_qpos[0].copy()
        self.prev_qpos = self.init_qpos.copy()

        # set the action space to be -1, 1
        self.action_space = Box(low=-np.ones(7), high=np.ones(7))
        obs_size = len(self.get_obs())
        self.observation_space = Box(low=-np.ones(obs_size) * 100, high=np.ones(obs_size) * 100)

        # TODO multitask stuff
        self._goal = self.data.site_xpos[self.model.site_name2id('goal_p1')].copy()

        self.reset()

    def get_obs(self):
        ''' state observation is joint angles + joint velocities + ee pose '''
        angles = self._get_joint_angles()
        velocities = self._get_joint_velocities()
        ee_pose = self._get_ee_pose()
        return np.concatenate([angles, velocities, ee_pose])

    def _get_joint_angles(self):
        return self.data.qpos.copy()

    def _get_joint_velocities(self):
        return self.data.qvel.copy()

    def _get_ee_pose(self):
        ''' ee pose is xyz position + orientation quaternion '''
        # TODO this is only position right now!
        ee_id = self.model.body_names.index('end_effector')
        return self.data.body_xpos[ee_id].copy()

    def reset(self):
        ''' reset to the same starting pose defined by joint angles '''
        angles = self.init_qpos
        velocities = np.zeros(len(self.data.qvel))
        self.set_state(angles, velocities)
        self.sim.forward()
        return self.get_obs()

    def step(self, action):
        ''' apply the 7DoF action provided by the policy '''
        torques = action * self.action_scale
        # for now, the sim rate is 5 times the control rate
        self.do_simulation(torques, self.frame_skip)
        obs = self.get_obs()
        reward = self.compute_reward()
        done = False
        return obs, reward, done, {}

    def compute_reward(self):
        ''' reward is the GPS cost function on the distance of the ee
        to the goal position '''
        # get coordinates of the points on the peg in the world frame
        # n.b. `data` coordinates are in world frame, while `model` coordinates are in local frame
        p1 = self.data.site_xpos[self.model.site_name2id('ee_p1')].copy()
        p2 = self.data.site_xpos[self.model.site_name2id('ee_p2')].copy()
        p3 = self.data.site_xpos[self.model.site_name2id('ee_p3')].copy()
        stacked_peg_points = np.concatenate([p1, p2, p3])

        # get coordinates of the goal points in the world frame
        g1 = self.data.site_xpos[self.model.site_name2id('goal_p1')].copy()
        g2 = self.data.site_xpos[self.model.site_name2id('goal_p2')].copy()
        g3 = self.data.site_xpos[self.model.site_name2id('goal_p3')].copy()
        stacked_goal_points = np.concatenate([g1, g2, g3])

        # compute distance between the points
        dist = np.linalg.norm(stacked_goal_points - stacked_peg_points)
        # hack to get the right scale for the desired cost fn. shape
        # the best shape is when the dist is in [-5, 5]
        dist *= 3

        # use GPS cost function: log + quadratic encourages precision near insertion
        return -(dist ** 2 + math.log10(dist ** 2 + 1e-5))

    def viewer_setup(self):
        # side view
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.4
        self.viewer.cam.lookat[1] = 0.75
        self.viewer.cam.lookat[2] = 0.4
        self.viewer.cam.distance = 0.2
        self.viewer.cam.elevation = -55
        self.viewer.cam.azimuth = 180
        self.viewer.cam.trackbodyid = -1

    def get_all_task_idx(self):
        return [0]

    def reset_task(self, idx):
        self.reset()
