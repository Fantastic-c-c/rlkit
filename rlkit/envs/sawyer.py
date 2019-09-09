"""Robot Domain."""

import collections
import os
import numpy as np
import math
from gym.spaces import Box

from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_env import specs

from rlkit.envs import meshes
from rlkit.envs import textures
from rlkit.envs.mujoco_env import MujocoEnv
from . import register_env

_DEFAULT_TIME_LIMIT = 30
_CONTROL_TIMESTEP = .04

SUITE = containers.TaggedTasks()

def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'assets/sawyer_peg_insertion.xml')
    with open(filename, 'r') as f:
        data = f.read().replace('\n', '')
    assets = meshes.ASSETS
    assets.update(textures.ASSETS)
    return data, assets

def sawyer(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Test task."""
    physics = mujoco.Physics.from_xml_string(*get_model_and_assets())
    task = PegInsertion()
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        flat_observation=True, **environment_kwargs)


class PegInsertion(base.Task):
    def __init__(self, random=None):
        super(PegInsertion, self).__init__(random=random)

    def action_spec(self, physics):
        action_dim = 7
        minima = np.full(action_dim, fill_value=-1, dtype=np.float)
        maxima = np.full(action_dim, fill_value=1, dtype=np.float)
        return specs.BoundedArray(shape=(action_dim,), dtype=np.float, minimum=minima, maximum=maxima)

    def initialize_episode(self, physics):
        ''' reset to same starting pose defined by joint angles '''
        # set the reset position as defined in XML
        angles = physics.model.key_qpos[0].copy()
        velocities = np.zeros(len(physics.data.qvel))
        physics.data.qpos[:] = angles
        physics.data.qvel[:] = velocities

    def get_observation(self, physics):
        ''' get ee position and velocity from the physics '''
        obs = collections.OrderedDict()
        obs['position'] = physics.named.data.site_xpos['ee_p1'].copy()
        site_xmat = physics.named.data.site_xmat['ee_p1'].copy()
        site_xquat = np.empty(4)
        mjlib.mju_mat2Quat(site_xquat, site_xmat)
        obs['orientation'] = site_xquat
        return obs

    def get_reward(self, physics):
        ''' reward is the GPS cost function on the distance of the ee
        to the goal position '''
        # get coordinates of the points on the peg in the world frame
        # n.b. `data` coordinates are in world frame, while `model` coordinates are in local frame
        p1 = physics.named.data.site_xpos['ee_p1'].copy()
        p2 = physics.named.data.site_xpos['ee_p2'].copy()
        p3 = physics.named.data.site_xpos['ee_p3'].copy()
        stacked_peg_points = np.concatenate([p1, p2, p3])

        # get coordinates of the goal points in the world frame
        g1 = physics.named.data.site_xpos['goal_p1'].copy()
        g2 = physics.named.data.site_xpos['goal_p2'].copy()
        g3 = physics.named.data.site_xpos['goal_p3'].copy()
        stacked_goal_points = np.concatenate([g1, g2, g3])

        # compute distance between the points
        dist = np.linalg.norm(stacked_goal_points - stacked_peg_points)
        # hack to get the right scale for the desired cost fn. shape
        # the best shape is when the dist is in [-5, 5]
        dist *= 5

        # use GPS cost function: log + quadratic encourages precision near insertion
        return -(dist ** 2 + math.log10(dist ** 2 + 1e-5))


@register_env('ee-peg-insert')
class SawyerEEPegInsertionEnv(MujocoEnv):
    '''
    Top down peg insertion with 6DoF end-effector control via IK solver
    '''
    def __init__(self, max_path_length=30, n_tasks=1, randomize_tasks=False):
        self.max_path_length = 30
        self.frame_skip = 5
        self._sim_env = sawyer()
        self.initialize_camera()

        # NOTE for compatibility with code written for gym envs
        obs_dim = 7
        action_dim = 7
        self.observation_space = Box(low=-np.full(obs_dim, -np.inf), high=np.full(obs_dim, np.inf))
        self.action_space = Box(low=-np.ones(action_dim), high=np.ones(action_dim))

        self.reset()

    def reset(self):
        timestep = self._sim_env.reset()
        return timestep.observation['observations']

    def step(self, action):
        ''' apply EE pose action provided by policy '''
        #print('action', action)
        action = action.astype(np.float64)
        ik_result = qpos_from_site_pose(self._sim_env.physics, 'ee_p1', target_pos=action[:3], target_quat=action[3:])
        angles = ik_result.qpos
        #print('angles', angles)
        timestep = self._sim_env.step(angles)
        obs = timestep.observation['observations']
        reward = timestep.reward
        done = False
        return obs, reward, done, {}

    #### multi-task
    def get_all_task_idx(self):
        return [0]

    def reset_task(self, idx):
        self.reset()

    #### rendering
    def get_image(self, width=512, height=512):
        return self._sim_env.physics.render(width, height, camera_id='fixed')

    def initialize_camera(self):
        ''' set camera parameters '''
        pass


