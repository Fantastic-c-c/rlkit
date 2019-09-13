"""Robot Domain."""

import collections
import os
import numpy as np
import math
from gym.spaces import Box
from PIL import Image

from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control import mujoco
from dm_control.rl.control import Environment
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
# NOTE this makes the frame skip 16
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

@register_env('ee-peg-insert')
def sawyer_peg_insertion(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    ''' create Environment containing task and physics '''
    physics = mujoco.Physics.from_xml_string(*get_model_and_assets())
    task = PegInsertion()
    environment_kwargs = environment_kwargs or {}
    return SawyerEEPegInsertionEnv(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        flat_observation=True, **environment_kwargs)


class PegInsertion(base.Task):
    def __init__(self, random=None):
        super(PegInsertion, self).__init__(random=random)

    def action_spec(self, physics):
        ''' action spec is for joint angles NOT ee pose '''
        action_dim = 7
        minima = np.full(action_dim, fill_value=-1, dtype=np.float)
        maxima = np.full(action_dim, fill_value=1, dtype=np.float)
        return specs.BoundedArray(shape=(action_dim,), dtype=np.float, minimum=minima, maximum=maxima)

    def safety_box(self, physics):
        ''' define a safety box to contain the end-effector '''
        reset_xpos = physics.named.data.site_xpos['ee_p1'].copy()
        print('reset xpos', reset_xpos)
        low = reset_xpos - np.array([.3, .3, .2])
        high = reset_xpos + np.array([.3, .3, .01])
        safety_box = Box(low=low, high=high)
        return safety_box

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
        # TODO does not include velocity
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


class SawyerEEPegInsertionEnv(Environment):
    '''
    Top down peg insertion with 6DoF end-effector control via IK solver

    wrap the dm_control Environment to compute IK in the step function and
    include features needed for PEARL codebase
    '''
    def __init__(self, physics, task, max_path_length=30, n_tasks=1, randomize_tasks=False, **kwargs):
        # TODO compute time_limit from max_path_length
        super(SawyerEEPegInsertionEnv, self).__init__(physics, task, **kwargs)
        self.max_path_length = 30


        # NOTE this is needed because the safety box used to define the action space
        # uses the xpos of the reset position!!
        _ = self.reset()

        # NOTE PEARL codebase uses action and obs space to normalize so these must be correct
        obs_dim = 7
        action_dim = 7
        self.observation_space = Box(low=-np.full(obs_dim, -np.inf), high=np.full(obs_dim, np.inf))
        safety_box = self.task.safety_box(self.physics)
        pos_low = safety_box.low
        pos_high = safety_box.high
        self.action_space = Box(low=np.concatenate([pos_low, -np.ones(4)]), high=np.concatenate([pos_high, np.ones(4)]))
        print('action space', self.action_space.low, self.action_space.high)

        # multitask stuff
        self._goal = self.physics.named.data.site_xpos['goal_p1'].copy()

        self.init_obs = self.reset()

    def reset(self):
        ''' reset to same initial pose '''
        timestep = super().reset()
        return timestep.observation['observations']

    def step(self, action):
        ''' apply EE pose action provided by policy '''
        # compute the action to apply using IK
        action = action.astype(np.float64) # required by the IK for some reason
        # TODO debug, hard-code the orientation
        action[3:] = self.init_obs.copy()[3:]
        ik_result = qpos_from_site_pose(self.physics, 'ee_p1', target_pos=action[:3], target_quat=action[3:])
        angles = ik_result.qpos

        # apply the action and step the sim
        timestep = super().step(angles)
        obs = timestep.observation['observations']
        reward = timestep.reward
        done = False
        return obs, reward, done, {}

    #### everything below here for PEARL code

    # multi-task
    def get_all_task_idx(self):
        return [0]

    def reset_task(self, idx):
        self.reset()

    # rendering
    def get_image(self, width=512, height=512):
       im = Image.fromarray(self.physics.render(width, height, camera_id='sideview'))
       return im.rotate(270)

