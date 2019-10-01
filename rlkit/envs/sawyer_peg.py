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

# do not think this does anything...
_DEFAULT_TIME_LIMIT = 30

SUITE = containers.TaggedTasks()

def get_model_and_assets(path):
    """Returns a tuple containing the model XML string and a dict of assets."""
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, path)
    with open(filename, 'r') as f:
        data = f.read().replace('\n', '')
    assets = meshes.ASSETS
    assets.update(textures.ASSETS)
    return data, assets

@register_env('ee-peg-insert')
def sawyer_ee_peg_insertion(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    ''' sawyer robot, end-effector control, peg insertion task '''
    physics = EEPositionController.from_xml_string(*get_model_and_assets('assets/sawyer_ee_peg_insertion.xml'))
    task = PegInsertion()
    environment_kwargs = environment_kwargs or {}
    return SawyerPegInsertionEnv(
        physics, task, time_limit=time_limit, flat_observation=True,
        **environment_kwargs)

@register_env('torque-peg-insert')
def sawyer_torque_peg_insertion(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    ''' sawyer robot, joint torque control, peg insertion task '''
    physics = JointTorqueController.from_xml_string(*get_model_and_assets('assets/sawyer_torque_peg_insertion.xml'))
    task = PegInsertion()
    environment_kwargs = environment_kwargs or {}
    return SawyerPegInsertionEnv(
        physics, task, time_limit=time_limit, flat_observation=True,
        **environment_kwargs)


class EEPositionController(mujoco.Physics):
    '''
    command the robot with end-effector positions and quaternions

    - use inverse kinematics (IK) to convert ee pose into joint angles
    - joint angles are fed to a PD controller defined in the XML that
    produces torques
    - bias exploration with a safety box that constrains possible end-effector
    positions
    '''

    def __init__(self, *args, **kwargs):
        self.include_vel_in_state = True
        super(EEPositionController, self).__init__(*args, **kwargs)

    def get_observation(self):
        ''' obs consists of ee pose and optionally velocity '''
        obs = collections.OrderedDict()
        obs['position'] = self.named.data.site_xpos['ee_p1'].copy()
        site_xmat = self.named.data.site_xmat['ee_p1'].copy()
        site_xquat = np.empty(4)
        mjlib.mju_mat2Quat(site_xquat, site_xmat)
        obs['orientation'] = site_xquat
        if self.include_vel_in_state:
            # TODO get velocity by using previous xpos and dt
            raise NotImplementedError
        return obs

    def get_observation_space(self):
        ''' get obs bounds that will be used to normalize observations '''
        obs_dim = 7
        if self.include_vel_in_state:
            obs_dim = 14
        return Box(low=-np.full(obs_dim, -np.inf), high=np.full(obs_dim, np.inf))

    def get_action_space(self):
        ''' get action bounds that will be used to normalize actions '''
        action_dim = 7
        safety_box = self.safety_box()
        pos_low = safety_box.low
        pos_high = safety_box.high
        return Box(low=np.concatenate([pos_low, -np.ones(4)]), high=np.concatenate([pos_high, np.ones(4)]))

    def safety_box(self):
        ''' define a safety box to contain the end-effector '''
        reset_xpos = self.named.data.site_xpos['ee_p1'].copy()
        low = reset_xpos - np.array([.3, .3, .2])
        high = reset_xpos + np.array([.3, .3, .01])
        safety_box = Box(low=low, high=high)
        return safety_box

    def prepare_action(self, action, init_obs):
        ''' convert ee pose into joint angles for control '''
        # compute the action to apply using IK
        action = action.astype(np.float64) # required by the IK for some reason
        # TODO debug, hard-code the orientation
        action[3:] = init_obs.copy()[3:]
        ik_result = qpos_from_site_pose(self, 'ee_p1', target_pos=action[:3], target_quat=action[3:])
        angles = ik_result.qpos
        return angles

class JointTorqueController(mujoco.Physics):
    '''
    command the robot with raw joint torques

    - use torque limits defined in XML to constrain applied forces
    '''
    def __init__(self, *args, **kwargs):
        self.include_ee_in_state = True
        super(JointTorqueController, self).__init__(*args, **kwargs)

    def get_observation(self):
        obs = collections.OrderedDict()
        obs['angles'] = self.data.qpos.copy()
        obs['velocities'] = self.data.qvel.copy()
        if self.include_ee_in_state:
            obs['ee_position'] = self.named.data.site_xpos['ee_p1'].copy()
            site_xmat = self.named.data.site_xmat['ee_p1'].copy()
            site_xquat = np.empty(4)
            mjlib.mju_mat2Quat(site_xquat, site_xmat)
            obs['ee_orientation'] = site_xquat
        return obs

    def get_observation_space(self):
        ''' get obs bounds that will be used to normalize observations '''
        obs_dim = 14
        if self.include_ee_in_state:
            obs_dim = 21
        return Box(low=-np.full(obs_dim, -np.inf), high=np.full(obs_dim, np.inf))

    def get_action_space(self):
        ''' get action bounds that will be used to normalize actions '''
        bounds = self.model.actuator_ctrlrange
        return Box(low=bounds[:, 0], high=bounds[:, 1])

    def prepare_action(self, action, init_obs):
        ''' raw torques are ready to feed to the simulator '''
        return action


class PegInsertion(base.Task):
    '''
    define the peg insertion task

    by the dm_control API, this method must define:
     - action_spec() - note this isn't used as our actions are already
       in (-1, 1) and we have our own NormalizedEnv wrapper to rescale
     - get_observation()
     - initialize_episode()
    '''
    def __init__(self, random=None):
        super(PegInsertion, self).__init__(random=random)

    def action_spec(self, physics):
        ''' action spec is for joint angles NOT ee pose '''
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
        return physics.get_observation()

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


class SawyerPegInsertionEnv(Environment):
    '''
    wrap the dm_control Environment to interface with PEARL codebase
    '''
    def __init__(self, physics, task, max_path_length=30, n_tasks=1, randomize_tasks=False, **kwargs):
        super(SawyerPegInsertionEnv, self).__init__(physics, task, **kwargs)
        print('num sim steps per control step', self._n_sub_steps)
        self.frame_rate =  1 / kwargs['control_timestep']
        self.max_path_length = 30

        # NOTE this is a hack needed because the safety box used to define
        # the action space uses the xpos of the reset position!!
        _ = self.reset()

        # set obs and action spaces appropriately for the controller
        # NOTE PEARL codebase uses action and obs space to normalize so
        # these must be correct
        self.observation_space = self.physics.get_observation_space()
        self.action_space = self.physics.get_action_space()

        # multitask stuff
        self._goal = self.physics.named.data.site_xpos['goal_p1'].copy()

        self.init_obs = self.reset()

    def reset(self):
        ''' reset to same initial pose '''
        timestep = super().reset()
        return timestep.observation['observations']

    def step(self, action):
        ''' apply action provided by policy '''
        angles = self.physics.prepare_action(action, self.init_obs)
        timestep = super().step(angles)
        obs = timestep.observation['observations']
        reward = timestep.reward
        done = False
        return obs, reward, done, {}

    def get_all_task_idx(self):
        return [0]

    def reset_task(self, idx):
        self.reset()

    def get_image(self, width=512, height=512):
       im = Image.fromarray(self.physics.render(width, height, camera_id='sideview'))
       # rotation is a hack to compensate for the fixed camera defined in XML
       return im.rotate(270)

