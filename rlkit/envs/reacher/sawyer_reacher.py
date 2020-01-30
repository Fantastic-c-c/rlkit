from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box
import gym
from gym.envs.mujoco import mujoco_env
import os

from rlkit.envs import register_env

SCRIPT_DIR = os.path.dirname(__file__)

##################################################################
##################################################################
##################################################################
##################################################################


class SawyerReachingEnv(mujoco_env.MujocoEnv):

    '''
    Reaching to a desired end-effector position while controlling the 7 joints of sawyer
    '''

    def __init__(self, xml_path=None, goal_site_name=None, sparse_reward=False, action_mode='joint_position', obs_mode='state'):

        # starting positions
        self.start_positions = []
        self.start_positions.append(np.array([-0.5795, -0.95, 0.03043, 1.3698, 1.66656, 0.02976, -0.37696]))
        self.start_positions.append(np.array([0.7625, -0.77, 0.39559, 0.761, 0.11904, -0.47616, 0.04712]))
        self.start_positions.append(np.array([0.4575, -0.23, 0.12172, 0.63924, -0.23808, 0.05952, 0.04712]))
        self.start_positions.append(np.array([-0.0305, -0.92, 1.18677, 1.33936, -0.4464, 0.38688, 0.37696]))
        self.start_positions.append(np.array([-0.3355, -0.59, 1.33892, 1.09584, -1.16064, 1.3392, 0.37696]))

        self.action_mode = action_mode
        self.obs_mode = obs_mode
        self.image_dim = 64
        self.num_joint_dof = 7
        self.frame_skip = 100
        if xml_path is None:
            xml_path = os.path.join(SCRIPT_DIR, 'assets/sawyer_reach.xml')
        if goal_site_name is None:
            goal_site_name = 'goal_reach_site'
        self.body_id_ee = 0
        self.site_id_ee = 0
        self.site_id_goal = 0
        self.is_eval_env = False

        # Sparse reward setting
        self.sparse_reward = sparse_reward
        self.truncation_dist = 0.3
        # if distance from goal larger than this,
        # get dist(self.truncation_dis) reward every time steps.
        # The dist is around 1 in starting pos

        # create the env
        self.startup = True
        mujoco_env.MujocoEnv.__init__(self, xml_path, self.frame_skip) # self.model.opt.timestep is 0.0025 (w/o frameskip)
        self.startup = False

        # initial position of joints
        self.init_qpos = self.sim.model.key_qpos[0].copy()
        self.init_qvel = np.zeros(len(self.data.qvel))

        # joint limits
        self.limits_lows_joint_pos = self.model.actuator_ctrlrange.copy()[:, 0]
        self.limits_highs_joint_pos = self.model.actuator_ctrlrange.copy()[:, 1]

        # set the action space (always between -1 and 1 for this env)
        self.action_highs = np.ones((self.num_joint_dof,))
        self.action_lows = -1*np.ones((self.num_joint_dof,))
        self.action_space = Box(low=self.action_lows, high=self.action_highs)

        # set the observation space
        obs_size = self.get_obs_dim()
        self.observation_space = Box(low=-np.ones(obs_size) * np.inf, high=np.ones(obs_size) * np.inf)

        # vel limits
        joint_vel_lim = 0.04 ##### 0.07 # magnitude of movement allowed within a dt [deg/dt]
        self.limits_lows_joint_vel = -np.array([joint_vel_lim]*self.num_joint_dof)
        self.limits_highs_joint_vel = np.array([joint_vel_lim]*self.num_joint_dof)

        # ranges
        self.action_range = self.action_highs-self.action_lows
        self.joint_pos_range = (self.limits_highs_joint_pos - self.limits_lows_joint_pos)
        self.joint_vel_range = (self.limits_highs_joint_vel - self.limits_lows_joint_vel)

        # ids
        self.body_id_ee = self.model.body_names.index('end_effector')
        self.site_id_ee = self.model.site_name2id('ee_site')
        self.site_id_goal = self.model.site_name2id(goal_site_name)

    def override_action_mode(self, action_mode):
        self.action_mode = action_mode

    def override_reward_mode(self, is_sparse_reward):
        self.sparse_reward = is_sparse_reward

    def get_obs_dim(self):
        return len(self.get_obs())

    def get_image(self, width=64, height=64, camera_name='track'):
        # use sim.render to avoid MJViewer which doesn't seem to work without display
        img = self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )
        return np.flipud(img)

    def get_obs(self, obs_mode=None):
        ''' state observation is joint angles + joint velocities + ee pose '''
        if obs_mode is None:
            obs_mode = self.obs_mode
        if obs_mode == 'image':
            img = self.get_image()
            img = img.astype(np.float32) / 255.
            return img.flatten()
        else:
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
        return self.data.body_xpos[self.body_id_ee].copy()

    def reset_model(self):

        #### FIXED START
        # angles = self.init_qpos.copy()

        #### RAND START
        angles_idx = np.random.randint(len(self.start_positions))
        angles = self.start_positions[angles_idx]

        velocities = self.init_qvel.copy()
        self.set_state(angles, velocities) #this sets qpos and qvel + calls sim.forward

        return self.get_obs()

    def do_step(self, action):

        if self.startup:
            feasible_desired_position = 0*action
        else:
            # clip to action limits
            action = np.clip(action, self.action_lows, self.action_highs)

            # get current position
            curr_position = self._get_joint_angles()

            if self.action_mode=='joint_position':
                # scale incoming (-1,1) to self.joint_limits
                desired_position = (((action - self.action_lows) * self.joint_pos_range) / self.action_range) + self.limits_lows_joint_pos

                # make the
                feasible_desired_position = self.make_feasible(curr_position, desired_position)

            elif self.action_mode=='joint_delta_position':
                # scale incoming (-1,1) to self.vel_limits
                desired_delta_position = (((action - self.action_lows) * self.joint_vel_range) / self.action_range) + self.limits_lows_joint_vel

                # add delta
                feasible_desired_position = curr_position + desired_delta_position

        self.do_simulation(feasible_desired_position, self.frame_skip)

    def step(self, action):
        ''' apply the 7DoF action provided by the policy '''

        self.do_step(action)
        obs = self.get_obs()
        reward, score, sparse_reward = self.compute_reward(get_score=True)
        done = False
        info = dict(score=score)
        return obs, reward, done, info

    def make_feasible(self, curr_position, desired_position):

        # compare the implied vel to the max vel allowed
        max_vel = self.limits_highs_joint_vel
        implied_vel = np.abs(desired_position-curr_position)

        # limit the vel
        actual_vel = np.min([implied_vel, max_vel], axis=0)

        # find the actual position, based on this vel
        sign = np.sign(desired_position-curr_position)
        actual_difference = sign*actual_vel
        feasible_position = curr_position+actual_difference

        return feasible_position

    def compute_reward(self, get_score=False, goal_id_override=None):

        # get goal id
        if goal_id_override is None:
            goal_id = self.site_id_goal
        else:
            goal_id = goal_id_override

        # get coordinates of the sites in the world frame
        ee_xyz = self.data.site_xpos[self.site_id_ee].copy()
        goal_xyz = self.data.site_xpos[goal_id].copy()

        # score
        score = -np.linalg.norm(ee_xyz - goal_xyz)

        # distance
        dist = 5*np.linalg.norm(ee_xyz - goal_xyz)
        sparse_dist = min(dist, self.truncation_dist) # if dist too large: return the reward at truncate_dist

        # dense reward
        # use GPS cost function: log + quadratic encourages precision near insertion
        reward = -(dist ** 2 + math.log10(dist ** 2 + 1e-5))

        # sparse reward
        # offset the whole reward such that when dist>truncation_dist, the reward will be exactly 0
        sparse_reward = -(sparse_dist ** 2 + math.log10(sparse_dist ** 2 + 1e-5))
        sparse_reward = sparse_reward - (-(self.truncation_dist ** 2 + math.log10(self.truncation_dist ** 2 + 1e-5)))

        # case of only wanting sparse reward
        if self.sparse_reward:
            reward = sparse_reward

        if get_score:
        	return reward, score, sparse_reward
        else:
        	return reward

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

    def reset(self):

        # reset task (this is a single-task case)
        self.model.site_pos[self.site_id_goal] = np.array([0.7,0.2,0.4])

        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def goal_visibility(self, visible):

        ''' Toggle the goal visibility when rendering: video should see goal, but image obs shouldn't '''

        if visible:
            self.model.site_rgba[self.site_id_goal] = np.array([1, 0, 0, 1])
        else:
            self.model.site_rgba[self.site_id_goal] = np.array([1, 0, 0, 0])

##################################################################
##################################################################
##################################################################
##################################################################

@register_env('reacher')
class SawyerReachingEnvMultitask(SawyerReachingEnv):

    '''
    This env is the multi-task version of reaching. The reward always gets concatenated to obs.
    '''

    def __init__(self, xml_path=None, goal_site_name=None, sparse_reward=False, action_mode='joint_position', obs_mode='state'):

        # goal range for reaching
        self.goal_range = Box(low=np.array([0.85, -0.3, 0.45]), high=np.array([0.9, 0.3, 0.65]))

        if goal_site_name is None:
            goal_site_name = 'goal_reach_site'

        super(SawyerReachingEnvMultitask, self).__init__(xml_path=xml_path, goal_site_name=goal_site_name, sparse_reward=sparse_reward, action_mode=action_mode, obs_mode=obs_mode)

    def get_obs_dim(self):
        return len(self.get_obs())

    def step(self, action):

        self.do_step(action)
        obs = self.get_obs()
        reward, score, sparse_reward = self.compute_reward(get_score=True)
        done = False
        info = dict(score=score)

        return (obs, reward, done, info)

    def reset(self):

        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()

        return ob

    ##########################################
    ### These are called externally
    ##########################################

    def init_tasks(self, num_tasks, is_eval_env):

        '''
        Call this function externally, ONCE
        to define this env as either train env or test env
        and get the possible task list accordingly
        '''

        if is_eval_env:
            np.random.seed(100) #pick eval tasks as random from diff seed
        else:
            np.random.seed(101)

        possible_goals = [self.goal_range.sample() for _ in range(num_tasks)]
        return possible_goals


    def set_task_for_env(self, goal):

        '''
        Call this function externally,
        to reset the task
        '''

        # task definition = set the goal reacher location to be the given goal
        self.model.site_pos[self.site_id_goal] = goal.copy()

    ##########################################
    ##########################################
