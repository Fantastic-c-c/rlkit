from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box
import gym
from gym.envs.mujoco import mujoco_env
import os
from gym.envs.registration import register
SCRIPT_DIR = os.path.dirname(__file__)
from rlkit.envs import register_env
##################################################################
##################################################################
##################################################################
##################################################################

@register_env('SawyerReach-v0')
class SawyerReachingEnv(mujoco_env.MujocoEnv):

    '''
    Reaching to a desired end-effector position while controlling the 7 joints of sawyer
    '''

    def __init__(self, xml_path=None, goal_site_name=None, sparse_reward=False, action_mode='joint_position'):

    	# vars


        self.image_dim = 64  ###########new: size of image obs


        self.action_mode = action_mode
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
        joint_vel_lim = 0.07 # magnitude of movement allowed within a dt [deg/dt]
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

    def get_obs(self):
        ''' state observation is joint angles + joint velocities + ee pose '''

        ###############original##################
        # angles = self._get_joint_angles()
        # velocities = self._get_joint_velocities()
        # ee_pose = self._get_ee_pose()
        # return np.concatenate([angles, velocities, ee_pose])

        ###############new: now it's returning the image observation
        image = self.sim.render(self.image_dim, self.image_dim, camera_name="track")


        ################new: to visualize the image, use img.show() #############
        # from PIL import Image
        # import numpy as np
        # img = Image.fromarray(image)
        # import pdb; pdb.set_trace()
        ######################################################################

        ob = np.moveaxis(image, 2, 0)
        return ob


    def _get_joint_angles(self):
        return self.data.qpos.copy()

    def _get_joint_velocities(self):
        return self.data.qvel.copy()

    def _get_ee_pose(self):
        ''' ee pose is xyz position + orientation quaternion '''
        return self.data.body_xpos[self.body_id_ee].copy()

    def reset_model(self):

        angles = self.init_qpos.copy()
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
        obs = self.get_obs()   # will change the get_obs function so that it returns the image directly
        reward, score = self.compute_reward(get_score=True)
        done = False
        info = np.array([score, 0, 0, 0, 0])  # can populate with more info, as desired, for tb logging
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

        # distance between the points
        score = np.linalg.norm(ee_xyz - goal_xyz)
        dist = 5*score

        # Sparse reward setting
        if self.sparse_reward:
            dist = min(dist, self.truncation_dist) # if dist too large: return the reward at truncate_dist

        # use GPS cost function: log + quadratic encourages precision near insertion
        reward = -(dist ** 2 + math.log10(dist ** 2 + 1e-5))

        if self.sparse_reward:
            # offset the whole reward such that when dist>truncation_dist, the reward will be exactly 0
            reward = reward - (-(self.truncation_dist ** 2 + math.log10(self.truncation_dist ** 2 + 1e-5)))

        if get_score:
        	return reward, score
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
        self.model.site_pos[self.site_id_goal] = np.array([0.7,0.2,0.4])        # the single goal is fixed in here

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

@register_env('SawyerReachMT-v0')
class SawyerReachingEnvMultitask(SawyerReachingEnv):

    '''
    This env is the multi-task version of reaching. The reward always gets concatenated to obs.
    '''

    def __init__(self, xml_path=None, goal_site_name=None, sparse_reward=False, action_mode='joint_position'):

        # goal range for reaching
        self.goal_range = Box(low=np.array([0.7, -0.5, 0.1]), high=np.array([0.8, 0.5, 0.6]))

        if goal_site_name is None:
            goal_site_name = 'goal_reach_site'

        super(SawyerReachingEnvMultitask, self).__init__(xml_path=xml_path, goal_site_name=goal_site_name, sparse_reward=sparse_reward, action_mode=action_mode)

    def get_obs_dim(self):
        return len(self.get_obs()) + 1 # the additional dim for reward

    def step(self, action):

        self.do_step(action)
        obs = self.get_obs()
        reward, score = self.compute_reward(get_score=True)
        done = False
        info = np.array([score, 0, 0, 0, 0])  # can populate with more info, as desired, for tb logging

        # append reward to obs
        # obs = np.concatenate((obs, np.array([reward])))    #######new: Q:why concatenating reward in obs??

        return obs, reward, done, info

    def reset(self):

        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()

        # # concatenate a dummy rew=0 to the obs
        # ob = np.concatenate((ob, np.array([0])))   #############new: not concatenating the dummy reward

        # print("        env has been reset... task is ", self.model.site_pos[self.site_id_goal])
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

        self.tasks = possible_goals
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

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task ##########new: dummy, since it's called in rl_algo line 369 collect path
        #
        # self.set_goal(self._goal)
        self.set_task_for_env(self._task)
        self.reset()