from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box
import gym
from gym.envs.mujoco import mujoco_env
import os

SCRIPT_DIR = os.path.dirname(__file__)

##################################################################
##################################################################
##################################################################
##################################################################
from rlkit.envs import register_env

@register_env('PointMass-v0')
class PointMassEnv(mujoco_env.MujocoEnv):

    '''
    Pointmass go to a desired location
    '''

    def __init__(self, xml_path=None, goal_site_name=None, sparse_reward=False, action_mode='joint_position'):

        # vars

        self.image_dim = 64  ###########new: size of image obs


        self.action_mode = action_mode
        self.num_joint_dof = 2
        self.frame_skip = 100
        if xml_path is None:
            xml_path = os.path.join(SCRIPT_DIR, 'assets/pointmass.xml')
        if goal_site_name is None:
            goal_site_name = 'goal_reach_site'
        self.body_id_ee = 0
        self.site_id_ee = 0
        self.site_id_goal = 0
        self.is_eval_env = False
        self.init_qpos = np.array([0,0,0]) #xyz position of pointmass

        # Sparse reward setting
        self.sparse_reward = sparse_reward
        self.exp_reward = True

        # create the env
        self.startup = True
        mujoco_env.MujocoEnv.__init__(self, xml_path, self.frame_skip) #self.model.opt.timestep is 0.0025 (w/o frameskip)
        self.startup = False

        # ids
        self.site_id_pointmass = self.model.site_name2id('point')
        self.site_id_goal = self.model.site_name2id(goal_site_name)

        # point mass start/limits
        self.init_qpos = np.array([0.01,0.01,0.01]) #xyz position of pointmass
        self.limits_lows_joint_pos = np.array([-2,-2])
        self.limits_highs_joint_pos = np.array([2,2])

        # set the action space (always between -1 and 1 for this env)
        self.action_highs = np.ones((self.num_joint_dof,))
        self.action_lows = -1*np.ones((self.num_joint_dof,))
        self.action_space = Box(low=self.action_lows, high=self.action_highs)

        # set the observation space
        obs_size = self.get_obs_dim()
        self.observation_space = Box(low=-np.ones(obs_size) * np.inf, high=np.ones(obs_size) * np.inf)

        # vel limits
        joint_vel_lim = 0.2 # magnitude of movement allowed within a dt [deg/dt]
        self.limits_lows_joint_vel = -np.array([joint_vel_lim]*self.num_joint_dof)
        self.limits_highs_joint_vel = np.array([joint_vel_lim]*self.num_joint_dof)

        # ranges
        self.action_range = self.action_highs-self.action_lows
        self.joint_pos_range = (self.limits_highs_joint_pos - self.limits_lows_joint_pos)
        self.joint_vel_range = (self.limits_highs_joint_vel - self.limits_lows_joint_vel)

    def override_action_mode(self, action_mode):
        self.action_mode = action_mode

    def override_reward_mode(self, is_sparse_reward):
        self.sparse_reward = is_sparse_reward

    def get_obs_dim(self):
        return len(self.get_obs())

    def get_obs(self):
        # xy position of pointmass site in the world frame
        if self.startup:
            position=self.init_qpos.copy()
        else:
            position = self.data.site_xpos[self.site_id_pointmass].copy()
        return position[:2]


    def reset_model(self):
        if not self.startup:
            # reset the pointmass

            ####### FIXED START
            # self.model.site_pos[self.site_id_pointmass] = self.init_qpos.copy()

            ####### RAND START
            x = np.random.uniform(self.limits_lows_joint_pos[0], self.limits_highs_joint_pos[0])
            y = np.random.uniform(self.limits_lows_joint_pos[1], self.limits_highs_joint_pos[1])
            self.model.site_pos[self.site_id_pointmass] = np.array([x,y,0])

            self.sim.forward()
        return self.get_obs()

    def do_step(self, action):

        curr_position = self.get_obs().copy()

        if self.startup:
            feasible_desired_position = 0*action 

        else:

            # clip to action limits
            action = np.clip(action, self.action_lows, self.action_highs)

            if self.action_mode=='joint_position':
                # scale incoming (-1,1) to self.joint_limits
                desired_position = (((action - self.action_lows) * self.joint_pos_range) / self.action_range) + self.limits_lows_joint_pos

                # make the position feasible within velocity limits
                feasible_desired_position = self.make_feasible(curr_position, desired_position)

            elif self.action_mode=='joint_delta_position':

                # scale incoming (-1,1) to self.vel_limits
                desired_delta_position = (((action - self.action_lows) * self.joint_vel_range) / self.action_range) + self.limits_lows_joint_vel

                # add delta
                feasible_desired_position = curr_position + desired_delta_position

        self.do_movement(feasible_desired_position, self.frame_skip)

    def do_movement(self, position, fs):
        #move the body site manually
        if not self.startup:
            position = np.clip(position, self.limits_lows_joint_pos, self.limits_highs_joint_pos)
            self.model.site_pos[self.site_id_pointmass] = np.array([position[0], position[1], 0])
            self.sim.forward()

    def step(self, action):
        ''' apply the 7DoF action provided by the policy '''

        self.do_step(action)
        obs = self.get_obs()
        reward, score, sparse_reward = self.compute_reward(get_score=True)
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

        if self.startup:
            reward, score, sparse_reward = 0,0,0

        else:
            threshold_for_sparse = 0.4 ###### TODO hardcoded

            # get goal id
            if goal_id_override is None:
                goal_id = self.site_id_goal
            else:
                goal_id = goal_id_override

            # get coordinates of the sites in the world frame
            pointmass_xyz = self.data.site_xpos[self.site_id_pointmass].copy()
            goal_xyz = self.data.site_xpos[goal_id].copy()

            # distance between the points
            dist = np.linalg.norm(pointmass_xyz[:2] - goal_xyz[:2])

            # score
            score = -dist

            # dense reward
            if self.exp_reward:
                reward = -(dist ** 2 + math.log10(dist ** 2 + 1e-5))
            else:
                reward = -2*dist
            
            # sparse reward
            sparse_reward=-20
            if dist<threshold_for_sparse:
                sparse_reward = reward

            # when we explicitly want sparse reward
            if self.sparse_reward:
                reward = sparse_reward
            print("dist: ", dist, "   reward: ", reward)
        if get_score:
            return reward, score, sparse_reward
        else:
            return reward

    def reset(self):

        # reset task (this is a single-task case)
        self.model.site_pos[self.site_id_goal] = np.array([-1, 1, 0])

        # original mujoco reset (to reset the pointmass)
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def goal_visibility(self, visible):

        ''' Toggle the goal visibility when rendering: video should see goal, but image obs shouldn't '''

        if visible:
            self.model.site_rgba[self.site_id_goal] = np.array([0, 1, 0, 1])
        else:
            self.model.site_rgba[self.site_id_goal] = np.array([0, 1, 0, 0])

##################################################################
##################################################################
##################################################################
##################################################################

@register_env('PointMassMT-v0')
class PointMassEnvMultitask(PointMassEnv):

    '''
    This env is the multi-task version of pointmass. The reward always gets concatenated to obs.
    '''

    def __init__(self, xml_path=None, goal_site_name=None, sparse_reward=False, action_mode='joint_position'):

        # goal range for reaching
        self.goal_options = [np.array([-1, 1, 0]), np.array([1,1,0])]

        if goal_site_name is None:
            goal_site_name = 'goal_reach_site'

        super(PointMassEnvMultitask, self).__init__(xml_path=xml_path, goal_site_name=goal_site_name, sparse_reward=sparse_reward, action_mode=action_mode)

    def get_obs_dim(self):
        return len(self.get_obs()) # the additional dims for reward and sparse reward  #######new: eliminated that -2 since we need dim of obs (state in this case) to match

    def step(self, action):

        self.do_step(action)
        obs = self.get_obs()
        reward, score, sparse_reward= self.compute_reward(get_score=True)
        done = False
        info = np.array([score, 0, 0, 0, 0])  # can populate with more info, as desired, for tb logging

        # append reward to obs
        # obs = np.concatenate((obs, np.array([sparse_reward]), np.array([reward])))  #############new: not concatenating the dummy reward

        return obs, reward, done, info

    def reset(self):

        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()

        # concatenate a dummy rew=0 to the obs
        # ob = np.concatenate((ob, np.array([0]), np.array([0])))   #############new: not concatenating the dummy reward

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

        ##### ONLY 1 OPTIONS  
        if num_tasks==1:
            possible_goals=[]
            possible_goals.append(self.goal_options[0])
            self.tasks = possible_goals
            return possible_goals  

        ##### ONLY 4 OPTIONS  
        elif num_tasks==4:
            possible_goals=[]
            possible_goals.append(np.array([-1, 1, 0]))
            possible_goals.append(np.array([1, 1, 0]))
            possible_goals.append(np.array([-1, -1, 0]))
            possible_goals.append(np.array([1, -1, 0]))
            self.tasks = possible_goals
            return possible_goals  

        ##### ONLY 2 OPTIONS
        elif num_tasks==2:
            possible_goals=[]
            possible_goals.append(self.goal_options[0])
            possible_goals.append(self.goal_options[1])
            self.tasks = possible_goals
            return possible_goals

        ###### MANY OPTIONS
        else:
            possible_goals=[]
            for _ in range(num_tasks):
                x = np.random.uniform(self.limits_lows_joint_pos[0], self.limits_highs_joint_pos[0])
                y = np.random.uniform(self.limits_lows_joint_pos[1], self.limits_highs_joint_pos[1])
                possible_goals.append(np.array([x,y,0]))
            self.tasks = possible_goals
            return possible_goals

    def set_task_for_env(self, goal):

        ''' 
        Call this function externally, 
        to reset the task 
        '''

        # task definition = set the goal reacher location to be the given goal
        self.model.site_pos[self.site_id_goal] = goal.copy()
        self.sim.forward()


    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task ##########new: dummy, since it's called in rl_algo line 369 collect path
        self._state_goal = self._goal ##########new: dummy
        # self.set_goal(self._goal)
        self.set_task_for_env(self._task)
        self.reset()

    ##########################################
    ##########################################
