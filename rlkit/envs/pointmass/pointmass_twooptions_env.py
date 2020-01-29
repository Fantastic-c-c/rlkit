from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box
import gym
from gym.envs.mujoco import mujoco_env
import os

SCRIPT_DIR = os.path.dirname(__file__)

from rlkit.envs.pointmass.pointmass_env import PointMassEnvMultitask

##################################################################
##################################################################
##################################################################
##################################################################


class PointMassEnvTwoOptions(PointMassEnvMultitask):

    '''
    Pointmass go to a desired location
    '''

    def __init__(self, xml_path=None, goal_site_name=None, sparse_reward=False, action_mode='joint_position'):

        # goal range for this task
        self.goal_options_twooptions = [np.array([-2, 2, 0]), np.array([2,-2,0])]

        if goal_site_name is None:
            goal_site_name = 'goal_reach_site'

        super(PointMassEnvTwoOptions, self).__init__(xml_path=xml_path, goal_site_name=goal_site_name, sparse_reward=sparse_reward, action_mode=action_mode)

    def reset_model(self):
        if not self.startup:
            # reset the pointmass FIXED START
            self.model.site_pos[self.site_id_pointmass] = np.array([0,0,0]).copy()
            self.sim.forward()
        return self.get_obs()


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

        possible_goals=[]
        for i in range(num_tasks):
            if i==0:
                idx=0
            elif i==1:
                idx=1
            else:
                idx = np.random.randint(2)
            possible_goals.append(self.goal_options_twooptions[idx])
        return possible_goals

    def set_task_for_env(self, goal):

        '''
        Call this function externally,
        to reset the task
        '''

        # task definition = set the goal reacher location to be the given goal
        self.model.site_pos[self.site_id_goal] = goal.copy()
        self.sim.forward()

    ##########################################
    ##########################################
