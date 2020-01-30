from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box
import gym
import os

from rlkit.envs import register_env

SCRIPT_DIR = os.path.dirname(__file__)
from rlkit.envs.reacher.sawyer_reacher import SawyerReachingEnv

##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

class SawyerPegInsertionEnv(SawyerReachingEnv):

    '''
    Inserting a peg into a box (which is at a fixed location)
    '''

    def __init__(self, xml_path=None, goal_site_name=None, sparse_reward=False, box_site_name=None, action_mode='joint_delta_position', obs_mode='state', *args, **kwargs):

        self.body_id_box = 0

        if xml_path is None:
            xml_path = os.path.join(SCRIPT_DIR, 'assets/sawyer_peg_insertion.xml')
        if goal_site_name is None:
            goal_site_name = 'goal_insert_site'
        super(SawyerPegInsertionEnv, self).__init__(xml_path=xml_path, goal_site_name=goal_site_name, sparse_reward=sparse_reward, action_mode=action_mode, obs_mode=obs_mode, *args, **kwargs)

        if box_site_name is None:
            box_site_name = "box"
        self.body_id_box = self.model.body_name2id(box_site_name)

    def reset_model(self):

        #### FIXED START
        angles = self.init_qpos.copy()

        velocities = self.init_qvel.copy()
        self.set_state(angles, velocities) #this sets qpos and qvel + calls sim.forward

        return self.get_obs()

    def reset(self):

        # reset task (this is a single-task case)
        self.model.body_pos[self.body_id_box] = np.array([0.5, 0, 0])

        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()

        return ob

    def get_image(self, width=64, height=64, camera_name='track'):
        '''
        peg insertion uses two cameras: scene, end-effector
        return one array with both images concatenated along axis 0
        '''
        # use sim.render to avoid MJViewer which doesn't seem to work without display
        ee_img = self.sim.render(
            width=width / 2,
            height=height,
            camera_name='track_aux_insert',
        )
        ee_img = np.flipud(ee_img)
        scene_img = self.sim.render(
            width=width / 2,
            height=height,
            camera_name='track',
        )
        scene_img = np.flipud(scene_img)
        img = np.concatenate([scene_img, ee_img], axis=1)
        assert img.shape == (width, height, 3)
        return img

##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

@register_env('peg')
class SawyerPegInsertionEnvMultitask(SawyerPegInsertionEnv):

    '''
    Inserting a peg into a box (which could be in various places).
    This env is the multi-task version of peg insertion. The reward always gets concatenated to obs.
    '''

    def __init__(self, xml_path=None, goal_site_name=None, sparse_reward=False, box_site_name=None, action_mode='joint_delta_position', obs_mode='state', *args, **kwargs):

        if xml_path is None:
            xml_path = os.path.join(SCRIPT_DIR, 'assets/sawyer_peg_insertion.xml')
        if goal_site_name is None:
            goal_site_name = 'goal_insert_site'
        if box_site_name is None:
            box_site_name = "box"
        super(SawyerPegInsertionEnvMultitask, self).__init__(xml_path=xml_path, goal_site_name=goal_site_name, sparse_reward=sparse_reward, box_site_name=box_site_name, action_mode=action_mode, obs_mode=obs_mode, *args, **kwargs)

        # limit the set of possible goal positions for the box
        self.limited_pos=False
        self.limited_goals = np.array([[0.5, 0.2, 0], [0.5, -0.2, 0], [0.7, 0.2, 0], [0.7, -0.2, 0]])

    def reset(self):

        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def get_random_box_pos(self):

        if self.limited_pos:
            idx = np.random.randint(0, len(self.limited_goals))
            return self.limited_goals[idx]

        x_low = 0.5
        x_high = 0.7
        y_low = -0.2
        y_high = 0.2
        z_low = z_high = 0

        x = np.random.uniform(x_low, x_high)
        y = np.random.uniform(y_low, y_high)
        z = np.random.uniform(z_low, z_high)

        return np.array([x, y, z])

    def get_obs_dim(self):
        return len(self.get_obs())

    def step(self, action):

        self.do_step(action)
        obs = self.get_obs()
        reward, score, sparse_reward = self.compute_reward(get_score=True)
        done = False
        info = dict(score=score)

        return (obs, reward, done, info)

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

        possible_goals = [self.get_random_box_pos() for _ in range(num_tasks)]
        return possible_goals


    def set_task_for_env(self, goal):

        '''
        Call this function externally,
        to reset the task
        '''

        # task definition = set the goal box location to be the given goal
        self.model.body_pos[self.body_id_box] = goal.copy()

    ##########################################
    ##########################################


##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################


@register_env('peg-4box')
class SawyerPegInsertionEnv4Box(SawyerPegInsertionEnvMultitask):

    '''
    Inserting a peg into 1 of 4 possible boxes.
    This env is a multi-task env. The reward always gets concatenated to obs.
    '''

    def __init__(self, xml_path=None, goal_site_name=None, sparse_reward=False, box_site_name=None, action_mode='joint_delta_position', *args, **kwargs):

        self.which_box = 0
        self.body_id_box1 = 0
        self.body_id_box2 = 0
        self.body_id_box3 = 0
        self.body_id_box4 = 0
        self.site_id_goals = None

        # colors
        self.red = np.array([1, 0, 0, 1])
        self.blue = np.array([0.3, 0.3, 1, 1])

        # init env
        if xml_path is None:
            xml_path = os.path.join(SCRIPT_DIR, 'assets/sawyer_peg_insertion_4box.xml')
        if goal_site_name is None:
            goal_site_name = 'goal_insert_site1'
        if box_site_name is None:
            box_site_name = "box1"
        super(SawyerPegInsertionEnv4Box, self).__init__(xml_path=xml_path, goal_site_name=goal_site_name, sparse_reward=sparse_reward, box_site_name=box_site_name, action_mode=action_mode, *args, **kwargs)

        # ids of sites and bodies
        self.site_id_goals = [self.model.site_name2id('goal_insert_site1'),
                              self.model.site_name2id('goal_insert_site2'),
                              self.model.site_name2id('goal_insert_site3'),
                              self.model.site_name2id('goal_insert_site4'), ]
        self.body_id_box1 = self.model.body_name2id("box1")
        self.body_id_box2 = self.model.body_name2id("box2")
        self.body_id_box3 = self.model.body_name2id("box3")
        self.body_id_box4 = self.model.body_name2id("box4")

    def reset(self):

        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()

        # concatenate dummy rew=0 to the obs
        ob = np.concatenate((ob, np.array([0]), np.array([0])))

        # print("        env has been reset... task is ", self.which_box, " - ", self.model.body_pos[self.body_id_box1], " , ", self.model.body_pos[self.body_id_box2], " ...")
        return ob

    def step(self, action):

        self.do_step(action)

        obs = self.get_obs()
        if self.site_id_goals is None:
            reward, score, sparse_reward = 0,0,0
        else:
            reward, score, sparse_reward = self.compute_reward(get_score=True, goal_id_override=self.site_id_goals[self.which_box])
        done = False
        info = dict(score=score)

        return (obs, reward, done, info)

    def goal_visibility(self, visible):

        """ Toggle the goal visibility when rendering: video should see goal, but image obs shouldn't """

        # box index (0, 1, 2, 3) in code --> (1, 2, 3, 4) in model
        box_num = str(self.which_box+1)

        # select color of box
        if visible:
            which_color = self.red.copy()
        else:
            which_color = self.blue.copy()

        # change box color
        self.model.geom_rgba[self.model.geom_name2id('box_bottom' + box_num)] = which_color
        self.model.geom_rgba[self.model.geom_name2id('box_top' + box_num)] = which_color
        self.model.geom_rgba[self.model.geom_name2id('box_left' + box_num)] = which_color
        self.model.geom_rgba[self.model.geom_name2id('box_right' + box_num)] = which_color


    ##########################################
    ### These are called externally
    ##########################################

    def init_tasks(self, num_tasks, is_eval_env):

        '''
        Call this function externally, ONCE
        to define this env as either train env or test env
        and get the possible task list accordingly
        '''

        # set seed
        if is_eval_env:
            np.random.seed(100) #pick eval tasks as random from diff seed
        else:
            np.random.seed(101)

        # sampling info
        # where to randomly place all of the boxes
        x_range_1 = (0.44, 0.45)
        x_range_2 = (0.6, 0.61)
        y_range_1 = (-0.08, -0.07)
        y_range_2 = (0.07, 0.08)
        unif = np.random.uniform

        possible_goals = []
        for _ in range(num_tasks):
          which_box = np.random.randint(0, 4)
          pos1 = np.array([unif(*x_range_1), unif(*y_range_2), 0.0])
          pos2 = np.array([unif(*x_range_1), unif(*y_range_1), 0.0])
          pos3 = np.array([unif(*x_range_2), unif(*y_range_2), 0.0])
          pos4 = np.array([unif(*x_range_2), unif(*y_range_1), 0.0])
          possible_goals.append([which_box, pos1, pos2, pos3, pos4])

        ##### TEMP (to make all tasks secretly the same)
        # possible_goals=[]
        # for _ in range(num_tasks):
        #     possible_goals.append([which_box, pos1, pos2, pos3, pos4])

        return possible_goals


    def set_task_for_env(self, goal):

        '''
        Call this function externally,
        to reset the task
        '''

        # set which box is the correct one
        self.which_box = goal[0]

        # set the location of each of the boxes
        self.model.body_pos[self.body_id_box1] = goal[1]
        self.model.body_pos[self.body_id_box2] = goal[2]
        self.model.body_pos[self.body_id_box3] = goal[3]
        self.model.body_pos[self.body_id_box4] = goal[4]

    ##########################################
    ##########################################
