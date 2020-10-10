from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box
import gym
import os
from PIL import Image

from rlkit.envs import register_env
SCRIPT_DIR = os.path.dirname(__file__)
from rlkit.envs.reacher.sawyer_reacher import SawyerReachingEnv

##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

@register_env('button')
class SawyerButtonsEnv(SawyerReachingEnv):

    '''
    Inserting a peg into 1 of 4 possible boxes.
    This env is a multi-task env. The reward always gets concatenated to obs.
    '''

    def __init__(self, xml_path=None, goal_site_name=None, action_mode='joint_delta_position', *args, **kwargs):

        if xml_path is None:
            xml_path = os.path.join(SCRIPT_DIR, 'assets/sawyer_buttons.xml')
        if goal_site_name is None:
            goal_site_name = 'goal_insert_site1'

        # init ids to 0 before env creation
        self.which_button = 0
        self.body_id_panel = 0
        self.site_id_goals = None

        self.auto_reset_task = False
        self.auto_reset_task_list = None

        super(SawyerButtonsEnv, self).__init__(xml_path=xml_path, goal_site_name=goal_site_name, action_mode=action_mode, *args, **kwargs)

        # ids of sites and bodies
        self.site_id_goals = [self.model.site_name2id('goal_insert_site1'),
                              self.model.site_name2id('goal_insert_site2'),
                              self.model.site_name2id('goal_insert_site3'),
                             ]
        self.body_id_panel = self.model.body_name2id("controlpanel")
        self.task2goal = {}
        self.train_tasks = None

        # make sure vis is off
        self.goal_visibility(visible=False)

    def reset_model(self):

        #### FIXED START
        angles = self.init_qpos.copy()

        # NOTICE PLAY AROUND WITH STARTING ANGLE
        # angles[0] = -0.5185 # to 0.3

        # angles[0] = np.random.uniform(-0.5185, 0.3)

        velocities = self.init_qvel.copy()
        self.set_state(angles, velocities) #this sets qpos and qvel + calls sim.forward

        return self.get_obs()


    def reset(self):

        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()

        return ob

    def get_image(self, width=64, height=64, camera_name='track'):
        '''
        peg insertion uses two cameras: scene, end-effector
        return one array with both images concatenated along axis 0
        '''
        # vis. determined by size of requested image - sketchy!!
        is_vis = width >= 128
        if is_vis:
            self.goal_visibility(visible=True)

        # use sim.render to avoid MJViewer which doesn't seem to work without display
        ee_img = self.sim.render(
            width=width,
            height=height,
            camera_name='track_aux_insert',
        )
        ee_img = np.flipud(ee_img)
        scene_img = self.sim.render(
            width=width,
            height=height,
            camera_name='track',
        )
        scene_img = np.flipud(scene_img)
        img = np.concatenate([scene_img, ee_img], axis=1)
        # resize image to be square
        img = Image.fromarray(img)
        img = img.resize((width, height))
        img = np.array(img)
        assert img.shape == (width, height, 3)
        return img

    def get_obs_dim(self):
        return len(self.get_obs())


    def step(self, action):

        self.do_step(action)

        obs = self.get_obs()
        if self.site_id_goals is None:
            reward, score, sparse_reward = 0,0,0
        else:
            reward, score, sparse_reward = self.compute_reward(get_score=True, goal_id_override=self.site_id_goals[self.which_button], button=self.which_button)
        done = False
        info = dict(score=score, sparse_reward=sparse_reward)

        return obs, reward, done, info

    def goal_visibility(self, visible):

        """ Toggle the goal visibility when rendering: video should see goal, but image obs shouldn't """

        # set all sites to invisible (alpha)
        for id_num in self.site_id_goals:
            self.model.site_rgba[id_num][3]=0.0

        # set only the one visible site to visible
        if visible:
            self.model.site_rgba[self.site_id_goals[self.which_button]] = 1.0


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

        x_low = 0.72
        x_high = 0.78
        y_low = 0.0
        y_high = 0.0
        z_low = z_high = -0.15

        assert num_tasks % 3 == 0

        possible_goals = []
        for task_id in range(num_tasks // 3):
            x = np.random.uniform(x_low, x_high)
            y = np.random.uniform(y_low, y_high)
            z = np.random.uniform(z_low, z_high)
            for button_idx in range(3):
                possible_goals.append([button_idx, x, y, z])

        return possible_goals


    def set_task_for_env(self, goal):

        '''
        Call this function externally,
        to reset the task
        '''

        # set which box is the correct one
        self.which_button = goal[0]

        # set the location of the panel
        self.model.body_pos[self.body_id_panel] = goal[1:]

    ##########################################
    ##########################################

    def set_auto_reset_task(self, task_list):
        self.auto_reset_task = True
        self.auto_reset_task_list = task_list

    ##########################################
    ##########################################

    def set_task_dict(self, train_tasks):
        self.train_tasks = train_tasks
        dummy_action = np.zeros(7)
        for task in train_tasks:
            self.set_task_for_env(task)
            self.do_step(dummy_action)
            goal_xyz = self.data.site_xpos[self.site_id_goals[self.which_button]].copy()
            self.task2goal[tuple(task)] = goal_xyz

