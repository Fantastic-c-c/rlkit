from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box
import gym
import os
from PIL import Image
from gym.envs.mujoco import mujoco_env
from rlkit.envs import register_env

SCRIPT_DIR = os.path.dirname(__file__)

@register_env('shelf')
class SawyerPegShelfEnvMultitask(mujoco_env.MujocoEnv):
    '''
    Inserting a peg into a box (which is at a fixed location)
    '''

    def __init__(self, xml_path='train40_eval10tasks.xml', box_site_name=None,
                 action_mode='joint_position', task_mode='weight', obs_mode='state', *args, **kwargs):
        assert action_mode in ["joint_position", "joint_delta_position", "torque"]
        assert task_mode in ["weight", "position"]

        assert task_mode == 'weight', "position mode deprecated, at least for now"
        self.task_mode = task_mode
        self.mug_weight = None
        self.single_task_eval = False
        self.step_counter = 0
        self.obs_mode = obs_mode

        '''
        if xml_path is None:
            print("creating a sample shelf env, for generating tasks")
            if action_mode == "torque":
                print("torque control")
                xml_path = os.path.join(SCRIPT_DIR, 'assets/sawyer_shelf_placing_torqueCtrl.xml')
            else:
                print("pos control")
                xml_path = os.path.join(SCRIPT_DIR, 'assets/sawyer_shelf_placing_posCtrl.xml')
        else:
        '''
        xml_path = os.path.join(SCRIPT_DIR, 'assets', xml_path)
        print("creating a multiple weight shelf env")
        print("Control mode: {}".format(action_mode))
        print("xml path:", xml_path)

        self.xml_path= xml_path
        self.goal_site_name = 'goal'

        # vars
        self.action_mode = action_mode
        self.num_joint_dof = 9 # 7 + 2 gripper
        self.frame_skip = 100

        self.is_eval_env = False

        # Sparse reward setting
        self.truncation_dist = 0.3
        # if distance from goal larger than this,
        # get dist(self.truncation_dis) reward every time steps.
        # The dist is around 1 in starting pos

        # create the env
        self.startup = True
        mujoco_env.MujocoEnv.__init__(self, xml_path,
                                      self.frame_skip)  # self.model.opt.timestep is 0.0025 (w/o frameskip)
        self.startup = False

        # initial position of joints
        self.init_qpos = self.sim.model.key_qpos[0].copy()
        self.init_qvel = np.zeros(len(self.data.qvel))

        # joint limits
        self.limits_lows_joint_pos = self.model.actuator_ctrlrange.copy()[:, 0] ### TODO verify
        self.limits_highs_joint_pos = self.model.actuator_ctrlrange.copy()[:, 1]


        # set the action space (always between -1 and 1 for this env)
        self.action_highs = np.ones((self.num_joint_dof,))
        self.action_lows = -1 * np.ones((self.num_joint_dof,))
        self.action_space = Box(low=self.action_lows, high=self.action_highs)

        # set the observation space
        obs_size = self.get_obs_dim()
        self.observation_space = Box(low=-np.ones(obs_size) * np.inf, high=np.ones(obs_size) * np.inf)

        # vel limits
        joint_vel_lim = 0.04  ##### 0.07 # magnitude of movement allowed within a dt [deg/dt]
        self.limits_lows_joint_vel = -np.array([joint_vel_lim] * self.num_joint_dof)
        self.limits_highs_joint_vel = np.array([joint_vel_lim] * self.num_joint_dof)

        # ranges
        self.action_range = self.action_highs - self.action_lows
        self.joint_pos_range = (self.limits_highs_joint_pos - self.limits_lows_joint_pos)
        self.joint_vel_range = (self.limits_highs_joint_vel - self.limits_lows_joint_vel)

        # ids
        self.site_id_goal = self.model.site_name2id(self.goal_site_name)
        self.site_id_goal_top = self.model.site_name2id(self.goal_site_name + '_top')

        self.site_id_mug = None
        self.site_id_mug_top = None
        # self.body_id_mug = self.model.body_name2id('mug')

        # goal position
        self.goal_x = 0
        self.goal_y = -0.1
        self.goal_z = 0.22
        self.goal_z_top = 0.32
        self.goal_pos = np.array([self.goal_x, self.goal_y, self.goal_z])
        self.init_mug_pose = np.array([-0.125, 0.7, 0]) # different from goal_pos just because different reference frame 0 0.6 0.007

        # dict
        self.weight2idx = None # need to call  py_env.assign_tasks(train_tasks) externally to set
        self.idx2weight = None
        self.curr_weight = None
        self.curr_idx = None
        self.curr_weight = None
        self.top_reward = None

        # temp
        self.is_slac = False

        # make sure vis is off
        self.goal_visibility(visible=False)

    def get_image(self, width=64, height=64, camera_name='track'):
        # use sim.render to avoid MJViewer which doesn't seem to work without display
        is_vis = width >= 128
        if is_vis:
            self.goal_visibility(visible=True)

        # render both images at full width, then downsample to make square
        ee_img = self.sim.render(
            width=width,
            height=height,
            camera_name='track_aux_shelf2',
        )
        ee_img = np.flipud(ee_img)
        scene_img = self.sim.render(
            width=width,
            height=height,
            camera_name='track',
        )
        scene_img = np.flipud(scene_img)
        img = np.concatenate([scene_img, ee_img], axis=1)
        img = Image.fromarray(img)
        img = img.resize((width, height))
        img = np.array(img)
        assert img.shape == (width, height, 3)
        if is_vis:
            self.goal_visibility(visible=False)
        return img


    def override_action_mode(self, action_mode):
        self.action_mode = action_mode

    def reset_model(self):

        #### FIXED START
        angles = self.init_qpos.copy()

        velocities = self.init_qvel.copy()
        self.set_state(angles, velocities)  # this sets qpos and qvel + calls sim.forward

        self.set_mug_weight(self.mug_weight) # set again # TODO notice here


        if self.is_slac:
          self.set_random_task_for_env()

        return self.get_obs()


    def reset(self):
        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()

        return ob

    def get_obs_dim(self):
        return len(self.get_obs())

    def _get_joint_velocities(self):
        return self.data.qvel.copy()

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
            # ee_pose = self._get_ee_pose()

            num_joint = 9 ### NOTICE the state ONLY contains joint info, not MUG
            angles = angles[:num_joint]
            velocities = velocities[:num_joint]
            return np.concatenate([angles, velocities,])

    def step(self, action):
        # if self.mug_weight:
        #     self.set_mug_weight(self.mug_weight)

        self.do_step(action)
        obs = self.get_obs()
        reward, score, sparse_reward = self.compute_reward(get_score=True)
        done = False
        info = {'score': score}

        # append reward to obs
        self.step_counter += 1

        return obs, reward, done, info

    def do_step(self, action):
        if self.startup:
            feasible_desired_position = 0 * action
            self.do_simulation(feasible_desired_position, self.frame_skip)
            return

        if self.action_mode == 'torque':
            torque_limit = 3 # TODO temporary hardcode
            action = np.clip(action, self.action_lows, self.action_highs)
            action = action * torque_limit
            self.do_simulation(action, self.frame_skip)
            return

        else:
            # clip to action limits
            action = np.clip(action, self.action_lows, self.action_highs)
            # get current position
            curr_position = self._get_joint_angles()[:self.num_joint_dof]
            # print("POSITION: ", curr_position)
            if self.action_mode == 'joint_position':
                # scale incoming (-1,1) to self.joint_limits
                desired_position = (((action - self.action_lows) * self.joint_pos_range) / self.action_range) + self.limits_lows_joint_pos
                # make the
                feasible_desired_position = self.make_feasible(curr_position, desired_position)
            elif self.action_mode == 'joint_delta_position':
                # scale incoming (-1,1) to self.vel_limits
                desired_delta_position = (((action - self.action_lows) * self.joint_vel_range) / self.action_range) + self.limits_lows_joint_vel
                # add delta
                feasible_desired_position = curr_position + desired_delta_position
            else:
                raise NotImplementedError

            # FOR GRIPPER STARTS
            # gripper_position = np.array([0.0064, 0])
            # feasible_desired_position = np.concatenate([feasible_desired_position, gripper_position])
            # FOR GRIPPER ENDS

            self.do_simulation(feasible_desired_position, self.frame_skip)
            return

    def _get_joint_angles(self):
        return self.data.qpos.copy()

    def make_feasible(self, curr_position, desired_position):

        # compare the implied vel to the max vel allowed
        max_vel = self.limits_highs_joint_vel
        implied_vel = np.abs(desired_position - curr_position)

        # limit the vel
        actual_vel = np.min([implied_vel, max_vel], axis=0)

        # find the actual position, based on this vel
        sign = np.sign(desired_position - curr_position)
        actual_difference = sign * actual_vel
        feasible_position = curr_position + actual_difference

        return feasible_position

    def compute_reward(self, get_score=False, goal_id_override=None):
        """
        Almost the same as reacher, just change the ee_xyz to mug_xyz

        """
        assert goal_id_override is None
        self.top_reward = True # TODO hardcoded, will affect reward, but not score

        self.site_id_goal = self.model.site_name2id(self.goal_site_name)
        self.site_id_goal_top = self.model.site_name2id(self.goal_site_name + "_top")
        if self.startup or self.single_task_eval:
            self.site_id_mug = self.model.site_name2id('mugSite0') # all env will have at least one mug
            self.site_id_mug_top = self.model.site_name2id('mugSite0_top')
        else:
            mug_site_name = "mugSite{}".format(self.curr_idx)
            self.site_id_mug = self.model.site_name2id(mug_site_name)
            self.site_id_mug_top = self.model.site_name2id(mug_site_name + "_top")

        # get goal id
        # if goal_id_override is None:
        #     goal_id = self.site_id_goal
        # else:
        #     goal_id = goal_id_override

        # get coordinates of the sites in the world frame
        mug_xyz = self.data.site_xpos[self.site_id_mug].copy()
        goal_xyz = self.data.site_xpos[self.site_id_goal].copy()

        mug_xyz_top = self.data.site_xpos[self.site_id_mug_top].copy()
        goal_xyz_top = self.data.site_xpos[self.site_id_goal_top].copy()

        # score
        score = -np.linalg.norm(mug_xyz - goal_xyz)

        # distance
        dist = 5 * np.linalg.norm(mug_xyz - goal_xyz)

        dist_top = 5 * np.linalg.norm(mug_xyz_top - goal_xyz_top)

        if self.top_reward:
            dist = (dist_top + dist) / 2

        sparse_dist = min(dist, self.truncation_dist)  # if dist too large: return the reward at truncate_dist

        # dense reward
        # use GPS cost function: log + quadratic encourages precision near insertion
        reward = -(dist ** 2 + math.log10(dist ** 2 + 1e-5))

        # sparse reward
        # offset the whole reward such that when dist>truncation_dist, the reward will be exactly 0
        sparse_reward = -(sparse_dist ** 2 + math.log10(sparse_dist ** 2 + 1e-5))
        sparse_reward = sparse_reward - (-(self.truncation_dist ** 2 + math.log10(self.truncation_dist ** 2 + 1e-5)))

        if get_score:
            return reward, score, sparse_reward
        else:
            return reward

    ##########################################
    ### These are called externally
    ##########################################

    def init_tasks(self, num_tasks, is_eval_env):

        '''
        Call this function externally, ONCE
        to define this env as either train env or test env
        and get the possible task list accordingly
        '''

        # NOTE loading pre-generated tasks here
        if num_tasks != 50:
            print('For number of tasks != 50, must re-generate xml files')
            raise Exception
        all_tasks = np.load(os.path.join(SCRIPT_DIR, 'train40_eval10tasks.npy'))
        self.assign_tasks(all_tasks)
        return all_tasks

        # this code will not be executed
        if is_eval_env:
            np.random.seed(0)  # pick eval tasks as random from diff seed
        else:
            np.random.seed(3)
            if num_tasks == 40:
                np.random.seed(2)

        all_weights = self.get_random_weights(num_tasks)
        all_positions = self.get_random_target_pos(num_tasks)

        all_tasks = [[w, p] for w, p in zip(all_weights, all_positions)]

    def get_random_weights(self, num_tasks):
        low = 5
        high = 30
        mug_weights = [np.random.uniform(low, high) for _ in range(num_tasks)]
        return mug_weights


    def get_random_target_pos(self, num_tasks):
        # +- 0.4 is
        low = -0.2
        high = 0.2
        xpos = [np.random.uniform(low, high) for _ in range(num_tasks)]
        return xpos


    def set_task_for_env(self, task):

        '''
        Call this function externally,
        to reset the task
        '''
        mug_weight, target_pos = task
        self.set_goal_pos(target_pos)  # same goal pos
        self.mug_weight = mug_weight
        self.set_mug_weight(mug_weight)

    def set_random_task_for_env(self):
        assert False, "deprecated"
        # self.set_goal_pos()  # same goal pos
        # rand_idx = np.random.randint(len(self.weight2idx))
        # mug_weight = self.idx2weight[rand_idx]
        # self.mug_weight = mug_weight
        # self.set_mug_weight(mug_weight)
        # print("set random task", self.mug_weight)

    def set_goal_pos(self, target_pos):
        print("set goal", target_pos)
        self.goal_pos = np.array([target_pos, self.goal_y, self.goal_z]) # NOTICE only varying x
        self.goal_pos_top = np.array([target_pos, self.goal_y, self.goal_z_top])  # higher z
        self.model.site_pos[self.site_id_goal] = self.goal_pos.copy()
        self.model.site_pos[self.site_id_goal_top] = self.goal_pos_top.copy()

    def assign_tasks(self, tasks):
        all_weights = [t[0] for t in tasks]
        weight2idx = {}
        idx2weight = {}
        for i, w in enumerate(all_weights):
            weight2idx[w] = i
            idx2weight[i] = w
        self.weight2idx = weight2idx
        self.idx2weight = idx2weight

    def set_mug_weight(self, weight):
        if self.single_task_eval:
            print("eval on single mug sample env")
            return

        if weight is None:
            print("unsuccessful")
            return
        # instead of setting the weight, move the one with corresponding weight to the init position
        assert not self.weight2idx is None
        self.curr_weight = weight
        print("set weight", weight)
        self.curr_idx = self.weight2idx[self.curr_weight]

        # 1. Move the current mug back
        # self.reset() # call it again, just to make sure

        # 2. Move the new mug there
        angles = self.init_qpos.copy()
        velocities = self.init_qvel.copy()
        idx_start = self.num_joint_dof + self.curr_idx * 7
        idx_end = self.num_joint_dof + self.curr_idx * 7 + 3 # include only xyz

        angles[idx_start:idx_end] = self.init_mug_pose.copy()
        self.set_state(angles, velocities)  # this sets qpos and qvel + calls sim.forward

        mug_body_name = "mug{}".format(self.curr_idx)
        # print("moved", mug_body_name)
        return

    def set_single_task_eval(self, val):
        self.single_task_eval = val

    def goal_visibility(self, visible):

        ''' Toggle the goal visibility when rendering: video should see goal, but image obs shouldn't '''

        if visible:
            self.model.site_rgba[self.site_id_goal] = np.array([1, 0, 0, 1])
            self.model.site_rgba[self.site_id_goal_top] = np.array([1, 0, 0, 1])
        else:
            self.model.site_rgba[self.site_id_goal] = np.array([1, 0, 0, 0])
            self.model.site_rgba[self.site_id_goal_top] = np.array([1, 0, 0, 0])

    ##########################################
    ##########################################

    def init_slac_settings_temp(self):
        self.is_slac = True
