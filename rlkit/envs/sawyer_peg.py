from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box

from rlkit.envs.mujoco_env import MujocoEnv
from . import register_env


@register_env('torque-reach')
class SawyerTorqueReachingEnv(MujocoEnv):
    '''
    Reaching to a desired pose with 7 DoF joint torque control
    '''
    def __init__(self, xml_path=None, max_path_length=30, n_tasks=1, randomize_tasks=False, tasks = [np.array([0.75, 0, 0])]):
        self.max_path_length = max_path_length
        self.frame_skip = 5
        goal_low = (0.4, -0.1, 0),
        goal_high = (0.6, 0.1, 0),

        # "tasks" gives the position of the location of the box (not goal_p1, goal_p1 = box position + [0, 0, 0.04])

        if xml_path is None:
            xml_path = 'sawyer_reach.xml'
        super(SawyerTorqueReachingEnv, self).__init__(
                xml_path,
                frame_skip=self.frame_skip, # sim rate / control rate ratio
                automatically_set_obs_and_action_space=True)
        # set the reset position as defined in XML
        self.init_qpos = self.sim.model.key_qpos[0].copy()

        # set the action space to be the control range
        # if wrapped in NormalizedBoxEnv, actions will be automatically scaled to this range
        bounds = self.model.actuator_ctrlrange.copy()
        self.action_space = Box(low=bounds[:, 0], high=bounds[:, 1])
        # set the observation space to be inf because we don't care
        obs_size = len(self.get_obs())
        self.observation_space = Box(low=-np.ones(obs_size) * np.inf, high=np.ones(obs_size) * np.inf)

        # TODO multitask stuff

        if randomize_tasks:
            self.tasks = [1 * np.random.uniform(goal_low, goal_high) for _ in range(n_tasks)]
        else:
            self.tasks = tasks
            if (n_tasks > len(self.tasks)):
                raise NotImplementedError("We don't have enough goals defined")

        # previous single task
        # self._goal = self.data.site_xpos[self.model.site_name2id('goal_p1')].copy()

        # self._goal = self.sim.model.site_pos[self.model.site_name2id('goal_p1')] + tasks[0]
        # self.sim.model.body_pos[self.model.body_name2id('box')] = np.array([0.9,0.4,0.04])

        self.reset_task(0)
        # self.reset()

    def get_obs(self):
        ''' state observation is joint angles + joint velocities + ee pose '''
        angles = self._get_joint_angles()
        velocities = self._get_joint_velocities()
        ee_pose = self._get_ee_pose()
        # import pdb; pdb.set_trace()
        return np.concatenate([angles, velocities, ee_pose])

    def _get_joint_angles(self):
        return self.data.qpos.copy()

    def _get_joint_velocities(self):
        return self.data.qvel.copy()

    def _get_ee_pose(self):
        ''' ee pose is xyz position + orientation quaternion '''
        # TODO this is only position right now!
        ee_id = self.model.body_names.index('end_effector')
        return self.data.body_xpos[ee_id].copy()

    def reset_model(self):
        ''' reset to the same starting pose defined by joint angles '''
        angles = self.init_qpos
        velocities = np.zeros(len(self.data.qvel))
        self.set_state(angles, velocities)
        # TODO is this sim forward needed?
        self.sim.forward()
        return self.get_obs()

    def step(self, action):
        ''' apply the 7DoF action provided by the policy '''
        torques = action
        # for now, the sim rate is 5 times the control rate
        self.do_simulation(torques, self.frame_skip)
        obs = self.get_obs()
        reward = self.compute_reward()
        done = False
        return obs, reward, done, {}

    def compute_reward(self):
        ''' reward is the GPS cost function on the distance of the ee
        to the goal position '''
        # get coordinates of the points on the peg in the world frame
        # n.b. `data` coordinates are in world frame, while `model` coordinates are in local frame
        p1 = self.data.site_xpos[self.model.site_name2id('ee_p1')].copy()
        p2 = self.data.site_xpos[self.model.site_name2id('ee_p2')].copy()
        p3 = self.data.site_xpos[self.model.site_name2id('ee_p3')].copy()
        stacked_peg_points = np.concatenate([p1, p2, p3])

        # get coordinates of the goal points in the world frame
        g1 = self.data.site_xpos[self.model.site_name2id('goal_p1')].copy()
        g2 = self.data.site_xpos[self.model.site_name2id('goal_p2')].copy()
        g3 = self.data.site_xpos[self.model.site_name2id('goal_p3')].copy()
        stacked_goal_points = np.concatenate([g1, g2, g3])

        # compute distance between the points
        dist = np.linalg.norm(stacked_goal_points - stacked_peg_points)
        # hack to get the right scale for the desired cost fn. shape
        # the best shape is when the dist is in [-5, 5]
        dist *= 5
        print("@@@@@@@@@@@@@@ dist: ", dist, "   reward: ", -(dist ** 2 + math.log10(dist ** 2 + 1e-5)))
        # use GPS cost function: log + quadratic encourages precision near insertion
        return -(dist ** 2 + math.log10(dist ** 2 + 1e-5))

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

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._goal = self.tasks[idx]
        # import pdb; pdb.set_trace()

        # self.data.site_xpos: absolute position
        # self.sim.model.site_pos: local position, same as declared in xml files

        # put the box at the goal position
        self.sim.model.body_pos[self.model.body_name2id('box')] = self._goal

        # move the goal position slightly to account for the hole offset from the box
        self._goal = self._goal + self.sim.model.site_pos[self.model.site_name2id('goal_p1')]
        self.reset()


@register_env('torque-peg-insert')
class SawyerTorquePegInsertionEnv(SawyerTorqueReachingEnv):
    '''
    Top down peg insertion with 7DoF joint torque control
    '''
    def __init__(self, *args, **kwargs):
        xml_path = 'sawyer_peg_insertion.xml'
        super(SawyerTorquePegInsertionEnv, self).__init__(xml_path=xml_path, *args, **kwargs)








class SawyerTorquePegInsertionBoxEnv(SawyerTorqueReachingEnv):
    '''
    Top down peg insertion with 7DoF joint torque control
    '''

    def __init__(self, xml_path=None, max_path_length=30, n_tasks=1, randomize_tasks=False,
                 tasks=[np.array([[0.35, 0.2, -0.07]])]):
        self.max_path_length = max_path_length
        self.frame_skip = 5

        task = tasks[0] # currently assumes only 1 task

        goal_low = (task[0][0], task[0][1], task[0][2]),
        goal_high = (task[0][0], task[0][1], task[0][2]+0.06),

        # "tasks" gives the position of the location of the box (not goal_p1, goal_p1 = box position + [0, 0, 0.04])

        if xml_path is None:
            xml_path = 'sawyer_peg_insertion.xml'
        super(SawyerTorqueReachingEnv, self).__init__(
            xml_path,
            frame_skip=self.frame_skip,  # sim rate / control rate ratio
            automatically_set_obs_and_action_space=True)
        # set the reset position as defined in XML
        self.init_qpos = self.sim.model.key_qpos[0].copy()

        # set the action space to be the control range
        # if wrapped in NormalizedBoxEnv, actions will be automatically scaled to this range
        bounds = self.model.actuator_ctrlrange.copy()
        self.action_space = Box(low=bounds[:, 0], high=bounds[:, 1])
        # set the observation space to be inf because we don't care
        obs_size = len(self.get_obs())
        self.observation_space = Box(low=-np.ones(obs_size) * np.inf, high=np.ones(obs_size) * np.inf)

        # TODO multitask stuff

        if randomize_tasks:
            self.tasks = [1 * np.random.uniform(goal_low, goal_high) for _ in range(n_tasks)]
        else:
            self.tasks = tasks
            if (n_tasks > len(self.tasks)):
                raise NotImplementedError("We don't have enough goals defined")

        # previous single task
        # self._goal = self.data.site_xpos[self.model.site_name2id('goal_p1')].copy()

        # self._goal = self.sim.model.site_pos[self.model.site_name2id('goal_p1')] + tasks[0]
        # self.sim.model.body_pos[self.model.body_name2id('box')] = np.array([0.9,0.4,0.04])

        self.reset_task(0)

    #
    # def reset_task(self, idx):
    #     self._goal = self.tasks[idx]
    #
    #     # self.sim.model.body_pos[self.model.body_name2id('box')] = self._goal
    #     self._goal = self._goal + self.sim.model.site_pos[self.model.site_name2id('goal_p1')]
    #     wall = self.model.body_names.index('wall')
    #     # import pdb; pdb.set_trace()
    #     self.sim.model.body_pos[wall][2] = self.tasks[idx][0][2]
    #     print(self.sim.model.body_pos[wall])
    #     # import pdb; pdb.set_trace()
    #     self.reset()

@register_env('torque-peg-wall')
class SawyerTorquePegInsertionWallEnv(SawyerTorquePegInsertionBoxEnv):
    def __init__(self, *args, **kwargs):
        xml_path = 'sawyer_peg_insertion_wall.xml'
        super(SawyerTorquePegInsertionWallEnv, self).__init__(xml_path=xml_path, *args, **kwargs)

    def reset_task(self, idx):
        self._goal = self.tasks[idx]

        # self.sim.model.body_pos[self.model.body_name2id('box')] = self._goal
        self._goal = self._goal + self.sim.model.site_pos[self.model.site_name2id('goal_p1')]
        wall = self.model.body_names.index('wall')
        # import pdb; pdb.set_trace()
        self.sim.model.body_pos[wall][2] = self.tasks[idx][0][2]
        print(self.sim.model.body_pos[wall])
        # import pdb; pdb.set_trace()
        self.reset()

@register_env('torque-peg-inclined-box')
class SawyerTorquePegInsertionIncBoxEnv(SawyerTorquePegInsertionBoxEnv):
    def __init__(self, xml_path=None, max_path_length=30, n_tasks=1, randomize_tasks=False,
                 tasks=[np.array([[0.9659258, 0, 0.258819, 0 ]])]):
        self.max_path_length = max_path_length
        self.frame_skip = 5

        task = tasks[0] # currently assumes only 1 task

        goal_low = ( 0.9659258, 0, -0.258819, 0 ),
        goal_high = (0.9659258, 0, 0.258819, 0  ),
        # goal_low = (1, 0, -0.2, 0)
        # goal_high = (1, 0, -0.2, 0)


        # "tasks" gives the position of the location of the box (not goal_p1, goal_p1 = box position + [0, 0, 0.04])

        if xml_path is None:
            xml_path = 'sawyer_peg_insertion.xml'
        super(SawyerTorqueReachingEnv, self).__init__(
            xml_path,
            frame_skip=self.frame_skip,  # sim rate / control rate ratio
            automatically_set_obs_and_action_space=True)
        # set the reset position as defined in XML
        self.init_qpos = self.sim.model.key_qpos[0].copy()

        # set the action space to be the control range
        # if wrapped in NormalizedBoxEnv, actions will be automatically scaled to this range
        bounds = self.model.actuator_ctrlrange.copy()
        self.action_space = Box(low=bounds[:, 0], high=bounds[:, 1])
        # set the observation space to be inf because we don't care
        obs_size = len(self.get_obs())
        self.observation_space = Box(low=-np.ones(obs_size) * np.inf, high=np.ones(obs_size) * np.inf)

        # TODO multitask stuff

        if randomize_tasks:
            self.tasks = [1 * np.random.uniform(goal_low, goal_high) for _ in range(n_tasks)]
        else:
            self.tasks = tasks
            if (n_tasks > len(self.tasks)):
                raise NotImplementedError("We don't have enough goals defined")

        # previous single task
        # self._goal = self.data.site_xpos[self.model.site_name2id('goal_p1')].copy()

        # self._goal = self.sim.model.site_pos[self.model.site_name2id('goal_p1')] + tasks[0]
        # self.sim.model.body_pos[self.model.body_name2id('box')] = np.array([0.9,0.4,0.04])

        self.reset_task(0)
    def reset_task(self, idx):
        quat = self.tasks[idx]
     #####approach: instead of giving the robot the position of the goal end effector(inside the hole)
        box = self.model.body_name2id('box')  ###### give it the quat instead... currently still giving the goal pos
        self.model.body_quat[box] = quat
        self.reset()
        self._goal = self.data.site_xpos[self.model.site_name2id('goal_p1')]
        ###put the above line at the last step since site_xpos is only calculated after reset


        # from metaworld.metaworld.envs.env_util import quat_to_zangle, zangle_to_quat
        # box = self.model.body_name2id('box')
        # zangle_delta = np.pi/8
        #
        # from pyquaternion import Quaternion
        #
        # new_mocap_zangle = Quaternion(axis=[0, 1, 1], angle=(zangle_delta))
        # # new_mocap_zangle = quat_to_zangle(self.model.body_quat[box]) + zangle_delta
        # # import pdb; pdb.set_trace()
        # self.model.body_quat[box] = new_mocap_zangle
        #
        # # self.model.body_quat[box][0], self.model.body_quat[box][3] = self.model.body_quat[box][3], self.model.body_quat[box][0]
        # self.reset()
        # # import pdb; pdb.set_trace()
        # self._goal = self.data.site_xpos[self.model.site_name2id('goal_p1')]




# class SawyerTorquePegNormal(SawyerTorqueReachingEnv):
#     '''
#     Top down peg insertion with 7DoF joint torque control
#     '''
#     def __init__(self, *args, **kwargs):
#         xml_path = 'sawyer_peg_insertion.xml'
#         super(SawyerTorquePegNormal, self).__init__(xml_path=xml_path, *args, **kwargs)
#
# class SawyerTorquePegPlate(SawyerTorqueReachingEnv):
#     '''
#     Top down peg insertion with 7DoF joint torque control
#     '''
#     def __init__(self, *args, **kwargs):
#         xml_path = 'sawyer_peg_insertion_plate.xml'
#         super(SawyerTorquePegPlate, self).__init__(xml_path=xml_path, *args, **kwargs)
#
# class SawyerTorquePegCross(SawyerTorqueReachingEnv):
#     '''
#     Top down peg insertion with 7DoF joint torque control
#     '''
#     def __init__(self, *args, **kwargs):
#         xml_path = 'sawyer_peg_insertion_cross.xml'
#         super(SawyerTorquePegCross, self).__init__(xml_path=xml_path, *args, **kwargs)
#
# class SawyerTorquePegL(SawyerTorqueReachingEnv):
#     '''
#     Top down peg insertion with 7DoF joint torque control
#     '''
#     def __init__(self, *args, **kwargs):
#         xml_path = 'sawyer_peg_insertion_l.xml'
#         super(SawyerTorquePegL, self).__init__(xml_path=xml_path, *args, **kwargs)
#
# class SawyerTorquePegEight(SawyerTorqueReachingEnv):
#     '''
#     Top down peg insertion with 7DoF joint torque control
#     '''
#     def __init__(self, *args, **kwargs):
#         xml_path = 'sawyer_peg_insertion_8.xml'
#         super(SawyerTorquePegEight, self).__init__(xml_path=xml_path, *args, **kwargs)


# @register_env('torque-peg-shapes')
# class SawyerTorquePegFiveShapesEnv(SawyerTorqueReachingEnv):
#     '''
#     Top down peg insertion with 7DoF joint torque control
#     idea: input number of num tasks is still the same, but the env actually keeps n*num_tasks tasks
#     where n is the number of shapes we are considering. after task is sampled, multiply the task number
#     by n to get the real task number, then reset task accordingly (change xml_path, goal location, etc)
#     '''
#
#     def __init__(self, xml_path=None, max_path_length=30, n_tasks=1, randomize_tasks=False, tasks = [np.array([0.7, 0, 0])]):
#         self.max_path_length = max_path_length
#         self.frame_skip = 5
#         goal_low = (0.4, -0.1, 0),
#         goal_high = (0.6, 0.1, 0),
#
#         self.num_types = 5
#
#         if randomize_tasks:
#             self.tasks = [1 * np.random.uniform(goal_low, goal_high) for _ in range(n_tasks / self.num_types)]
#         else:
#             self.tasks = tasks
#             if (n_tasks / self.num_types > len(self.tasks)):
#                 raise NotImplementedError("We don't have enough goals defined")
#
#         self.env_normal = SawyerTorqueReachingEnv(xml_path='sawyer_peg_insertion.xml', max_path_length=30, n_tasks=1, randomize_tasks=False, tasks=self.tasks)
#         self.env_plate = SawyerTorqueReachingEnv(xml_path='sawyer_peg_insertion_plate.xml', max_path_length=30, n_tasks=1, randomize_tasks=False, tasks=self.tasks)
#         self.env_cross = SawyerTorqueReachingEnv(xml_path='sawyer_peg_insertion_cross.xml', max_path_length=30, n_tasks=1, randomize_tasks=False, tasks=self.tasks)
#         self.env_l = SawyerTorqueReachingEnv(xml_path='sawyer_peg_insertion_l.xml', max_path_length=30, n_tasks=1, randomize_tasks=False, tasks=self.tasks)
#         self.env_eight = SawyerTorqueReachingEnv(xml_path='sawyer_peg_insertion_8.xml', max_path_length=30, n_tasks=1, randomize_tasks=False, tasks=self.tasks)
#
#         self.envs = [self.env_normal, self.env_plate, self.env_cross, self.env_l, self.env_eight]
#         self.type = 0
#         super(SawyerTorquePegFiveShapesEnv, self).__init__(tasks=tasks)
#         # import pdb;
#         # pdb.set_trace()
#
#         self.reset_task(self.type)
#
#     def get_obs(self):
#         ''' state observation is joint angles + joint velocities + ee pose '''
#         return self.envs[self.type].get_obs()
#
#     def _get_joint_angles(self):
#         return self.envs[self.type]._get_joint_angles()
#
#     def _get_joint_velocities(self):
#         return self.envs[self.type]._get_joint_velocities()
#
#     def _get_ee_pose(self):
#         ''' ee pose is xyz position + orientation quaternion '''
#         return self.envs[self.type]._get_ee_pose()
#
#     def reset_model(self):
#         ''' reset to the same starting pose defined by joint angles '''
#         return self.envs[self.type].reset_model()
#
#     def step(self, action):
#         ''' apply the 7DoF action provided by the policy '''
#         print("fffffff ", self.type)
#         return self.envs[self.type].step(action)
#
#     def compute_reward(self):
#         ''' reward is the GPS cost function on the distance of the ee
#         to the goal position '''
#         return self.envs[self.type].compute_reward()
#
#     def viewer_setup(self):
#         # side view
#         self.envs[self.type].viewer_setup()
#
#     def get_all_task_idx(self):
#         return range(len(self.envs[self.type].tasks) * self.num_types)
#
#     def reset_task(self, idx):
#         self.type = idx % self.num_types
#         print("wtf?? ", self.type)
#         self.envs[self.type].reset_task(idx // self.num_types)
#
#         self._goal = self.tasks[idx // self.num_types]
#
#         # self.data.site_xpos: absolute position
#         # self.sim.model.site_pos: local position, same as declared in xml files
#
#         # put the box at the goal position
#         self.sim.model.body_pos[self.model.body_name2id('box')] = self._goal
#
#         # move the goal position slightly to account for the hole offset from the box
#         self._goal = self._goal + self.sim.model.site_pos[self.model.site_name2id('goal_p1')]


@register_env('torque-peg-shapes')
class SawyerTorquePegFiveShapesEnv(SawyerTorqueReachingEnv):
    '''
    Top down peg insertion with 7DoF joint torque control
    idea: input number of num tasks is still the same, but the env actually keeps n*num_tasks tasks
    where n is the number of shapes we are considering. after task is sampled, multiply the task number
    by n to get the real task number, then reset task accordingly (change xml_path, goal location, etc)
    '''

    def __init__(self, xml_path=None, max_path_length=30, n_tasks=1, randomize_tasks=False, tasks = [np.array([0.7, 0, 0])]):
        self.max_path_length = max_path_length
        self.frame_skip = 5
        goal_low = (0.4, -0.1, 0),
        goal_high = (0.6, 0.1, 0),

        self.envs = ['sawyer_peg_insertion.xml', 'sawyer_peg_insertion_plate.xml', 'sawyer_peg_insertion_cross.xml', \
                     'sawyer_peg_insertion_l.xml', 'sawyer_peg_insertion_8.xml']
        self.num_types = 5
        self.type = 0
        # "tasks" gives the position of the location of the box (not goal_p1, goal_p1 = box position + [0, 0, 0.04])

        if xml_path is None:
            xml_path = 'sawyer_reach.xml'
        super(SawyerTorqueReachingEnv, self).__init__(
            xml_path,
            frame_skip=self.frame_skip,  # sim rate / control rate ratio
            automatically_set_obs_and_action_space=True)
        # set the reset position as defined in XML
        self.init_qpos = self.sim.model.key_qpos[0].copy()

        # set the action space to be the control range
        # if wrapped in NormalizedBoxEnv, actions will be automatically scaled to this range
        bounds = self.model.actuator_ctrlrange.copy()
        self.action_space = Box(low=bounds[:, 0], high=bounds[:, 1])
        # set the observation space to be inf because we don't care
        obs_size = len(self.get_obs())
        self.observation_space = Box(low=-np.ones(obs_size) * np.inf, high=np.ones(obs_size) * np.inf)

        # TODO multitask stuff

        if randomize_tasks:
            self.tasks = [1 * np.random.uniform(goal_low, goal_high) for _ in range(n_tasks // self.num_types)]
        else:
            self.tasks = tasks
            if (n_tasks // self.num_types > len(self.tasks)):
                raise NotImplementedError("We don't have enough goals defined")

        # previous single task
        # self._goal = self.data.site_xpos[self.model.site_name2id('goal_p1')].copy()

        # self._goal = self.sim.model.site_pos[self.model.site_name2id('goal_p1')] + tasks[0]
        # self.sim.model.body_pos[self.model.body_name2id('box')] = np.array([0.9,0.4,0.04])

        self.reset_task(0)
        # self.reset()

    def get_all_task_idx(self):
        return range(len(self.tasks) * self.num_types)

    def reset_task(self, idx):

        xml_path = self.envs[self.type]

        super(SawyerTorqueReachingEnv, self).__init__(
            xml_path,
            frame_skip=self.frame_skip,  # sim rate / control rate ratio
            automatically_set_obs_and_action_space=True)

        self._goal = self.tasks[idx // self.num_types]
        self.type = idx % self.num_types
        # self.data.site_xpos: absolute position
        # self.sim.model.site_pos: local position, same as declared in xml files

        # put the box at the goal position
        self.sim.model.body_pos[self.model.body_name2id('box')] = self._goal

        # move the goal position slightly to account for the hole offset from the box
        self._goal = self._goal + self.sim.model.site_pos[self.model.site_name2id('goal_p1')]

        print("current goal: ", self._goal, "   tasks: ",self.tasks)
        self.init_qpos = self.sim.model.key_qpos[0].copy()

        # set the action space to be the control range
        # if wrapped in NormalizedBoxEnv, actions will be automatically scaled to this range
        bounds = self.model.actuator_ctrlrange.copy()
        self.action_space = Box(low=bounds[:, 0], high=bounds[:, 1])
        # set the observation space to be inf because we don't care
        obs_size = len(self.get_obs())
        self.observation_space = Box(low=-np.ones(obs_size) * np.inf, high=np.ones(obs_size) * np.inf)
        self.reset()