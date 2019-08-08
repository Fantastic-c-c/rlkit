from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box
from pyquaternion import Quaternion


from . import register_env
from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from metaworld.envs.mujoco.utils.rotation import euler2quat
from metaworld.envs.env_util import quat_to_zangle, zangle_to_quat


@register_env('peg-insert')
class SawyerPegInsertionTopdown6DOFEnv(SawyerXYZEnv):
    def __init__(
            self,
            max_path_length=30,
            hand_low=(-0.5, 0.40, 0.05),
            hand_high=(0.5, 1, 0.5),
            obj_low=(-0.1, 0.6, 0.03),
            obj_high=(0.1, 0.7, 0.03),
            # NOTE: this is NOT the position of the hole, but the position of the box
            # must call reset_task() to set the true goal position to account for the shift
            tasks = [np.array([0, 0.60, 0.00])], ## if want to change the depth of the goal without changing the pos of the box, change xml instead
            # goal positions must be in front of the ee initial position
            goal_low=(-0.05, 0.60, 0.05),
            goal_high=(0.05, 0.70, 0.05),
            hand_init_pos = (0, 0.60, 0.30),  # hand_init_pos y axis should be +0.05 bigger than y of goal to be directly above, to account for offset
            rotMode='fixed', # options are 'fixed', 'rotz', 'quat'
            multitask=False, # this adds task ID to the state space, not for meta-learning
            n_tasks=1,
            if_render=False,
            randomize_tasks=False,  # if True, generate tasks randomly, else use the tasks above
            **kwargs
    ):
        self.quick_init(locals())
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100, # actions are 1cm maximum
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            # **kwargs
        )
        self.max_path_length = max_path_length
        self.rotMode = rotMode
        self.hand_init_pos = np.array(hand_init_pos)
        self.multitask = multitask
        self.n_tasks = n_tasks
        self._goal_idx = np.zeros(self.n_tasks)
        self.if_render = if_render

        # action space is defined by the type of actions allowed
        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            self.action_rot_scale = 1./35
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        elif rotMode == 'quat':
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            )
        else:
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi/2, -np.pi/2, 0, -1]),
                np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1]),
            )

        # observation space includes the task ID in the multitask setting
        if not multitask:
            self.observation_space = Box(np.array(obj_low), np.array(obj_high))
        else:
            self.observation_space = Box(
                    np.hstack((np.array(obj_low), np.zeros(n_tasks))),
                    np.hstack((np.array(obj_low), np.zeros(n_tasks))),
            )

        # in multi-task setting, generate tasks
        if randomize_tasks:
            self.tasks = [1 * np.random.uniform(goal_low, goal_high) for _ in range(n_tasks)]
        else:
            self.tasks = tasks
            if (n_tasks > len(self.tasks)):
                raise NotImplementedError("We don't have enough goals defined")

        # set the initial task
        self.reset_task(0)

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_peg_insertion_topdown.xml')

    def viewer_setup(self):
        # top view
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 1.0
        # self.viewer.cam.lookat[2] = 0.5
        # self.viewer.cam.distance = 0.6
        # self.viewer.cam.elevation = -45
        # self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1
        # side view
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.4
        self.viewer.cam.lookat[1] = 0.75
        self.viewer.cam.lookat[2] = 0.4
        self.viewer.cam.distance = 0.4
        self.viewer.cam.elevation = -55
        self.viewer.cam.azimuth = 180
        self.viewer.cam.trackbodyid = -1

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        ''' reset the environment for the given task '''
        self._goal = self.tasks[idx]

        # put the box at the goal position
        self.sim.model.body_pos[self.model.body_name2id('box')] = self._goal

        # move the goal position slightly to account for the hole offset from the box
        self._goal = self.sim.model.site_pos[self.model.site_name2id('hole')] + self.sim.model.body_pos[self.model.body_name2id('box')]        ## z coordinate of hole: -0.06

        self.reset()

    def reset(self):
        return self.reset_model()

    def step(self, action):
        if self.if_render:
            self.render()
        # self.set_xyz_action_rot(action[:7])
        if self.rotMode == 'euler':
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'fixed':
            self.set_xyz_action(action[:3])
        elif self.rotMode == 'rotz':
            self.set_xyz_action_rotz(action[:4])
        else:
            self.set_xyz_action_rot(action[:7])

        action[-1] = -1
        # TODO what is this doing? isn't the action set with the mocap?
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        obs = self._get_obs()
        reward = self.compute_reward(action, obs)
        self.curr_path_length += 1
        done = False
        # TODO do we want terminals here?
        #if self.curr_path_length == self.max_path_length:
            #done = True
            #print('DONE \n')
        return obs, reward, done, {}

    def _get_obs(self):
        #obs =  self.get_body_com('peg')
        obs = self.get_endeff_pos()
        if self.multitask:
            assert hasattr(self, '_goal_idx')
            return np.concatenate([
                    obs,
                    self._goal,
                    self._goal_idx
                ])
        return obs

    def _get_info(self):
        pass

    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._goal for
        logging, learning, etc.
        """
        objPos =  self.data.get_geom_xpos('handle')
        self.data.site_xpos[self.model.site_name2id('objSite')] = (
            objPos
        )


    def _set_obj_xyz_quat(self, pos, angle):
        quat = Quaternion(axis = [0,0,1], angle = angle).elements
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qpos[12:16] = quat.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)


    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def sample_task(self):
        task_idx = np.random.randint(0, self.n_tasks)
        return self.tasks[task_idx]

    def reset_model(self):
        self._reset_hand()
        self.curr_path_length = 0
        return self._get_obs()

    def _reset_hand(self):
        self.data.set_mocap_pos('mocap', self.hand_init_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

        # TODO: set initial rotation to be slightly rotated around z axis
        if False:
            zangle_delta = np.pi/4
            new_mocap_zangle = quat_to_zangle(self.data.mocap_quat[0]) + zangle_delta
            new_mocap_zangle = np.clip(
                new_mocap_zangle,
                -3.0,
                3.0,
            )
            if new_mocap_zangle < 0:
                new_mocap_zangle += 2 * np.pi

            self.data.set_mocap_quat('mocap', zangle_to_quat(new_mocap_zangle))

        self.sim.forward()
        for _ in range(100):
            self.sim.step()

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_reward(self, actions, obs):

        pegHeadPos = self.get_site_pos('pegHead')
        placingDistHead = np.linalg.norm(pegHeadPos - self._goal)

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2
        boxLeftPos = self.get_site_pos('boxLeft')
        boxRightPos = self.get_site_pos('boxRight')
        pegLeftPos = self.get_site_pos('pegLeft')
        pegRightPos = self.get_site_pos('pegRight')
        leftDist = np.linalg.norm(boxLeftPos - pegLeftPos)
        rightDist = np.linalg.norm(boxRightPos - pegRightPos)

        #totalDist = 50*(placingDistHead + leftDist + rightDist)
        # log-quadratic cost function from the GPS paper
        #reward = -(totalDist ** 4 + math.log10(totalDist ** 2 + 0.0001)) * 0.1
        # TODO using simple L2 distance for now
        totalDist = 10*placingDistHead
        reward = -(totalDist ** 2 + math.log10(totalDist ** 2 + 0.00001))
        return reward

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass

