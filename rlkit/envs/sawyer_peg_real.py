from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from sawyer_control.coordinates import quat_2_euler, euler_2_rot, euler_2_quat
from pyquaternion import Quaternion
from rlkit.core.serializable import Serializable
import numpy as np
from gym.spaces import Box
import logging
import time

from . import register_env


@register_env('sawyer-peg-real')
class MultitaskSawyerPegEnv(Serializable, SawyerReachXYZEnv):
    def __init__(self,
            randomize_tasks=True,
            n_tasks=1,
            goal_thresh=1, # don't terminate
            **kwargs):
        Serializable.quick_init(self, locals())
        SawyerReachXYZEnv.__init__(self, config_name='laudri_peg_config', **kwargs)

        self.goal_thresh = goal_thresh
        #self.pos_control_reset_position = np.array([0.732, .0195, .267])
        # TODO make it easier by starting right above the hole
        self.pos_control_reset_position = np.array([.745, .045, .295])
        # first axis is rotation about z
        # for sawyer, orientation around y, x, z
        self.reset_orientation = euler_2_quat(0, 0, np.pi)
        #self.reset_pose = np.concatenate((self.pos_control_reset_position, self.reset_orientation))
        # TODO remove this for rotation control
        self.reset_pose = np.concatenate((self.pos_control_reset_position, np.array([.373, .928, -.006, .005])))

        # set ee position and angle safety box
        self.ee_angle_lows = np.array([-0.05, -0.05, -np.pi * .75])
        self.ee_angle_highs = np.array([0.05, 0.05, np.pi * .75])
        self.ee_pos_lows = np.array([.671, -0.038, 0.195])
        self.ee_pos_highs = np.array([.787, .090, .270])
        self.position_action_scale = .03 # max action is 3cm


        # generate random goals
        if n_tasks == 1:
            self.goals = [np.array([.762, .0540, .1973])]
            # TODO fix me
            self.goal_pose = np.array([.745, .045, .236, .373, .928, -.006, .005])

            self._goal = self.goals[0]

        else:
            raise NotImplementedError

        print('goals \n', self.goals)
        print(self._get_obs())
        #self.reset_task(0)

        # logging
        #self.logger = logging.getLogger('exp.sawyer_peg_env')

    def _set_observation_space(self):
        # obs is full 7-dof pose + endpoint velocity
        lows = np.array([.2, -.2, .03, -1, -1, -1, -1]) #, -1, -1, -1, -1, -1, -1])
        highs = np.array([.6, .2, .5, 1, 1, 1, 1]) #, 1, 1, 1, 1, 1, 1])
        #lows = np.array([.2, -.2, .03])
        #highs = np.array([.6, .2, .5])
        self.observation_space = Box(lows, highs)

    def _set_action_space(self):
        self.action_space = Box(
                -1 * np.ones(4),
                np.ones(4),
                dtype=np.float32
        )

    def _get_obs(self):
        # obs is full current ee pose: position + orientation (length 7)
        # NOTE: obs also include ee velocity! (length 6)
        _, _, ee_pose, ee_vel = self.request_observation()
        return np.concatenate([ee_pose, ee_vel])

    def _pose_from_obs(self, obs):
        return obs[:7]

    def _vel_from_obs(self, obs):
        return obs[7:]

    def _move_by(self, action):
        # action consists of (x, y, z) position and rotation about the z axis
        # determine new desired pose
        # full pose is (x, y, z) + rot quaternion
        ee_full_pose = self._pose_from_obs(self._get_obs())

        # first deal with the position part
        pos_act = action[:3] * self.position_action_scale
        ee_pos = ee_full_pose[:3]
        target_ee_pos = pos_act + ee_pos
        old_ee_pos = np.copy(target_ee_pos)
        target_ee_pos = np.clip(target_ee_pos, self.ee_pos_lows, self.ee_pos_highs)
        #self.logger.info('\n POSITION')
        print('old', ee_pos)
        print('new', target_ee_pos)
        if np.any(old_ee_pos - target_ee_pos):
            print('safety box violated, position clipped')

        # next deal with the orientation
        # scale down the rotation to within +-15 degrees
        # TODO: add other DoF here
        #print('\n ORIENTATION')
        rot_act = action[-1] * 0.262
        rot_act = np.array([0, 0, rot_act])
        # get current euler angles from the pose
        curr_eulers = np.array(quat_2_euler(ee_full_pose[3:]))
        #print('old eulers', curr_eulers)
        # compute new euler angles by adding the action
        new_eulers = curr_eulers + rot_act
        #print('new eulers', new_eulers)
        # handle angle wrap
        if new_eulers[-1] < -np.pi:
            new_eulers[-1] = new_eulers[-1] + 2 * np.pi
        elif new_eulers[-1] > np.pi:
            new_eulers[-1] = new_eulers[-1] - 2 * np.pi
        #print(new_eulers)
        # clip angles to keep them safe
        if new_eulers[-1] < 0:
            # TODO: hack to prevent double solutions
            new_eulers[-1] = -np.pi
            #new_eulers[-1] = min(new_eulers[-1], -np.pi * .75)
        elif new_eulers[-1] > 0:
            new_eulers[-1] = max(new_eulers[-1], np.pi * .75)
        #print('new eulers after wrap, clip', new_eulers)
        # compute the desired quaternion
        quat_new = euler_2_quat(*new_eulers)
        #print('curr quat', ee_full_pose[3:])
        #print('new quat', quat_new)
        # TODO no rotation for now
        quat_new = self.goal_pose[3:]
        self._move_to(target_ee_pos, quat_new)

    def _move_to(self, target_pos, target_quat):
        # combine position and orientation and send to IK
        target_ee_pose = np.concatenate((target_pos, target_quat))
        angles = self.request_ik_angles(target_ee_pose, self._get_joint_angles())
        self.request_angle_action(angles, target_ee_pose, clip_joints=False)

    def _compute_reward(self, action, pose):
        # penalize the full pose
        # compute rotation matrix between goal and curent frame
        goal_quat = Quaternion(self.goal_pose[3:])
        curr_quat = Quaternion(pose[3:])
        rotation = (curr_quat * goal_quat.inverse).rotation_matrix

        # determine translation vector
        translation = (pose[:3] - self.goal_pose[:3])[..., None]

        # construct transformation matrix in homogenous coordinates
        transformation = np.concatenate([rotation, translation], axis=1)
        assert transformation.shape == (3, 4)
        # pick three points roughly on the peg in the goal ee frame
        # (include all three directions in order to penalize the full pose)
        x = np.array([-.01, 0, -.01, 1])
        y = np.array([0, -.01, -.01, 1])
        z = np.array([0, 0, -.05, 1])

        # transform each point into the current ee frame
        x_t = transformation.dot(x)
        y_t = transformation.dot(y)
        z_t = transformation.dot(z)

        # TODO debugging save these for inspection
        points = np.array([x, y, z, x_t, y_t, z_t])

        # stack points into one vector and compute distance
        dist = np.linalg.norm(np.concatenate([x[:-1], y[:-1], z[:-1]]) - np.concatenate([x_t, y_t, z_t]))
        # use cm as base unit
        dist *= 100

        # GPS paper cost function
        gps_reward = -(1.0 * np.square(dist) + 1.0 * np.log(np.square(dist) + 1e-5))
        return gps_reward, points

    def reset(self):
        print('resetting...')
        print('from:', self._get_obs()[:3])
        print('to:', self.reset_pose[:3])
        # return to fixed ee position and orientation
        curr_error = self._pose_from_obs(self._get_obs()) - self.reset_pose
        counter = 1
        while np.any(np.abs(curr_error[:3]) > .005) or np.any(np.abs(curr_error[3:]) > .03) or counter > 200:
            self._move_to(self.reset_pose[:3], self.reset_pose[3:])
            # TODO: seems like we might need this while waiting for ROS to complete the action?
            time.sleep(.05)
            curr_error = self._pose_from_obs(self._get_obs()) - self.reset_pose
            print('curr error', curr_error)
            counter += 1
        return self._pose_from_obs(self._get_obs())

    def step(self, action):
        # for now action is 4 DOF
        print('before pose', self._get_obs()[:3])
        s = time.time()
        self._move_by(action)
        # TODO: seems like we might need this while waiting for ROS to complete the action?
        time.sleep(.05)
        e = time.time()
        obs = self._get_obs()
        pose = self._pose_from_obs(obs)
        print('after pose', pose[:3])
        # reward based on position only for now
        reward, points = self._compute_reward(action, pose)
        print('reward', reward)
        info = self._get_info()
        done = False
        info['points'] = points
        return pose, reward, done, info

    def reset_task(self, idx):
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return list(range(len(self.goals)))

    def get_tasks(self):
        return self.goals




