from sawyer_control.envs.sawyer_reaching import SawyerReachXYZEnv
from sawyer_control import transformations
from rlkit.core.serializable import Serializable
import numpy as np
from gym.spaces import Box

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
        self.pos_control_reset_position = np.array([0.6935, -0.01864, .3266])
        # first axis is rotation about z
        self.reset_orientation = transformations.quaternion_from_euler(3.13, 0, 0)
        self.reset_pose = np.concatenate((self.pos_control_reset_position, self.reset_orientation))

        # generate random goals
        if n_tasks == 1:
            self.goals = [np.array([.7501, .0639, .1973])]
            self._goal = self.goals[0]

        else:
            raise NotImplementedError
            np.random.seed(1337)
            self.goals = [1 * np.random.uniform(self.goal_low, self.goal_high) for _ in range(n_tasks)]

        print('goals \n', self.goals)
        self.reset_task(0)

    def _set_observation_space(self):
        lows = np.array([.2, -.2, .03, -1, -1, -1, -1])
        highs = np.array([.6, .2, .5, 1, 1, 1, 1])
        self.observation_space = Box(lows, highs)

    def _set_action_space(self):
        self.action_space = Box(
                -1 * np.concatenate((np.ones(3), np.array([np.pi]))),
                np.concatenate((np.ones(3), np.array([np.pi]))),
                dtype=np.float32
        )

    def _move_by(self, action):
        # action consists of (x, y, z) position and rotation about the z axis
        # determine new desired pose
        # full pose is (x, y, z) + rot quaternion
        ee_full_pose = self._get_obs()

        # first deal with the position part
        pos_act = action[:3] * self.position_action_scale
        ee_pos = ee_full_pose[:3]
        target_ee_pos = pos_act + ee_pos
        old_ee_pos = target_ee_pos
        target_ee_pos = np.clip(target_ee_pos, self.config.POSITION_SAFETY_BOX_LOWS, self.config.POSITION_SAFETY_BOX_HIGHS)
        if np.any(old_ee_pos - target_ee_pos):
            print('safety box violated')
            print(old_ee_pos)
            print(target_ee_pos)

        # next deal with the orientation
        # scale down the rotation to within +-45 degrees
        rot_act = action[-1] * 0.05 #0.785
        # convert desired rotation to a quaternion
        quat_act = transformations.quaternion_from_euler(rot_act, 0, 0)
        # compute desired quaternion by multiplying current with act
        quat_new = transformations.quaternion_multiply(quat_act, ee_full_pose[3:])
        print('new euler', transformations.euler_from_quaternion(quat_new))
        self._move_to(target_ee_pos, quat_new)

    def _move_to(self, target_pos, target_quat):
        # combine position and orientation actions, send to IK

        # TODO remove these three lines
        #f = self.config.POSITION_CONTROL_EE_ORIENTATION
        #orientation = np.array([f.x, f.y, f.z, f.w])
        #target_ee_pose = np.concatenate((target_pos, orientation))
        target_ee_pose = np.concatenate((target_pos, target_quat))
        # given desired ee pose, ik solves for joint angles
        angles = self.request_ik_angles(target_ee_pose, self._get_joint_angles())
        self.send_angle_action(angles, target_ee_pose)

    def reset(self):
        # return to fixed ee position and orientation
        curr_error = self._get_obs() - self.reset_pose
        # sometimes it takes more than one step to reset (perhaps sawyer controller has a timeout?) but always moving for a fixed number of times causes shaking
        while np.any(np.abs(curr_error) > .05):
            self._move_to(self.reset_pose[:3], self.reset_pose[3:])
            curr_error = self._get_obs() - self.reset_pose
        return self._get_obs()

    def _get_obs(self):
        # obs is full current ee pose: position + orientation
        _, _, pose = self.request_observation()
        return pose

    def step(self, action):
        # for now action is 4 DOF
        self._move_by(action)
        pose = self._get_obs()
        # reward based on position only for now
        reward = self.compute_reward(action, pose)
        print(reward)
        info = self._get_info()
        done = False
        return pose, reward, done, info

    def compute_reward(self, action, obs):
        # sort of an inverse huber here
        # see GPS paper for more details
        obs = obs[:3]
        dist = np.linalg.norm(obs - self._goal)
        return -(1.0 * np.square(dist) + 1.0 * np.log(np.square(dist) + 1e-5))

    def reset_task(self, idx):
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return list(range(len(self.goals)))

    def get_tasks(self):
        return self.goals




