from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box
from . import register_env


from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from pyquaternion import Quaternion
from multiworld.envs.mujoco.utils.rotation import euler2quat


@register_env('testing-env')
class SawyerPegInsertionTopdown6DOFEnv(SawyerXYZEnv):
    def __init__(
            self,
            hand_low=(-0.5, 0.40, 0.05),
            hand_high=(0.5, 1, 0.5),
            obj_low=(-0.1, 0.6, 0.03),
            obj_high=(0.1, 0.7, 0.03),
            random_init=False,
            tasks = [{'goal': np.array([0, 0.6, 0.09]), 'obj_init_pos':np.array([0, 0.6, 0.3])}], ## if want to change the depth of the goal without changing the pos of the box, change xml instead
            goal_low=(-0.05, 0.6, 0.09),
            goal_high=(0.05, 0.7, 0.09),
            hand_init_pos = (0, 0.65, 0.3),  # hand_init_pos y axis should be +0.05 bigger than y of goal to be directly above, 0.24 when peg is just above the box
            liftThresh = 0.04,
            rotMode='fixed',#'fixed',
            rewMode='orig',
            multitask=False,
            multitask_num=1,
            if_render=False,
            n_tasks=10,
            randomize_tasks=True,  ## randomize_tasks si currently passed in from the configs file; if True, generate tasks randomly, else use the tasks above
            **kwargs
    ):
        self.quick_init(locals())
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            # **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low

        if goal_low is None:
            goal_low = self.hand_low

        if obj_high is None:
            obj_high = self.hand_high

        if goal_high is None:
            goal_high = self.hand_high

        self.random_init = random_init
        self.liftThresh = liftThresh
        self.max_path_length = 200#200#150
        # self.tasks = tasks
        self.num_tasks = len(tasks)
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.hand_init_pos = np.array(hand_init_pos)
        self.multitask = multitask
        self.multitask_num = multitask_num
        self._state_goal_idx = np.zeros(self.multitask_num)
        self.if_render = if_render
        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            self.action_rot_scale = 1./50
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
        self.obj_and_goal_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        if not multitask:
            self.observation_space = Box(
                    np.hstack((self.hand_low, obj_low, obj_low)),
                    np.hstack((self.hand_high, obj_high, obj_high)),
            )
        else:
            self.observation_space = Box(
                    np.hstack((self.hand_low, obj_low, goal_low, np.zeros(multitask_num))),
                    np.hstack((self.hand_high, obj_high, goal_high, np.zeros(multitask_num))),
            )



        ####################generate and save generated tasks############################
        self.goal_low = goal_low
        self.goal_high = goal_high
        init_task_idx = 0

        directions = list(range(n_tasks))

        if randomize_tasks:
            # goals = self.sample_goals(n_tasks)
            goals = [1 * np.random.uniform(self.goal_low, self.goal_high) for _ in directions]
        else:
            # add more goals in n_tasks > 7
            goals = [x['goal'] for x in tasks
            ]  #use the task from the input
            if (n_tasks > len(goals)):
                raise NotImplementedError("We don't have enough goals defined")
        self.goals = np.asarray(goals)
        self.tasks = [{'direction': direction} for direction in directions]

        # set the initial goal
        self.reset_task(init_task_idx)


        # self.reset()



    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
    }

    @property
    def model_name(self):

        return get_asset_full_path('sawyer_xyz/sawyer_peg_insertion_topdown.xml')
        #return get_asset_full_path('sawyer_xyz/pickPlace_fox.xml')

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

    def reset_goal(self, direction):
        return self.goals[direction]

    def reset_task(self, idx):   ## need to change _state_goal
        self._task = self.tasks[idx]
        self._goal = self.reset_goal(self._task['direction'])

        self.set_goal(self._goal)

        self.sim.model.body_pos[self.model.body_name2id('box')] = self._state_goal  # np.array(task['goal'])

        self._state_goal = self.sim.model.site_pos[self.model.site_name2id('hole')] + self.sim.model.body_pos[
            self.model.body_name2id('box')]        ## z coordinate of hole: -0.06

        self.obj_init_pos = self.get_body_com('peg')
        if self.random_init:
            goal_pos = np.random.uniform(
                self.obj_and_goal_space.low,
                self.obj_and_goal_space.high,
                size=(self.obj_and_goal_space.low.size),
            )
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = np.random.uniform(
                    self.obj_and_goal_space.low,
                    self.obj_and_goal_space.high,
                    size=(self.obj_and_goal_space.low.size),
                )
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            self.sim.model.body_pos[self.model.body_name2id('box')] = goal_pos[-3:]
            self._state_goal = self.sim.model.site_pos[self.model.site_name2id('hole')] + self.sim.model.body_pos[
                self.model.body_name2id('box')]
        # self._set_obj_xyz(self.obj_init_pos)

        self.reset()

    def set_goal(self, goal):
        self._state_goal = goal
        # self._set_goal_marker(self._state_goal)  is this necessary??

    def reset(self):
        return self.reset_model()


    def step(self, action):
        # action[0] = 0
        # action[1] = 0
        # action[2] = 0
        # action[3] = 0

        print("------------------------------------------------------------")
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
        # print("rotation: ", action[3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward , reachRew, reachDist, pickRew, placeRew , placingDist = self.compute_reward(action, obs_dict, mode = self.rewMode)
        self.curr_path_length +=1
        #info = self._get_info()
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        # return ob, reward, done, { 'reachRew':reachRew, 'reachDist': reachDist, 'pickRew':pickRew, 'placeRew': placeRew, 'epRew' : reward, 'placingDist': placingDist}
        return ob, reward, done, {'reachDist': reachDist, 'pickRew':pickRew, 'epRew' : reward, 'goalDist': placingDist, 'success': float(placingDist <= 0.07)}

    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos =  self.get_body_com('peg')
        flat_obs = np.concatenate((hand, objPos))
        if self.multitask:
            assert hasattr(self, '_state_goal_idx')
            return np.concatenate([
                    flat_obs,
                    self._state_goal,
                    self._state_goal_idx
                ])
        return np.concatenate([
                flat_obs,
                self._state_goal
            ])

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        objPos =  self.get_body_com('peg')
        flat_obs = np.concatenate((hand, objPos))
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=objPos,
        )

    def _get_info(self):
        pass

    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
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

    def sample_goals(self, batch_size):
        #Required by HER-TD3
        goals = []
        for i in range(batch_size):
            task = self.tasks[np.random.randint(0, self.num_tasks)]
            goals.append(task['goal'])
        return {
            'state_desired_goal': goals,
        }


    def sample_task(self):
        task_idx = np.random.randint(0, self.num_tasks)
        return self.tasks[task_idx]   # return a directionary containing an index (direction)


    def reset_model(self):   ##
        self._reset_hand()
        # task = self.sample_task()
        self.obj_init_pos = self.get_body_com('peg')
        self.objHeight = self.get_body_com('peg').copy()[2]
        self.heightTarget = self.objHeight + self.liftThresh
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget
        self.target_reward = 1000*self.maxPlacingDist + 1000*2
        self.curr_path_length = 0
        #Can try changing this
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            from multiworld.envs.env_util import quat_to_zangle, zangle_to_quat

            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

            ##########################init rotate around z axis#######################
            # zangle_delta = np.pi/4
            # new_mocap_zangle = quat_to_zangle(self.data.mocap_quat[0]) + zangle_delta
            #
            # new_mocap_zangle = np.clip(
            #     new_mocap_zangle,
            #     -3.0,
            #     3.0,
            # )
            # if new_mocap_zangle < 0:
            #     new_mocap_zangle += 2 * np.pi
            #
            # self.data.set_mocap_quat('mocap', zangle_to_quat(new_mocap_zangle))
            #########################################################################

            self.do_simulation([-1, 1], self.frame_skip)


            #self.do_simulation(None, self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')

        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obsBatch):
        #Required by HER-TD3
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, actions, obs, mode='orig'):
        if isinstance(obs, dict):
            obs = obs['state_observation']

        objPos = obs[3:6]
        pegHeadPos = self.get_site_pos('pegHead')

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        placingGoal = self._state_goal

        reachDist = np.linalg.norm(objPos - fingerCOM)

        placingDistHead = np.linalg.norm(pegHeadPos - placingGoal)
        placingDist = np.linalg.norm(objPos - placingGoal)

        boxLeftPos = self.get_site_pos('boxLeft')
        boxRightPos = self.get_site_pos('boxRight')

        pegLeftPos = self.get_site_pos('pegLeft')
        pegRightPos = self.get_site_pos('pegRight')

        leftDist = np.linalg.norm(boxLeftPos - pegLeftPos)
        rightDist = np.linalg.norm(boxRightPos - pegRightPos)

        print("boxLeftPos: ", boxLeftPos, "     boxRightPos: ", boxRightPos)
        print("pegLeftPos: ", pegLeftPos, "     pegRightPos: ", pegRightPos)
        print("objPos: ", objPos, "  pegHead: ", pegHeadPos, "  placingGoal: ", placingGoal)

        def reachReward():
            # reachDistxy = np.linalg.norm(np.concatenate((objPos[:-1], [self.init_fingerCOM[-1]])) - fingerCOM)
            # if reachDistxy < 0.05: #0.02
            #     reachRew = -reachDist + 0.1
            #     if reachDist < 0.05:
            #         reachRew += max(actions[-1],0)/50
            # else:
            #     reachRew =  -reachDistxy
            # return reachRew , reachDist
            reachRew = -reachDist# + min(actions[-1], -1)/50
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - heightTarget)
            if reachDistxy < 0.04: #0.02
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - 2*zRew
            # reachRew = -reachDist
            #incentive to close fingers when reachDist is small
            if reachDist < 0.04:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew , reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if objPos[2] >= (heightTarget- tolerance):
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True


        def objDropped():
            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02)
            # Object on the ground, far away from the goal, and from the gripper
            #Can tweak the margin limits

        def objGrasped(thresh = 0):
            sensorData = self.data.sensordata
            return (sensorData[0]>thresh) and (sensorData[1]> thresh)

        def orig_pickReward():
            # hScale = 50
            hScale = 100
            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
            # elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
            elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def general_pickReward():
            hScale = 50
            if self.pickCompleted and objGrasped():
                return hScale*heightTarget
            elif objGrasped() and (objPos[2]> (self.objHeight + 0.005)):
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeReward():
            # c1 = 1000 ; c2 = 0.03 ; c3 = 0.003
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001

            return (100 ** (1/(100 * objPos[2]))) * (1/placingDist)
            return 1000*(self.maxPlacingDist - placingDistHead) + c1*(np.exp(-(placingDistHead**2)/c2) + np.exp(-(placingDistHead**2)/c3))

        reachRew, reachDist = reachReward()
        if mode == 'general':
            pickRew = general_pickReward()
        else:
            pickRew = orig_pickReward()
        placeRew = placeReward()
        # assert ((placeRew >=0) and (pickRew>=0))

        import math
        totalDist = placingDistHead + leftDist + rightDist
        reward = -(50*totalDist ** 4 + math.log10(50*totalDist ** 2 + 0.0001)) * 0.1
        print("leftDist: ", leftDist, "    rightDist: ", rightDist, "    placingDistHead: ", placingDistHead)
        print("totalDist: ", totalDist)

        print("*** ",reward, " ***")
        return [reward, reachRew, reachDist, pickRew, placeRew, placingDist]

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass

if __name__ == '__main__':
    import time
    env = SawyerPegInsertionTopdown6DOFEnv(random_init=True)
    for _ in range(1000):
        env.reset()
        # env._set_obj_xyz(np.array([0, 0.88, 0.04]))
        for _ in range(50):
            env.render()
            env.step(env.action_space.sample())
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
