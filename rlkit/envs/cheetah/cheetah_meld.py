import numpy as np

# from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
from gym.envs.mujoco import mujoco_env
from rlkit.envs import register_env

@register_env('cheetah-meld')
class HalfCheetahVelEnv(mujoco_env.MujocoEnv):

    def __init__(self, n_tasks=3, randomize_tasks=True, obs_mode='state'):
        ### Added by Tony
        self.obs_mode = obs_mode
        self.debug = False
        self.target_vel = 0  # placeholder
        self.truncate_vel_diff = 0.5
        self.frame_skip = 10 ### 0.01*10 = 0.1sec .... 50 steps = 5 sec per rollouts

        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', self.frame_skip)

    def get_image(self, width=64, height=64, camera_name='track'):
        # use sim.render to avoid MJViewer which doesn't seem to work without display
        img = self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )
        return np.flipud(img)

    def _get_obs(self, obs_mode=None):
        if obs_mode is None:
            obs_mode = self.obs_mode
        if obs_mode == 'image':
            img = self.get_image()
            img = img.astype(np.float32) / 255.
            return img.flatten()
        else:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):

        action*=0.8

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self.target_vel)  ### The reward is related to the target velocity
        ctrl_cost = 0.01 * np.sum(np.square(action))
        vel_error = abs(forward_vel - self.target_vel)

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        score = -vel_error
        infos = dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost, task=self.target_vel, score=score)

        if vel_error < self.truncate_vel_diff:
            sparse_reward = reward
        else:
            sparse_reward = 0


        return (observation, reward, done, infos)


    def reset(self):
        # original mujoco
        ob = self.reset_model()
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

        # set seed
        if is_eval_env:
            np.random.seed(100) #pick eval tasks as random from diff seed
        else:
            np.random.seed(101)


        max_vel = 4
        velocities = np.random.uniform(1, max_vel, size=(num_tasks,))

        self.possible_goals = velocities

        return self.possible_goals


    def set_task_for_env(self, goal):

        '''
        Call this function externally,
        to reset the task
        '''

        self.target_vel = goal

    def override_action_mode(self, action_mode):
        pass
