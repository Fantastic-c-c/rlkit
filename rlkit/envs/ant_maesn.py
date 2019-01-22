import numpy as np
import pickle

from rlkit.envs.mujoco_env import MujocoEnv
from rlkit.envs.ant_multitask_base import MultitaskAntEnv

# https://github.com/RussellM2020/maesn_suite/blob/master/maesn/rllab/envs/mujoco/ant_env_rand_goal_ring.py

class AntEnvRandGoalRing(MultitaskAntEnv):

    def __init__(self, *args, **kwargs):
        self.circle_radius = 2.0
        self.goal_radius = 0.8
        super(AntEnvRandGoalRing, self).__init__(*args, **kwargs)

    def sparsify_rewards(self, reward, error):
        ''' sparsify goal reward based on distance from dense reward region around goal '''
        assert len(reward.shape) == 2
        assert len(error.shape) == 2
        # assume the inputs are batch x value
        def sparsify(r, e):
            if np.linalg.norm(e) > self.goal_radius:
                dense_reward = self.dense_goal_reward(e)
                sparse_reward = -self.circle_radius + 4.0
                r = r - dense_reward + sparse_reward
            return r
        new_rewards = []
        for i in range(reward.shape[0]):
            new_rewards.append(sparsify(reward[i], error[i]))
        new_rewards = np.stack(new_rewards)
        return new_rewards

    def dense_goal_reward(self, error):
        ''' dense goal reward is L1 distance '''
        return -np.abs(error).sum() + 4.0

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        com = np.array(self.get_body_com("torso"))
        goal_reward = self.dense_goal_reward(com[:2] - self._goal)
        ctrl_cost = 0.1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        # return the error here so we can sparsify reward later if needed
        return ob, reward, done, dict(
            error=(com[:2] - self._goal),
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        angle = np.random.uniform(0, np.pi, size=(num_tasks,))
        xpos = self.circle_radius*np.cos(angle)
        ypos = self.circle_radius*np.sin(angle)
        goals = np.stack([xpos, ypos], axis=1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
