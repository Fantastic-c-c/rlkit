import numpy as np
import pickle

from rlkit.envs.mujoco_env import MujocoEnv

# https://github.com/RussellM2020/maesn_suite/blob/master/maesn/rllab/envs/mujoco/ant_env_rand_goal_ring.py

class AntEnvRandGoalRing(MultiTaskAntEnv):

    def __init__(self, task={}, n_tasks=200, **kwargs)
        super(AntEnvRandGoalRing, self).__init__(*args, **kwargs)
        self.goal_radius = 2.0

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        com = np.array(self.get_body_com("torso"))
        if self.sparse and np.linalg.norm(com[:2] - self._goal) > 0.8:
            goal_reward = -self.goal_radius + 4.0
        else:
            goal_reward = -np.sum(np.abs(com[:2] - self._goal)) + 4.0 # make it happy, not suicidal

        ctrl_cost = 0.1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1))),
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        angle = np.random.uniform(0, np.pi, size=(num_goals,))
        xpos = self.goal_radius*np.cos(angle)
        ypos = self.goal_radius*np.sin(angle)
        goals = np.stack([xpos, ypos], axis=1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
