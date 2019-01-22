import numpy as np

from rlkit.envs.mujoco_env import MujocoEnv


class WheeledEnv(MujocoEnv):

    def __init__(self, task={}, n_tasks=1, **kwargs):
        self.circle_radius = 2.0
        self.goal_radius = 0.8
        xml_path = 'wheeled.xml'
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal = self.tasks[0]['goal']
        super().__init__(
            xml_path,
            frame_skip=1,
            automatically_set_obs_and_action_space=True,
        )

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector
        self.reset()

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
        ])

    def reset_model(self):
        qpos  = self.init_qpos + \
                        np.random.normal(size=self.init_qpos.shape) * 0.01
        qvel = self.init_qvel + \
                        np.random.normal(size=self.init_qvel.shape) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        next_obs = self._get_obs()
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        error = next_obs[:2] - self._goal
        reward = self.dense_goal_reward(error) - ctrl_cost
        done = False
        return next_obs, reward, done, {'error': error}

    def dense_goal_reward(self, error):
        ''' dense goal reward is L2 distance '''
        return -np.linalg.norm(error)

    def sparsify_rewards(self, reward, error):
        ''' sparsify goal reward based on distance from dense reward region around goal '''
        assert len(reward.shape) == 2
        assert len(error.shape) == 2
        # assume the inputs are batch x value
        def sparsify(r, e):
            if np.linalg.norm(e) > self.goal_radius:
                dense_reward = self.dense_goal_reward(e)
                sparse_reward = -self.circle_radius
                r = r - dense_reward + sparse_reward
            return r
        new_rewards = []
        for i in range(reward.shape[0]):
            new_rewards.append(sparsify(reward[i], error[i]))
        new_rewards = np.stack(new_rewards)
        return new_rewards
