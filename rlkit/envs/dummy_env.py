class DummyEnv:
    def __init__(self, n_tasks, obs_space, ac_space):
        self.n_tasks = n_tasks
        self.observation_space = obs_space
        self.action_space = ac_space

    def get_all_task_idx(self):
        return range(self.n_tasks)
