import gym
from rlkit.core.serializable import Serializable


class BabyModeWrapper(gym.Wrapper, Serializable):

    def __init__(self, env, tasks):
        Serializable.quick_init(self, locals())
        super().__init__(env)
        self.env = env
        self.tasks = tasks

    '''
    MAML sampler api
    
    In baby mode, tasks will be reset in the environment reset
    function so we ignore task sampling/setting
    Tasks should be sampled before construnction.
    '''
    # def sample_tasks(self, meta_batch_size):
    #     return [None] * meta_batch_size
    
    # def set_task(self, task):
    #     pass

    # def log_diagnostics(self, paths, prefix):
    #     pass

    def step(self, action):
        return self.env.step(action)

    def reset_task(self, idx):
        return self.env.reset_to_idx(idx)

    def get_all_task_idx(self):
        return range(len(self.tasks))