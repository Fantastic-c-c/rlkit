import numpy as np


from rlkit.data_management.replay_buffer import ReplayBuffer


class PathReplayBuffer:
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._paths = list()
        self._size = 0
        self.clear()


    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError

    def add_path(self, path):
        self._paths += [path]
        self._size += len(path['observations'])

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def terminate_episode(self):
        raise NotImplementedError

    def size(self):
        return self._size

    def clear(self):
        self._size = 0
        self._paths = list()

    def sample_paths(self, num_paths):
        # returns paths so far, final timestep
        return np.random.choice(self._paths, num_paths)


    def sample_data(self, indices):
        raise NotImplementedError

    def random_batch(self, batch_size):
        raise NotImplementedError

    def random_sequence(self, batch_size):
        raise NotImplementedError

    def num_steps_can_sample(self):
        return self._size
