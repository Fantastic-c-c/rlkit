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
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            sparse_rewards=self._sparse_rewards[indices],
        )

    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        ''' batch of trajectories '''
        # take random trajectories until we have enough
        i = 0
        indices = []
        while len(indices) < batch_size:
            # TODO hack to not deal with wrapping episodes, just don't take the last one
            start = np.random.choice(self.episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        # cut off the last traj if needed to respect batch size
        indices = indices[:batch_size]
        return self.sample_data(indices)

    def num_steps_can_sample(self):
        return self._size
