import numpy as np

from rlkit.data_management.replay_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation
        self._advance()

    def terminate_episode(self):
        pass

    def size(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices):
        return dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )

    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        ''' batch of transitions in order '''
        # TODO should it always start at beginning of episode?
        idx = int(np.random.randint(0, self._size - batch_size, 1))
        indices = list(range(idx, idx + batch_size))
        return self.sample_data(indices)

    def all_data(self, starts=True):
        data = self.sample_data(list(range(0, self._size)))
        if starts:
            data['starts'] = self._episode_starts
        return data

    def num_steps_can_sample(self):
        return self._size
