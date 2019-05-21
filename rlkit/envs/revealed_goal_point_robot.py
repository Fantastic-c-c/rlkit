import numpy as np
from gym import spaces
from gym import Env

from . import register_env


@register_env('revealed-point-robot')
class RevealedGoalPointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, randomize_tasks=False, n_tasks=2):

        self.goal = self.reset_task()
        goals = [np.array([-.75, .25])]
        self.goals = goals

        self.reset_task(0)
        self._show_task = True
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, idx=0):
        ''' reset goal AND reset the agent '''
        theta = np.random.uniform(2*np.pi)

        self._goal = np.array([np.cos(theta), np.sin(theta)]) # np.random.uniform(size=2)
        self.reset()

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
        # self._state = np.random.uniform(-1., 1., size=(2,))
        self._state = np.zeros(2)
        self._show_task = True
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.concatenate([np.copy(self._state), self._show_task * np.copy(self._goal)])

    def _get_state(self):
        return np.copy(self._state)

    def step(self, action):
        # self._show_task = False
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(state=np.concatenate([np.copy(self._state), self._show_task * np.copy(self._goal)]))

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)


@register_env('sparse-revealed-point-robot')
class SparseRevealedGoalPointEnv(RevealedGoalPointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, goal_radius=0.2):
        super().__init__(randomize_tasks, n_tasks)
        self.goal_radius = goal_radius

        if randomize_tasks:
            np.random.seed(1337)
            radius = 1.0
            angles = np.linspace(0, np.pi, num=n_tasks)
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            goals = np.stack([xs, ys], axis=1)
            np.random.shuffle(goals)
            goals = goals.tolist()

        self.goals = goals
        self.reset_task(0)

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d
