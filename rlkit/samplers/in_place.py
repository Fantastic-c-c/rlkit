from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import MakeDeterministic


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, deterministic=False, num_samples=None, num_trajs=None, resample=1):
        """
        Obtains samples in the environment until either we reach either num_samples transitions or
        num_traj trajectories

        """
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0
        max_samp = self.max_path_length
        max_trajs = max_samp # function will default to operating based on num_samples over trajectories
        if num_samples is not None:
            max_samp = num_samples
        if num_trajs is not None:
            max_trajs = num_trajs
        while n_steps_total < max_samp and n_trajs < max_trajs:
            path = rollout(
                self.env, policy, max_path_length=self.max_path_length)
            # save the latent context that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            if n_trajs % resample == 0:
                policy.sample_z()
        return paths, n_steps_total

