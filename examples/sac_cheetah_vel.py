import click
import os
import pathlib

from rlkit.launchers.launch_experiment import experiment
from examples.default import make_variant

@click.command()
@click.argument('gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(gpu, docker, debug):

    max_path_length = 200
    variant = make_variant(max_path_length)

    variant['env_name'] = 'cheetah-vel'
    variant['n_train_tasks'] = 100
    variant['n_eval_tasks'] = 30

    env_params = variant['env_params']
    env_params['n_tasks'] = 130

    algo_params = variant['algo_params']
    algo_params['meta_batch'] = 10
    algo_params['num_train_steps_per_itr'] = 2000
    algo_params['num_evals'] = 1
    algo_params['embedding_batch_size'] = 100
    algo_params['embedding_mini_batch_size'] = 100
    algo_params['reward_scale'] = 5.
    algo_params['kl_lambda'] = .1
    algo_params['train_embedding_source'] ='online_exploration_trajectories'
    algo_params['resample_z'] = 2

    util_params = variant['util_params']
    util_params['gpu_id'] = gpu

    experiment(variant)

if __name__ == "__main__":
    main()
