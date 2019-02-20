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

    max_path_length = 20
    variant = make_variant(max_path_length)

    variant['env_name'] = 'sparse-point-robot'
    variant['n_train_tasks'] = 80
    variant['n_eval_tasks'] = 20

    env_params = variant['env_params']
    env_params['n_tasks'] = 100
    env_params['goal_radius'] = 0.2

    algo_params = variant['algo_params']
    algo_params['num_tasks_sample'] = 10
    algo_params['num_evals'] = 3
    algo_params['embedding_batch_size'] = 1024
    algo_params['embedding_mini_batch_size'] = 1024
    algo_params['discount'] = 0.90
    algo_params['reward_scale'] = 100.
    algo_params['sparse_rewards'] = True
    algo_params['kl_lambda'] = .1
    algo_params['train_embedding_source'] ='online_exploration_trajectories'
    algo_params['dump_eval_paths'] = True

    util_params = variant['util_params']
    util_params['gpu_id'] = gpu

    experiment(variant)

if __name__ == "__main__":
    main()
