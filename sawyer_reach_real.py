#!/usr/bin/python3
"""
Run Prototypical Soft Actor Critic on real-world sawyer robot.

"""
import os
import numpy as np
import click
import datetime
import pathlib

from rlkit.envs.sawyer_reach_real_env import PearlSawyerReachXYZEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import ProtoSoftActorCritic
from rlkit.torch.sac.proto import ProtoAgent
import rlkit.torch.pytorch_util as ptu


RANDOMIZE_TASKS = True

NUM_TASKS = 40
ROBOT_CONFIG = 'pearl_lordi_config'
ACTION_MODE = 'position'  # position or torque - NOTE: torque safety box has not been tested
MAX_SPEED = 0.15
MAX_PATH_LENGTH = 10
INITIAL_STEPS = 50
NUM_TASKS_SAMPLE = 8
META_BATCH = 8
STEPS_PER_TASK = 2 * MAX_PATH_LENGTH # 5 * MAX_PATH

def datetimestamp(divider=''):
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f').replace('-', divider)

def experiment(variant):
    env = NormalizedBoxEnv(PearlSawyerReachXYZEnv(config_name = ROBOT_CONFIG,
                                                  action_mode = ACTION_MODE,
                                                  max_speed = MAX_SPEED,
                                                  position_action_scale = 1/7,
                                                  height_2d=None,

                                                  reward_type='hand_distance',
                                                  goal_low=np.array([0.45, -0.3, 0.2]),
                                                  goal_high=np.array([0.65, 0.3, 0.4]),
                                                  **variant['task_params']))
    ptu.set_gpu_mode(variant['use_gpu'], variant['gpu_id'])

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    latent_dim = 5
    task_enc_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1

    net_size = variant['net_size']
    # start with linear task encoding
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    task_enc = encoder_model(
            hidden_sizes=[200, 200, 200], # deeper net + higher dim space generalize better
            input_size=obs_dim + action_dim + reward_dim,
            output_size=task_enc_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = ProtoAgent(
        latent_dim,
        [task_enc, policy, qf1, qf2, vf],
        **variant['algo_params']
    )

    algorithm = ProtoSoftActorCritic(
        render = False, # whether we wnt to render or not
        env=env,
        train_tasks=list(tasks[:int(NUM_TASKS * 0.8)]),
        eval_tasks=list(tasks[int(NUM_TASKS * 0.8):]),
        nets=[agent, task_enc, policy, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.to()
    algorithm.train()


@click.command()
@click.argument('gpu', default=0)
@click.option('--docker', default=0)
def main(gpu, docker):
    max_path_length = MAX_PATH_LENGTH
    # noinspection PyTypeChecker
    variant = dict(
        task_params=dict(
            n_tasks=NUM_TASKS,
            randomize_tasks=RANDOMIZE_TASKS,
        ),
        algo_params=dict(
            num_initial_steps=INITIAL_STEPS,
            initial_data_path='initial_data40-3d.pkl',

            meta_batch=16,
            num_iterations=10000,
            num_tasks_sample=NUM_TASKS_SAMPLE,
            num_steps_per_task=STEPS_PER_TASK,
            num_train_steps_per_itr=1000,
            num_evals=1, # number of evals with separate task encodings
            num_steps_per_eval=2 * max_path_length,  # num transitions to eval on
            batch_size=256,  # to compute training grads from
            embedding_batch_size=64,
            embedding_mini_batch_size=64,
            max_path_length=max_path_length,
            discount=0.99,
            soft_target_tau=0.005,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            context_lr=3e-4,
            reward_scale=100.,
            sparse_rewards=False,
            reparameterize=True,
            kl_lambda=.1,
            use_information_bottleneck=True,
            train_embedding_source='online_exploration_trajectories',
            # embedding_source should be chosen from
            # {'initial_pool', 'online_exploration_trajectories', 'online_on_policy_trajectories'}
            eval_embedding_source='online_exploration_trajectories',
            recurrent=False, # recurrent or averaging encoder
            dump_eval_paths=False,
        ),
        net_size=300,
        use_gpu=True,
        gpu_id=gpu,
    )

    exp_id = 'sawyer-real-reach'
    exp_name = 'proto-sac-' + exp_id + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    log_dir = '/mounts/output' if docker == 1 else 'output'
    experiment_log_dir = setup_logger(exp_name, variant=variant, exp_id=exp_id, base_log_dir=log_dir)

    # creates directories for pickle outputs of trajectories (point mass)
    pickle_dir = experiment_log_dir + '/eval_trajectories'
    pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # debugging triggers a lot of printing
    DEBUG = 0
    os.environ['DEBUG'] = str(DEBUG)

    experiment(variant)

if __name__ == "__main__":
    main()
