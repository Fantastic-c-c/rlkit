import joblib
import pdb


"""
Script to evaluate a trained PEARL policy.

Loads saved params from params.pkl file, and runs the meta-learning training and
evaluation.

"""
import os
import numpy as np
import click
import datetime
import pathlib
from gym.envs.mujoco import HalfCheetahEnv

from rlkit.envs.half_cheetah_vel import HalfCheetahVelEnv
from rlkit.envs.point_mass import PointEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.sac import ProtoSoftActorCritic
from rlkit.torch.sac.proto import ProtoAgent
import rlkit.torch.pytorch_util as ptu

from multiworld.envs.mujoco.sawyer_xyz.sawyer_hand_insert import SawyerHandInsertEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_sweep import SawyerSweepEnv


def datetimestamp(divider=''):
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f').replace('-', divider)

def experiment(variant):
    task_params = variant['task_params']
    params = joblib.load('/home/deirdre/rlkit/output/metaworld/metaworld-test/params.pkl')

    # env.render()
    ptu.set_gpu_mode(variant['use_gpu'], variant['gpu_id'])

    tasks = []
    # Initialize copies of each environment with random goals.
    for _ in range(1):
        tasks.append(SawyerHandInsertEnv())
    for _ in range(1):
        tasks.append(SawyerSweepEnv())

    obs_dim = int(np.prod(tasks[0].observation_space.shape))
    action_dim = int(np.prod(tasks[0].action_space.shape))
    latent_dim = 5
    task_enc_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1

    net_size = variant['net_size']
    # start with linear task encoding
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    task_enc = params['task_enc']
    qf1 = params['qf1']
    qf2 = params['qf2']
    vf = params['vf']
    policy = params['policy']
    rf = params['rf']
    agent = params['exploration_policy']

    algorithm = ProtoSoftActorCritic(
        envs=tasks,
        train_tasks=list(tasks),
        eval_tasks=list(tasks),
        nets=[agent, task_enc, policy, qf1, qf2, vf, rf],
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
    max_path_length = 100
    # noinspection PyTypeChecker
    variant = dict(
        task_params=dict(
            n_tasks=2,
            randomize_tasks=True,
        ),
        algo_params=dict(
            meta_batch=16,
            num_iterations=10000,
            num_tasks_sample=5,
            num_steps_per_task=1 * max_path_length,
            num_train_steps_per_itr=1,
            num_evals=5, # number of evals with separate task encodings
            num_steps_per_eval=3 * max_path_length,  # num transitions to eval on
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
            rf_loss_scale=1.,
            use_information_bottleneck=True,
            train_embedding_source='online_exploration_trajectories',
            # embedding_source should be chosen from
            # {'initial_pool', 'online_exploration_trajectories', 'online_on_policy_trajectories'}
            eval_embedding_source='online_exploration_trajectories',
            recurrent=False, # recurrent or averaging encoder
            dump_eval_paths=False,
            task_idx_for_render=0,
        ),
        net_size=300,
        use_gpu=True,
        gpu_id=gpu,
    )


    exp_name = 'metaworld_videos'

    log_dir = '/mounts/output' if docker == 1 else 'output'
    experiment_log_dir = setup_logger(exp_name, variant=variant, exp_id='point-mass', base_log_dir=log_dir)

    # creates directories for pickle outputs of trajectories (point mass)
    pickle_dir = experiment_log_dir + '/eval_trajectories'
    pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)
    variant['algo_params']['output_dir'] = pickle_dir

    # debugging triggers a lot of printing
    DEBUG = 0
    os.environ['DEBUG'] = str(DEBUG)

    experiment(variant)

if __name__ == "__main__":
    main()
