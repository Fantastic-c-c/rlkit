"""
Run Prototypical Soft Actor Critic on point mass.
"""
import os
import numpy as np
import click
import datetime
import pathlib

# from rlkit.envs.point_mass import PointEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
# from rlkit.envs.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.sawyer_xyz.env_lists import MEDIUM_TRAIN_AND_TEST_LIST
from rlkit.envs.ml10_env import MediumEnv



from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy, MultiHeadTanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, MultiHeadMlp
from rlkit.torch.sac.sac import ProtoSoftActorCritic
from rlkit.torch.sac.proto import ProtoAgent
import rlkit.torch.pytorch_util as ptu



def datetimestamp(divider=''):
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f').replace('-', divider)

def experiment(variant):
    ptu.set_gpu_mode(variant['use_gpu'], variant['gpu_id'])

    env = MediumEnv(MEDIUM_TRAIN_AND_TEST_LIST, test_mode=True)
    tasks = env.get_all_task_idx()


    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    latent_dim = len(tasks)
    task_enc_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1

    num_tasks = len(tasks)
    net_size = variant['net_size']
    # start with linear task encoding
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    task_enc = encoder_model(
            hidden_sizes=[200, 200, 200], # deeper net + higher dim space generalize better
            input_size=obs_dim + action_dim + reward_dim,
            output_size=task_enc_output_dim,
    )
    qf1 = MultiHeadMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
        num_tasks=num_tasks,
    )
    qf2 = MultiHeadMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
        num_tasks=num_tasks,
    )
    vf = MultiHeadMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
        num_tasks=num_tasks,
    )
    policy = MultiHeadTanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
        num_tasks=num_tasks,
    )

    rf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1
    )

    alpha_network = FlattenMlp(
        hidden_sizes=[64, 64],
        input_size=num_tasks,
        output_size=1,
    )

    agent = ProtoAgent(
        latent_dim,
        [task_enc, policy, qf1, qf2, vf, rf],
        **variant['algo_params']
    )

    algorithm = ProtoSoftActorCritic(
        env=env,
        train_tasks=list(tasks)[:10],
        eval_tasks=list(tasks)[10:],
        nets=[agent, task_enc, policy, qf1, qf2, vf, rf, alpha_network],
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
    max_path_length = 150
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            meta_batch=2,
            num_iterations=10000,
            num_tasks_sample=2,
            num_steps_per_task=10 * max_path_length,
            num_train_steps_per_itr=1,
            num_evals=5, # number of evals with separate task encodings
            num_steps_per_eval=10 * max_path_length,  # num transitions to eval on
            batch_size=256,  # to compute training grads from
            embedding_batch_size=256,
            embedding_mini_batch_size=256,
            max_path_length=max_path_length,
            discount=0.99,
            soft_target_tau=0.005,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            context_lr=3e-4,
            reward_scale=10.,
            sparse_rewards=False,
            reparameterize=True,
            kl_lambda=.1,
            rf_loss_scale=1.,
            use_information_bottleneck=False,
            train_embedding_source='online_exploration_trajectories',
            # embedding_source should be chosen from
            # {'initial_pool', 'online_exploration_trajectories', 'online_on_policy_trajectories'}
            eval_embedding_source='online_exploration_trajectories',
            recurrent=False, # recurrent or averaging encoder
            dump_eval_paths=False,
            render_eval_paths=False,
            render=False,
            eval_per_itr=1,
        ),
        net_size=300,
        use_gpu=False,
        gpu_id=gpu,
    )

    exp_name = 'medium-test'

    log_dir = '/mounts/output' if docker == 1 else 'output'
    experiment_log_dir = setup_logger(exp_name, variant=variant, exp_id='metaworld', base_log_dir=log_dir)

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