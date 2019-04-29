import numpy as np
import datetime
import os
import torch

from webcam import Webcam

from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.envs.sawyer_reach_real_env import PearlSawyerReachXYZEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.torch.sac.policies import TanhGaussianPolicy
from configs.default import default_config
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder

video_path = 'demo/video_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".avi"
model_path = ""
IS_ONLINE = False

ROBOT_CONFIG = 'pearl_lordi_config'
ACTION_MODE = 'position'  # position or torque - NOTE: torque safety box has not been tested
MAX_SPEED = 0.15
MAX_PATH_LENGTH = 10
STEPS_PER_EVAL = 2 * MAX_PATH_LENGTH # 5 * MAX_PATH

config = default_config

class PolicyRunner:
    def __init__(self, num_steps_per_eval, max_path_length):
        self.env = NormalizedBoxEnv(PearlSawyerReachXYZEnv(config_name=ROBOT_CONFIG,
                                                  action_mode=ACTION_MODE,
                                                  max_speed=MAX_SPEED,
                                                  position_action_scale=1/7,
                                                  height_2d=None,

                                                  reward_type='hand_distance',
                                                  goal_low=np.array([0.45, -0.3, 0.2]),
                                                  goal_high=np.array([0.65, 0.3, 0.4]),
                                                  ))

        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        latent_dim = config['latent_size']
        reward_dim = 1

        net_size = config['net_size']
        recurrent = config['algo_params']['recurrent']

        context_encoder = latent_dim * 2 if config['algo_params']['use_information_bottleneck'] else latent_dim
        encoder_model = RecurrentEncoder if recurrent else MlpEncoder
        context_encoder = encoder_model(
            hidden_sizes=[200, 200, 200],
            input_size=obs_dim + action_dim + reward_dim,
            output_size=context_encoder,
        )

        policy = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim + latent_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )
        self.agent = PEARLAgent(
            latent_dim,
            context_encoder,
            policy,
            **config['algo_params']
        )

        policy.load_state_dict(torch.load(os.path.join(model_path, 'policy.pth')))
        context_encoder.load_state_dict(torch.load(os.path.join(model_path, 'context_encoder.pth')))

        self.cap = Webcam(video_path)
        self.eval_sampler = InPlacePathSampler(
            env=self.env,
            policy=self.agent,
            max_path_length=MAX_PATH_LENGTH,
            )

    def mark_policy(self, target_goal):
        self.env.move_to_pos(target_goal)
        # TAKE A PICTURE

    def eval_policy(self, target_goal):
        self.env.reset()
        self.env.set_goal(target_goal)  # TODO: confirm this works
        print("GOAL: " + str(self.env.get_goal()))

        self.cap.start_record()

        paths = self.eval_sampler.obtain_samples(deterministic=True)

        self.cap.stop_record()
        self.cap.close()
