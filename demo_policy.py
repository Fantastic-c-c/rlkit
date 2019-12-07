import numpy as np
import datetime
import os
import torch
import time

from webcam import Webcam

from rlkit.torch.sac.policies import MakeDeterministic
from rlkit
from rlkit.core import eval_util
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.torch.sac.policies import TanhGaussianPolicy
from configs.default import default_config
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder

video_path = 'demo/video_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".avi"
image_folder = r'demo/'
model_path = "/home/justin/Documents/rlkit/output/sawyer-reach-sim-2d/2019_05_08_18_12_39/"
IS_ONLINE = False

ROBOT_CONFIG = 'pearl_lordi_config'
ACTION_MODE = 'position'  # position or torque - NOTE: torque safety box has not been tested
MAX_SPEED = 0.15
MAX_PATH_LENGTH = 50
STEPS_PER_EVAL = 2 * MAX_PATH_LENGTH # 5 * MAX_PATH

config = default_config

use_sim = True

class PolicyRunner:
    def __init__(self, num_steps_per_eval, max_path_length, use_webcam=True):
        if use_sim:
            from rlkit.envs.sawyer_xyreach_sim_env import PearlSawyerReachXYSimEnv
            self.env = NormalizedBoxEnv(PearlSawyerReachXYSimEnv())
        else:
            env = NormalizedBoxEnv(DClawPoseEnv(**env_params))


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

        if use_webcam:
            self.cap = Webcam(video_path, image_folder, cap_num=1)
        else:
            self.cap = None

        if use_sim:
            self.eval_sampler = InPlacePathSampler(
                env=self.env,
                policy=self.agent,
                max_path_length=MAX_PATH_LENGTH,
                animated=True
                )
        else:
            self.eval_sampler = InPlacePathSampler(
                env=self.env,
                policy=self.agent,
                max_path_length=MAX_PATH_LENGTH,
                )

    def mark_policy(self, target_goal):
        self.env.move_to_pos(target_goal)
        # TAKE A PICTURE
        if self.cap:
            self.cap.take_picture()
        # self.env.reset()

    def eval_policy(self, target_goal):
        self.env.reset()
        self.env.set_goal(target_goal)  # TODO: confirm this works
        print("GOAL: " + str(self.env.get_goal()))

        if self.cap:
            self.cap.start_record()

        paths = self.collect_paths()
        # paths = self.eval_sampler.obtain_samples(deterministic=True, max_samples=MAX_PATH_LENGTH, accum_context=True)
        print("Avg return: {}".format(eval_util.get_average_returns(paths)))
        print("Final return: {}".format(eval_util.get_final_return(paths)))

        if self.cap:
            self.cap.stop_record()
            self.cap.close()

    def collect_paths(self):
        paths = []
        num_transitions = 0
        num_trajs = 0
        num_steps_per_eval = MAX_PATH_LENGTH * 3
        num_exp_traj_eval = 1
        while num_transitions < num_steps_per_eval:
            path, num = self.eval_sampler.obtain_samples(deterministic=num_transitions,
                                                    max_samples=num_steps_per_eval - num_transitions,
                                                    max_trajs=1, accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal  # goal

        return paths

def main():
    print("Please close any webcam viewers")
    policyRunner = PolicyRunner(STEPS_PER_EVAL, MAX_PATH_LENGTH, use_webcam=False)
    print("Initial position: " + str(policyRunner.env._get_obs()))

    # print("MARKING GOAL")
    if use_sim:
        target_goal = np.asarray([-0.15, 0.45, 0.38])
    else:
        target_goal = np.asarray([0.63, -0.15,  0.50])
    policyRunner.mark_policy(target_goal)

    print("EVAL POLICY")
    print(policyRunner.env.reset())
    for i in range(2):
        policyRunner.eval_policy(target_goal)
    policyRunner.eval_policy(target_goal)
    print("Final position: " + str(policyRunner.env._get_obs()) + " | " + str(policyRunner.env.data.mocap_pos))

if __name__ == "__main__":
    main()
