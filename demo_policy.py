import numpy as np
import datetime

from webcam import Webcam

from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.envs.sawyer_reach_real_env import PearlSawyerReachXYZEnv
from rlkit.envs.wrappers import NormalizedBoxEnv

video_path = 'demo/video_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".avi"
IS_ONLINE = False

ROBOT_CONFIG = 'pearl_lordi_config'
ACTION_MODE = 'position'  # position or torque - NOTE: torque safety box has not been tested
MAX_SPEED = 0.15
MAX_PATH_LENGTH = 10
STEPS_PER_EVAL = 2 * MAX_PATH_LENGTH # 5 * MAX_PATH

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

        self.agent =
        self.cap = Webcam(video_path)
        self.eval_sampler = InPlacePathSampler(
            env=self.env,
            policy=self.agent,
            max_path_length=self.max_path_length,
            )

    def mark_policy(self, target_goal):
        self.env.move_to_pos(target_goal)
        # TAKE A PICTURE

    def eval_policy(self, target_goal):
        self.cap.start_record()
        self.cap.stop_record()
        self.cap.close()

        self.env.set_goal(target_goal)  # TODO: confirm this works

        paths = self.eval_sampler.obtain_samples(deterministic=True, is_online=IS_ONLINE)
