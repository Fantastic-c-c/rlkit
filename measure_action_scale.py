import numpy as np
import time

from rlkit.envs.sawyer_torque_reach_real import PearlSawyerReachXYZTorqueEnv
from rlkit.envs.wrappers import NormalizedBoxEnv

env = NormalizedBoxEnv(PearlSawyerReachXYZTorqueEnv())

all_dists = {}
all_other_dists = {}
for i in range(7):
    for j in range(2):
        dists = []
        other_dists = []
        action = np.zeros(7)
        action[i] = 2 * j - 1  # action is either -1 or 1
        true = env._wrapped_env.action_space.high[i]
        for reps in range(10):
            env.reset()
            # measure starting position
            init_pos = env._get_joint_angles()
            time.sleep(.5)
            for s in range(1):
                env.step(action)
            time.sleep(.5)
            final_pos = env._get_joint_angles()
            dists.append(init_pos[i] - final_pos[i])

            abs_angle_diffs = np.abs( (np.concatenate([init_pos[:i], init_pos[(i + 1):]]) -
                                       np.concatenate([final_pos[:i], final_pos[(i + 1):]]))
                                      )
            other_dists.append(np.sum(abs_angle_diffs))
        key_str = "joint{}_direction_{}_true{}".format(i, j, true)
        all_dists[key_str] = np.mean(np.asarray(dists))
        all_other_dists[key_str] = np.mean(np.asarray(other_dists))
print("All dists: " + str(all_dists))
print("All other dists: " + str(all_other_dists))
