import numpy as np
import time
import pickle

from rlkit.envs.sawyer_peg import SawyerTorquePegInsertionEnv
from rlkit.envs.wrappers import NormalizedBoxEnv

env = NormalizedBoxEnv(SawyerTorquePegInsertionEnv())

all_dists = {}
for i in range(7):
    for j in range(2):
        action = np.zeros(7)
        action[i] = 2 * j - 1  # action is either -1 or 1
        true = env._wrapped_env.action_space.high[i]

        key_str = "joint{}_direction_{}_true{}".format(i, j, true)

        all_trajs = []
        for reps in range(10):
            action_to_joints = []
            env.reset()
            # Add starting position
            action_to_joints.append((np.zeros(7), env._get_joint_angles()))
            time.sleep(.5)
            for s in range(15):
                env.step(action)
                action_to_joints.append((action, env._get_joint_angles()))
            time.sleep(.5)
            all_trajs.append(action_to_joints)
        all_dists[key_str] = all_trajs

# open a file, where you ant to store the data
file = open('torque_joint_data.pkl', 'wb')

# dump information to that file
pickle.dump(all_dists, file)

# close the file
file.close()

print("ACTION SCALE: " + str(env.config.MAX_TORQUES))
print("All dists: " + str(all_dists))
