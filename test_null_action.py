import numpy as np

from rlkit.envs.sawyer_torque_reach_real import PearlSawyerReachXYZTorqueEnv
from rlkit.envs.wrappers import NormalizedBoxEnv

env = NormalizedBoxEnv(PearlSawyerReachXYZTorqueEnv())

action = np.zeros(7)
for i in range(15):
    env.step(action)
