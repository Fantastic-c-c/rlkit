from PIL import Image
import numpy as np
from rlkit.envs.half_cheetah_dir import HalfCheetahDirEnv
from rlkit.envs.ant_dir import AntDirEnv
from rlkit.envs.humanoid_dir import HumanoidDirEnv

#env = HalfCheetahDirEnv()
env = AntDirEnv()
img = np.flipud(env.sim.render(1024, 1024))
img = Image.fromarray(img)
img.save('ant.png')
#action_dim = int(np.prod(env.action_space.shape))
#action = np.zeros(action_dim)
