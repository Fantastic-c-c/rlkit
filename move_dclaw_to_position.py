import sys
try:
    sys.path.remove("/home/abhigupta/Libraries/mujoco-py")  # needed for running valve DClaw
except:
    pass
import numpy as np

from rlkit.envs.dclaw_pose import DClawPoseEnv

# Create a hardware environment for the D'Claw turn task.
# `device_path` refers to the device port of the Dynamixel USB device.
# e.g. '/dev/ttyUSB0' for Linux, '/dev/tty.usbserial-*' for Mac OS.
env = DClawPoseEnv(device_path='/dev/ttyUSB0', frame_skip=60, n_tasks=20)

# Reset the environent and perform a random action.
env.reset()
act = np.array([0, -np.pi / 2, np.pi / 2] * 3)

for i in range(50):
    env.step(act)
