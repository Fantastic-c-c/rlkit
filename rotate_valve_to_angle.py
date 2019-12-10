import sys
try:
    sys.path.remove("/home/abhigupta/Libraries/mujoco-py")  # needed for running valve DClaw
except:
    pass

import time
from rlkit.envs.dclaw_turn import DClawTurnEnv

# Create a hardware environment for the D'Claw turn task.
# `device_path` refers to the device port of the Dynamixel USB device.
# e.g. '/dev/ttyUSB0' for Linux, '/dev/tty.usbserial-*' for Mac OS.
env = DClawTurnEnv(device_path='/dev/ttyUSB0', frame_skip=60, n_tasks=40)

env._reset()
act = [0, -0.5, 0.6] * 3

for i in range(50):
    env.step(act)
# env._reset()
# print("Displaying 0")
# time.sleep(0.5)

# # Reset the environent and perform a random action.
# env._initial_object_pos = 3
# env._reset()
# print("DISPLAYING 1")
# time.sleep(0.5)
# env._initial_object_pos = -3
# env._reset()
# print("DISPLAYING 2")
# time.sleep(0.5)
