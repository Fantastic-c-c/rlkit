from rlkit.envs.dclaw_pose import DClawPoseEnv

# Create a hardware environment for the D'Claw turn task.
# `device_path` refers to the device port of the Dynamixel USB device.
# e.g. '/dev/ttyUSB0' for Linux, '/dev/tty.usbserial-*' for Mac OS.
env = DClawPoseEnv(device_path='/dev/ttyUSB1', frame_skip=60)

# Reset the environent and perform a random action.
env.reset()
act = [0.33216093, -0.34648382, 1.05373163, 0.3457113, -0.75016362, 0.01385098,
                 0.04234343, -0.35791852, -0.56737231]

for i in range(50):
    env.step(act)
