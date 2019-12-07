import robel
import gym
    
# Create a hardware environment for the D'Claw turn task.
# `device_path` refers to the device port of the Dynamixel USB device.
# e.g. '/dev/ttyUSB0' for Linux, '/dev/tty.usbserial-*' for Mac OS.
env = gym.make('DClawPoseFixed-v0', device_path='/dev/ttyUSB0')

# Reset the environent and perform a random action.
env.reset()
env.step(env.action_space.sample())