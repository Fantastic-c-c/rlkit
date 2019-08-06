README last updated on: 08/05/2019

# rlkit
Meta-learning with SAC in PyTorch, forked from the original [RLkit](https://github.com/vitchyr/rlkit).

Soft Actor Critic (SAC)
    - [example script](examples/sac.py)
    - [SAC paper](https://arxiv.org/abs/1801.01290)
    - [TensorFlow implementation from author](https://github.com/haarnoja/sac)

## Installation
Install and use the included Ananconda environment
```
$ conda env create -f environment.yml
$ source activate pearl
(pearl) $ python launch_experiment.py ./configs/point-robot.json
```

## Starting Experiments
We are using the [sawyer_control](https://github.com/mdalal2020/sawyer_control) repo with some changes to send velocity commands to the robot rather than positions, and to run the controller at a fixed frequency.
You must run `saw` in every shell before running any script that interacts with ROS (so anything that interacts with `sawyer_control`).
There are three things to do to run an experiment. Do them in three separate terminal panes.

#### Enable the robot
1. `cd ~/catkin_ws`
2. `saw`
3. `enable`

#### Start the server
1. `cd ~/catkin_ws`
2. `saw`
3. if you are in any kind of conda env, you must exit it! Type `sd` to exit.
4. Run `exp_nodes` to start the server. Adding logging lines like `rospy.logerr(msg)` to code will result in printing to this terminal.

#### Launch experiment script
1. `cd ~/rlkit`
2. `saw`
3. `sa pearl` to start conda env
3. `python launch_experiment.py ./configs/[CONFIG_NAME_HERE].json`

## Ending Experiments
When you are done working with the robot for the day, remember to turn it off! Turn it off by pressing and holding the power button on the robot computer until the light goes out.
When I leave it for just a few hours, I disable the robot (type `disable` in a shell).

## Debugging
1. If you press E-stop to stop the robot doing something unsafe, *re-enable the robot right away!!* If you do not do it right away, the robot can get stuck in an error mode that is very difficult to fix. To disable E-stop mode, press down on the E-stop button and twist. Then try to enable the robot. Move the robot around manually to make sure no joints are stuck.
2. If the program hangs, it's probably because you forgot to run `saw` somewhere and your script can't communicate with ROS.
3. If you changed something in `sawyer_control`, restart `exp_nodes` just to be sure.
4. If you change anything about the ROS messages, you need to re-run `catkin_make`
5. If ros master node cannot be found, reboot the robot's computer
6. If the robot starts moving weirdly reboot `exp_nodes`

## Development
The peg insertion environment is found in `envs/sawyer_peg_real.py`.
This environment inherits from `sawyer_base_env.py` in `sawyer_control`.
Other important files in `sawyer_control` are `angle_action_server.py` - contains the controller as well as the functions that send the actions to ROS.

## Credits
A lot of the coding infrastructure is based on [rllab](https://github.com/rll/rllab).
The serialization and logger code are basically a carbon copy of the rllab versions.

The Dockerfile is based on the [OpenAI mujoco-py Dockerfile](https://github.com/openai/mujoco-py/blob/master/Dockerfile).
