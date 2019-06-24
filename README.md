README last updated on: 06/20/2019

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
Note that you'll need to [get your own MuJoCo key](https://www.roboti.us/license.html) if you want to use MuJoCo.

## Real World Experiments
We are using this version of [sawyer_control](https://github.com/larrywyang/sawyer_control) on branch `reaching`.
You must run `saw` in every shell before running any script that interacts with ROS (so anything that interacts with `sawyer_control`).

### Enabling the Controller
1. `cd ~/catkin_ws`
2. Make sure you are not in a conda env! If you are, type `sd` to exit it. Then run `exp_nodes` to start the controller
3. Open a new terminal: `enable` to enable the robot
* To disable the robot, you can run `disable`
* Note: sometimes the IK controller will fail and you will see an error message. The
robot controller is still running even after this so you do not need to restart it.

### Running the script
1. run `saw`
2. run `sa pearl` to enter conda env (you are in Python 3 now)
3. `cd ~/rlkit`
4. `python launch_experiment.py configs/sawyer_reach_real_3d.json`

### Debugging
1. If ros master node cannot be found, reboot the robot's computer
2. If you can't enable, check e-stop button, try again
3. If the robot starts moving weirdly reboot exp_nodes
4. If the program hangs, it's probably because you forgot to run `saw` somewhere and your script can't communicate with ROS.

### Other Stuff
This is deprecated. See the base env in `sawyer_control` for the bounds.
To define the safety bounds of the Sawyer you can run:
1. `cd ~/ros_ws`
2. `./lordi.sh` (or whatever robot you are going to use). You need to start a
separate terminal for the script (in addition to the controller script)
3. `cd ~/Documents/sawyer_pearl`
4. `python print_cartesian_positions.py`. The coordinates of the end-effector (as well as its orientation) will be displayed.
Make sure you are looking at the end-effector position and not its orientation.
The x-value is forward and back, the y-value is left and right, the z-value is up and down.
5. Make the necessary edits to `~/ros_ws/src/sawyer_control/src/sawyer_control/configs/pearl_lordi_config.py`

If you want to define a torque safety box, you can look at `base_config` in the `sawyer_control` repor for an example.

## Visualizing a policy and seeing results
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `rlkit.launchers.config.LOCAL_LOG_DIR`. Default name is 'output'.
 - `<foldername>` is auto-generated and based off of the date.

Alternatively, if you don't want to clone all of `rllab`, a repository containing only viskit can be found [here](https://github.com/vitchyr/viskit).
Then you can similarly visualize results with.
```bash
python viskit/viskit/frontend.py LOCAL_LOG_DIR/<exp_prefix>/
```

## SAC Algorithm-Specific Comments
The SAC implementation provided here only uses Gaussian policy, rather than a Gaussian mixture model, as described in the original SAC paper.

## Credits
A lot of the coding infrastructure is based on [rllab](https://github.com/rll/rllab).
The serialization and logger code are basically a carbon copy of the rllab versions.

The Dockerfile is based on the [OpenAI mujoco-py Dockerfile](https://github.com/openai/mujoco-py/blob/master/Dockerfile).
