# Enabling the Controller
1. `cd ~/ros_ws`
2. `./lordi.sh` (or whatever robot you are going to use)
3. `roslaunch ~/ros_ws/src/sawyer_control/exp_nodes.launch` To start the controller
4. Open a new terminal:
5. `rosrun intera_interface enable_robot.py -e` To enable the robot
* To disable the robot, you can run `rosrun intera_interface enable_robot.py -d` 
* Note: sometimes the IK controller will fail and you will see an error message. The
robot controller is still running even after this so you do not need to restart it.

# Running the script
1. `cd ~/ros_ws`
2. `./lordi.sh` (or whatever robot you are going to use). You need to start a 
separate terminal for the script (in addition to the controller script)
3. `cd ~/Documents/sawyer_pearl`
4. `python3 sawyer_reach_PEARL.py`

# Other Stuff
To define the safety bounds of the Sawyer you can run:
1. `cd ~/ros_ws`
2. `./lordi.sh` (or whatever robot you are going to use). You need to start a 
separate terminal for the script (in addition to the controller script)
3. `cd ~/Documents/sawyer_pearl`
4. `python print_cartesian_positions.py`. The coordinates of the end-effector (as well as its orientation) will be displayed.
Make sure you are looking at the end-effector position and not its orientation.
The x-value is forward and back, the y-value is left and right, the z-value is up and down. 
5. Make the necessary edits to `~/ros_ws/src/sawyer_control/src/sawyer_control/configs/pearl_lordi_config.py`

If you want to define a torque safety box, you can look at `base_config` for an example.
