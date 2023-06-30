# Package Description

This package is used to control the Unitree A1 robot with a pre-trained AI model. The training is done by this [repository](https://github.com/leggedrobotics/legged_gym). The main purpose is to provide an interface to use pre-trained AI models with unitree robots using unitree_ros package.

# Installation

First install unitree ros package from [github](https://github.com/unitreerobotics/unitree_ros). Then inside of the src folder clone this package.

```bash
git clone https://github.com/mertgungor/unitree_model_control.git
```

After cloning cd into the unitree_ros folder again and build the package.

```bash
catkin_make
```

Finally change the `publish_rate` parameter to 5000 at `unitree_ros/src/robots/a1_description/config/robot_control.yaml`. The model needs high publish rate to run properly, otherwise you migh see shaky behavior.

# Usage

First start the Gazebo simulation.

```bash
source devel/setup.bash
roslaunch unitree_gazebo normal.launch rname:=a1
```

In a new therminal start the AI model.

```bash
source devel/setup.bash
rosrun unitree_model_control model_run
```

Finally, in a new terminal run `teleop_keyboard_twist` node to command the robot to walk. 

```bash
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```