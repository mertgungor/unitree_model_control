# Package Description

This package is used to control the Unitree A1 robot with a pre-trained AI model. The training is done by this [repository](https://github.com/leggedrobotics/legged_gym). The main purpose is to provide an interface to use pre-trained AI models with unitree robots using unitree_ros package.

# Installation

To run the robot from the terminal teleop-twist-keyboard package is required.

```bash
sudo apt install ros-noetic-teleop-twist-keyboard
```

Model depends on libtorch and it needs to be installed too.
```bash
cd ~/Downloads
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
```
Unzip the folder and inside bashrc add this command with the path of your libtorch installation. `export Torch_DIR=/path/to/your/torchlib`

Install unitree ros package. 

```bash
git clone https://github.com/mertgungor/unitree_ros.git ~/unitree_ros/src --recurse-submodules
```

After cloning cd into the unitree_ros and install model control package.

```bash
cd ~/unitree_ros/src
git clone https://github.com/mertgungor/unitree_model_control.git
```

Build the workspace
```bash
cd ~/unitree_ros
catkin_make
```

Finally inside `unitree_model_control/models` folder there is a file called policy_1.pt. Copy the path of this file and past at line 77 in model_node.cpp file inside `unitree_model_control/src` folder.

# Usage

First start the Gazebo simulation.

```bash
cd ~/unitree_ros
source devel/setup.bash
roslaunch unitree_gazebo normal.launch rname:=a1
```

In a new therminal start the AI model.

```bash
cd ~/unitree_ros
source devel/setup.bash
rosrun unitree_model_control model_run
```

Finally, in a new terminal run `teleop_keyboard_twist` node to command the robot to walk. 

```bash
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```