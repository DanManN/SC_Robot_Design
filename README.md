# SC Robot Design Project
A robot system that guides people to their destination.

# Setup
1. Get [Ubuntu 22.04 LTS](https://ubuntu.com/download/desktop); Consider using...
	- [virtual machine](https://www.virtualbox.org/wiki/Linux_Downloads)
	- [docker](https://hub.docker.com/layers/library/ros/humble/images/sha256-54d8ab351fd3fda6d646759df9f6741e6ccc3a9a75c9645e3616d830bf2a13ba?context=explore)
	- [apptainer](https://apptainer.org/docs/user/latest/quick_start.html) (image definition file include in this repo)
	- [robostack](https://robostack.github.io/GettingStarted.html) (not recommended; difficult to track dependencies)
2. Install [ROS Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html) (skip if using docker/apptainer/robostack)
3. Install nav stack:
```
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
```
4. Download create3_examples:
```
git submodule init & git submodule update
```
5. Build the workspace:
```
colcon build
```

# Sanity Checks

Run the basic [ROS Nav2](https://navigation.ros.org/getting_started/index.html) demo:
```
# install turtlebot3 simulator
sudo apt install ros-humble-turtlebot3-gazebo

# set up environment variables
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models

# run demo
ros2 launch nav2_bringup tb3_simulation_launch.py headless:=False
```

Teleoperate the create3 robot base:
1. Connect to the robot via ethernet or its internal hotspot (hold down buttons \* and \*\* until lights flash blue)
2. Check the "topics" the robot exposes:
```
ros2 topic list
```
3. You should see a topic called `/cmd_vel`
4. Run the teleoperateion node:
```
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

# Run
#TODO
