#!/usr/bin/env bash

# Launch the robot
source /opt/ros/humble/setup.bash 
source /home/locobot/interbotix_ws/install/setup.bash

echo "Launching application, please wait!"
ros2 run joy joy_node
