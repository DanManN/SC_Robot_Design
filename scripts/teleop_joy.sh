#!/usr/bin/env bash

# Launch the robot
source /opt/ros/humble/setup.bash 
source /home/locobot/interbotix_ws/install/setup.bash

echo "Launching application, please wait!"
ros2 launch teleop_twist_joy teleop-launch.py joy_config:='ps3' require_enable_button:=false joy_vel:='locobot/mobile_base/cmd_vel'
