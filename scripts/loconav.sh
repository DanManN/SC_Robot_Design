#!/usr/bin/env bash
terminator -e '~/interbotix_ws/flask_app1/trial_app1.sh'
terminator -e '~/interbotix_ws/speech_app/speech_app.sh || read'


# Launch the robot
source /opt/ros/humble/setup.bash
source /home/locobot/interbotix_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export INTERBOTIX_XSLOCOBOT_BASE_TYPE=create3
export INTERBOTIX_XSLOCOBOT_LIDAR_TYPE=rplidar_a2m8

echo "Launching application, please wait!"
ros2 launch interbotix_xslocobot_nav xslocobot_rtabmap.launch.py robot_model:=locobot_base use_lidar:=true slam_mode:=localization use_rviz:=true camera_tilt_angle:=-0.5