# Launch the robot
source /opt/ros/humble/setup.bash
source /home/locobot/interbotix_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export INTERBOTIX_XSLOCOBOT_BASE_TYPE=create3
export INTERBOTIX_XSLOCOBOT_LIDAR_TYPE=rplidar_a2m8

python ~/interbotix_ws/speech_app/restructured_dialougue.py
