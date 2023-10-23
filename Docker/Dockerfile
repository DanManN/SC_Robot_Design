# syntax=docker/dockerfile:1
FROM ros:humble

WORKDIR /home/locobot
RUN apt update && apt install -y \
	git \
	curl \
	ros-humble-desktop-full \
	ros-humble-navigation2 \
	ros-humble-nav2-bringup \
	ros-humble-turtlebot3-gazebo \
	ros-humble-rplidar-ros \
	ros-humble-irobot-create-description

# get interbotix install script
RUN curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh && chmod +x xsarm_amd64_install.sh

# run it to install interbotix workspace (~ 15 min)
RUN ./xsarm_amd64_install.sh -p /home/locobot/interbotix_ws -n

# user config
RUN useradd -m locobot -p locobot -s /usr/bin/bash
RUN printf '\n%s'\
'alias ll="ls -al"\n'\
'# Interbotix Configurations\n'\
'export RMW_IMPLEMENTATION=rmw_fastrtps_cpp\n'\
'export INTERBOTIX_XSLOCOBOT_BASE_TYPE=create3\n'\
'export INTERBOTIX_XSLOCOBOT_LIDAR_TYPE=rplidar_a2m8\n'\
'export INTERBOTIX_XSLOCOBOT_ROBOT_MODEL=locobot_wx250s\n'\
'source /opt/ros/humble/setup.bash\n'\
'source /usr/share/gazebo/setup.sh\n'\
'source /root/apriltag_ws/install/setup.bash\n'\
'source /home/locobot/interbotix_ws/install/setup.bash\n'\
'alias locoslam="ros2 launch interbotix_xslocobot_nav xslocobot_rtabmap.launch.py use_lidar:=true slam_mode:=mapping use_rviz:=true rtabmap_args:=-d "\n'\
'alias loconav="ros2 launch interbotix_xslocobot_nav xslocobot_rtabmap.launch.py use_lidar:=true slam_mode:=localization use_rviz:=true "\n'\
'alias locosim="ros2 launch interbotix_xslocobot_sim xslocobot_gz_classic.launch.py use_lidar:=true use_rviz:=false "\n'\
>> /home/locobot/.bashrc
RUN chown -R locobot:locobot /home/locobot/ /root
USER locobot
SHELL ["bash", "-c"]

# git packages
WORKDIR /home/locobot/interbotix_ws/src/
RUN git clone -b humble https://github.com/Interbotix/interbotix_ros_rovers.git
RUN sed -i 's#/mobile_base##' $(grep '/mobile_base' interbotix_ros_rovers -Rl)
WORKDIR /home/locobot/interbotix_ws/
RUN source ~/.bashrc && rm -rf build install log && colcon build --symlink-install

# runtime stuff
WORKDIR /home/locobot

ENTRYPOINT ["bash"]
