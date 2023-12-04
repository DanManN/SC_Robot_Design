import subprocess
from flask import Flask, render_template, request, redirect
import time
app = Flask(__name__, static_url_path='/static')

location_dict = {
    'home': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: 'now', frame_id: 'map'}, pose: {position: {x: 3.3491172790527344, y: -0.004497826099395752, z: 0.002140045166015625}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room1': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: 'now', frame_id: 'map'}, pose: {position: {x: 6.916250228881836, y: -0.15002679824829102, z: 0.0015544891357421875}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room2': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: 'now', frame_id: 'map'}, pose: {position: {x: 9.780485153198242, y: -4.3226823806762695, z: 0.0046672821044921875}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room3': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: 'now', frame_id: 'map'}, pose: {position: {x: 3.4306554794311523, y: -7.267694473266602, z: 0.0068721771240234375}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room4': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: 'now', frame_id: 'map'}, pose: {position: {x: -1.6239104270935059, y: 0.7523196935653687, z: 0.00223541259765625}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
    'room5': "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: 'now', frame_id: 'map'}, pose: {position: {x: 1.6131210327148438, y: 2.954784393310547, z: 0.00484466552734375}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'",
}

@app.route('/')
def index():
    return render_template('index-2.html')

@app.route('/send_location', methods=['POST'])
def send_location():
    location = request.form['location']
    count = 1# int(request.form['count'])

    if location in location_dict:
        value_for_location = location_dict[location]
    else:
        value_for_location = "ros2 topic pub -1 /goal_pose geometry_msgs/msg/PoseStamped '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: 'map'}, pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, w: 1.0}}}'"

    # for indd in range(count):
        # print(indd,": this is indd" )
    full_message = f"Location: {location}, Command: {value_for_location}"
    try:
        talker_command = f"ros2 topic pub -1 /chatter std_msgs/msg/String '{{data: \"{full_message}\"}}'"
        #subprocess.check_output(talker_command, shell=True, stderr=subprocess.STDOUT)
        subprocess.check_output(value_for_location, shell=True, stderr=subprocess.STDOUT)
        time.sleep(1)
        return redirect("/")
        # break
    except subprocess.CalledProcessError as e:
        return f"Error: {e.output.decode('utf-8')}"
        # break
    

    return f"ROS 2 command for location '{location}' sent {count} times."

if __name__ == '__main__':
    app.run(debug=True)

