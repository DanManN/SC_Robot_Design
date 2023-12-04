import sys
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from people_msgs.msg import People, Person


class PeopleVelocityPublisher(Node):

    def __init__(self):
        super().__init__('people_velocity_publisher')
        self.publisher_ = self.create_publisher(People, 'people', 10)
        self.dt = 0.01  # seconds
        self.t = 0  # seconds
        self.dxdt = float(sys.argv[1])  # meters per second
        self.start = np.array([
            3.9955430030822754,
            -7.303525447845459,
            0.001216888427734375,
        ])
        self.goal = np.array([
            -0.09661579132080078,
            -0.06329691410064697,
            0.0034303665161132812,
        ])
        direction = (self.goal -
                     self.start) / np.linalg.norm(self.goal - self.start)
        self.velocity = self.dxdt * direction
        self.timer = self.create_timer(self.dt, self.timer_callback)

    def timer_callback(self):
        self.t += self.dt
        self.get_logger().info(f'Time: "{self.t}"')

        person_msg = Person()
        person_msg.name = 'person'
        person_msg.position = Point()
        position = self.start + (self.t-5) * self.velocity
        person_msg.position.x = position[0]
        person_msg.position.y = position[1]
        person_msg.position.z = position[2]
        person_msg.velocity = Point()
        person_msg.velocity.x = self.velocity[0]
        person_msg.velocity.y = self.velocity[1]
        person_msg.velocity.z = self.velocity[2]
        people_msg = People()
        people_msg.header.frame_id = 'map'
        people_msg.people = [person_msg]
        self.get_logger().info(f'People: "{people_msg}"')
        self.publisher_.publish(people_msg)


def main(args=None):
    rclpy.init(args=args)

    publisher = PeopleVelocityPublisher()

    rclpy.spin(publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
