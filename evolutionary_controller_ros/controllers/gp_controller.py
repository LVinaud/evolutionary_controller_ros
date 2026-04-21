"""ROS2 node: GP tree controller. Loads a genome (dict JSON) and drives the robot.

Evaluates the tree at each control tick; the selected leaf (action, duration_ms)
runs for that duration before the tree is re-evaluated.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class GPController(Node):
    def __init__(self):
        super().__init__('gp_controller')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 10)
        self.create_subscription(Odometry, '/odom_gt', self._on_odom, 10)
        self.create_subscription(Image, '/robot_cam/colored_map', self._on_cam, 10)
        self.create_timer(0.1, self._step)

        self.scan = None
        self.odom = None
        self.cam = None
        self.genome = None

    def _on_scan(self, msg):
        self.scan = msg

    def _on_odom(self, msg):
        self.odom = msg

    def _on_cam(self, msg):
        self.cam = msg

    def _step(self):
        # TODO: build sensor dict, evaluate tree, publish Twist for current action
        self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = GPController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
