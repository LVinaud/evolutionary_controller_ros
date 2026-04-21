"""ROS2 node: neural network whose weights come from an evolutionary genome."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class NNController(Node):
    def __init__(self):
        super().__init__('nn_controller')
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
        # TODO: network forward pass with self.genome + features from (scan, odom, cam) -> Twist
        self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = NNController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
