"""Nó ROS2: rede neural cujos pesos vêm de um genoma evolutivo."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist


class ControladorNN(Node):
    def __init__(self):
        super().__init__('controlador_nn')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self._on_scan, 10)
        self.create_subscription(Odometry, '/odom_gt', self._on_odom, 10)
        self.create_subscription(Image, '/robot_cam/colored_map', self._on_cam, 10)
        self.create_timer(0.1, self._step)

        self.scan = None
        self.odom = None
        self.cam = None
        self.genoma = None

    def _on_scan(self, msg):
        self.scan = msg

    def _on_odom(self, msg):
        self.odom = msg

    def _on_cam(self, msg):
        self.cam = msg

    def _step(self):
        # TODO: forward da rede com self.genoma + features de (scan, odom, cam) -> Twist
        self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = ControladorNN()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
