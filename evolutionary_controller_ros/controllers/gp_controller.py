"""ROS2 node: GP tree controller.

Loads a genome (JSON dict) and drives the robot by executing one leaf
`(action, duration_ms)` at a time. The tree is re-evaluated only when
the current action expires; during its duration the same `cmd_vel` is
re-published every tick (keep-alive against the diff_drive controller's
`cmd_vel_timeout=0.5s`).

Single node for both training and demo — `episode.py` only observes
(via `/gp_controller/holding_flag` and the usual odom/labels topics)
and updates ROS parameters between scenarios (genome, target poses,
our_team). It never injects control state into this node.

Sensing contract
    `/scan`                  LaserScan
    `/odom`                  Odometry (wheel odometry with drift — NOT gt)
    `/robot_cam/labels_map`  Image mono8 (pixel = semantic label id)

Actuation
    `/cmd_vel`                      Twist
    `/gripper_controller/commands`  Float64MultiArray (3 joints)

Outbound observations for the trainer
    `/gp_controller/holding_flag`   Bool — latched via transient_local

Parameters (ros2 run --ros-args -p ...):
    genome_json        : str  — the genome as a JSON string
    our_team           : str  — "blue" (default) or "red"
    enemy_flag_x/y     : float
    own_base_x/y       : float
    enemy_base_x/y     : float
    deploy_zone_x/y    : float
    v_frente / v_re                     (default  0.4 / -0.3)
    w_gira                              (default  0.8)
    grab_reach_m                        (default  0.3)
    gripper_open / gripper_close        (default [0,0,0] / [0.02, -0.02, 0.02])
    tick_hz                             (default 10.0)
"""
import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64MultiArray

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

import numpy as np

from ..evolution import genome as g
from ..utils import sensors as sens


_TEAM_TO_LABELS = {
    # (enemy_flag_label, enemy_base_label, own_base_label)
    "blue": (20, 10, 15),
    "red":  (25, 15, 10),
}


class GPController(Node):
    def __init__(self):
        super().__init__("gp_controller")
        self._declare_params()

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.gripper_pub = self.create_publisher(
            Float64MultiArray, "/gripper_controller/commands", 10)

        latched = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.holding_pub = self.create_publisher(
            Bool, "/gp_controller/holding_flag", latched)

        self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(Image, "/robot_cam/labels_map",
                                 self._on_labels, 10)

        self._bridge = CvBridge() if CvBridge is not None else None
        self._scan = None
        self._odom = None
        self._labels = None

        self._genome = self._load_genome()

        self._current_action = None
        self._action_started_ns = 0
        self._action_duration_ns = 0
        self._last_twist = Twist()
        self._holding_flag = False
        self._publish_holding(False)

        tick_hz = self.get_parameter("tick_hz").value
        self.create_timer(1.0 / float(tick_hz), self._tick)

        self.add_on_set_parameters_callback(self._on_params_changed)
        self.get_logger().info("gp_controller up")

    # ----------------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------------

    def _declare_params(self):
        self.declare_parameter("genome_json", "")
        self.declare_parameter("our_team", "blue")
        self.declare_parameter("enemy_flag_x", 0.0)
        self.declare_parameter("enemy_flag_y", 0.0)
        self.declare_parameter("own_base_x",   0.0)
        self.declare_parameter("own_base_y",   0.0)
        self.declare_parameter("enemy_base_x", 0.0)
        self.declare_parameter("enemy_base_y", 0.0)
        self.declare_parameter("deploy_zone_x", 0.0)
        self.declare_parameter("deploy_zone_y", 0.0)
        self.declare_parameter("v_frente",  0.4)
        self.declare_parameter("v_re",     -0.3)
        self.declare_parameter("w_gira",    0.8)
        self.declare_parameter("grab_reach_m", 0.3)
        self.declare_parameter("gripper_open",  [0.0, 0.0, 0.0])
        self.declare_parameter("gripper_close", [0.02, -0.02, 0.02])
        self.declare_parameter("tick_hz", 10.0)

    def _on_params_changed(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "genome_json" and p.value:
                try:
                    self._genome = g.from_json(p.value)
                    self.get_logger().info("genome reloaded from parameter")
                except ValueError as e:
                    return SetParametersResult(successful=False, reason=str(e))
        return SetParametersResult(successful=True)

    def _load_genome(self):
        raw = self.get_parameter("genome_json").value
        if not raw:
            self.get_logger().warn("no genome_json parameter set; idling")
            return None
        return g.from_json(raw)

    # ----------------------------------------------------------------------
    # Subscriptions
    # ----------------------------------------------------------------------

    def _on_scan(self, msg):
        self._scan = msg

    def _on_odom(self, msg):
        self._odom = msg

    def _on_labels(self, msg):
        if self._bridge is None:
            # Fallback: raw mono8 unpack (sensor_msgs/Image → uint8 array).
            if msg.encoding not in ("mono8", "8UC1"):
                return
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            self._labels = buf.reshape(msg.height, msg.width)
        else:
            try:
                self._labels = self._bridge.imgmsg_to_cv2(
                    msg, desired_encoding="mono8")
            except Exception as e:
                self.get_logger().warn(f"labels_map conversion failed: {e}")

    # ----------------------------------------------------------------------
    # Tick — the main control loop
    # ----------------------------------------------------------------------

    def _tick(self):
        if self._genome is None or self._odom is None or self._scan is None:
            return

        now = self.get_clock().now().nanoseconds
        running = (self._current_action is not None and
                   (now - self._action_started_ns) < self._action_duration_ns)

        if running:
            self.cmd_pub.publish(self._last_twist)
            return

        feats = self._build_features()
        if feats is None:
            return
        action, dur_ms = g.evaluate(self._genome, feats)
        self._begin_action(action, dur_ms, feats, now)

    def _begin_action(self, action, dur_ms, feats, now_ns):
        twist, gripper_cmd = self._action_to_commands(action, feats)
        self._last_twist = twist
        self._current_action = action
        self._action_started_ns = now_ns
        self._action_duration_ns = int(dur_ms * 1_000_000)
        self.cmd_pub.publish(twist)
        if gripper_cmd is not None:
            self.gripper_pub.publish(gripper_cmd)

    def _action_to_commands(self, action, feats):
        v_f = float(self.get_parameter("v_frente").value)
        v_r = float(self.get_parameter("v_re").value)
        w_g = float(self.get_parameter("w_gira").value)
        twist = Twist()
        gripper = None

        if action == "FRENTE":
            twist.linear.x = v_f
        elif action == "RE":
            twist.linear.x = v_r
        elif action == "GIRA_ESQ":
            twist.angular.z = w_g
        elif action == "GIRA_DIR":
            twist.angular.z = -w_g
        elif action == "PARAR":
            pass
        elif action == "PEGAR":
            gripper = self._gripper_msg("gripper_close")
            self._try_grab()
        elif action == "SOLTAR":
            gripper = self._gripper_msg("gripper_open")
            self._set_holding(False)
        return twist, gripper

    def _gripper_msg(self, param_name):
        arr = list(self.get_parameter(param_name).value)
        msg = Float64MultiArray()
        msg.data = [float(x) for x in arr]
        return msg

    # ----------------------------------------------------------------------
    # Flag-grabbing state
    # ----------------------------------------------------------------------

    def _try_grab(self):
        if self._holding_flag:
            return
        rx, ry, _ = self._robot_pose_xyy()
        fx = float(self.get_parameter("enemy_flag_x").value)
        fy = float(self.get_parameter("enemy_flag_y").value)
        reach = float(self.get_parameter("grab_reach_m").value)
        if math.hypot(fx - rx, fy - ry) < reach:
            self._set_holding(True)

    def _set_holding(self, val: bool):
        if self._holding_flag == val:
            return
        self._holding_flag = val
        self._publish_holding(val)

    def _publish_holding(self, val: bool):
        msg = Bool()
        msg.data = val
        self.holding_pub.publish(msg)

    # ----------------------------------------------------------------------
    # Feature extraction bridge
    # ----------------------------------------------------------------------

    def _robot_pose_xyy(self):
        pose = self._odom.pose.pose
        q = pose.orientation
        # yaw = atan2(2(wz + xy), 1 − 2(y² + z²))
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return pose.position.x, pose.position.y, yaw

    def _build_features(self):
        team = self.get_parameter("our_team").value
        if team not in _TEAM_TO_LABELS:
            self.get_logger().error(f"invalid our_team: {team!r}")
            return None
        enemy_flag_lbl, enemy_base_lbl, _own_base_lbl = _TEAM_TO_LABELS[team]

        rx, ry, ryaw = self._robot_pose_xyy()
        twist = self._odom.twist.twist

        return sens.compute_features(
            scan_ranges=list(self._scan.ranges),
            scan_angle_min=self._scan.angle_min,
            scan_angle_increment=self._scan.angle_increment,
            robot_pose=(rx, ry, ryaw),
            linear_vel=twist.linear.x,
            angular_vel=twist.angular.z,
            labels_map=self._labels,
            enemy_flag_pose=(
                float(self.get_parameter("enemy_flag_x").value),
                float(self.get_parameter("enemy_flag_y").value),
            ),
            own_base_pose=(
                float(self.get_parameter("own_base_x").value),
                float(self.get_parameter("own_base_y").value),
            ),
            enemy_base_pose=(
                float(self.get_parameter("enemy_base_x").value),
                float(self.get_parameter("enemy_base_y").value),
            ),
            deploy_zone_pose=(
                float(self.get_parameter("deploy_zone_x").value),
                float(self.get_parameter("deploy_zone_y").value),
            ),
            holding_flag=self._holding_flag,
            enemy_flag_label=enemy_flag_lbl,
            enemy_base_label=enemy_base_lbl,
        )


def main():
    rclpy.init()
    node = GPController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
