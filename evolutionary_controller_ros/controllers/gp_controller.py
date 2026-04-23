"""ROS2 node: GP tree controller (go-to-goal, with optional CTF state machine).

Loads a genome (JSON dict) and drives the robot toward a target pose
injected via ROS parameters (`target_x`, `target_y`). Each leaf of the
tree is `(action, duration_ms)`; the action is re-published at every
tick until it expires (keep-alive against diff_drive's
`cmd_vel_timeout=0.5s`), and the tree is re-evaluated only then.

Two operating modes (`mode` parameter):

    "train" (default) — target is whatever `target_x`/`target_y` say.
        `episode.py` sets them per scenario and the tree just navigates.
        No flag-grabbing / dropping happens here.

    "ctf" — the node runs a 3-phase mission state machine:
        phase 0: target = (enemy_base_x, enemy_base_y)
        phase 1: target = (enemy_flag_x, enemy_flag_y) after reaching
                 the enemy base (distance < phase_reach_radius_m).
        phase 2: PEGAR (hardcoded) when close to the flag; target
                 becomes (own_base_x, own_base_y). Flag is dragged with
                 the robot (handled externally — here we only flip
                 holding_flag).
        phase 3: SOLTAR (hardcoded) when close to own base; done.

Sensing contract
    `/scan`                            LaserScan
    `/diff_drive_base_controller/odom` Odometry (wheel odom; /odom relay
                                       is disabled in prm_2026)

Actuation
    `/cmd_vel`                         Twist
    `/gripper_controller/commands`     Float64MultiArray (3 joints)

Outbound observations for the trainer / external tools
    `/gp_controller/holding_flag`      Bool — latched via transient_local

Parameters (selected):
    genome_json, mode ("train" | "ctf"),
    target_x, target_y (train mode: episode sets these; ctf: SM overwrites),
    enemy_flag_x/y, enemy_base_x/y, own_base_x/y   (ctf only)
    v_frente, v_re, w_gira, grab_reach_m,
    phase_reach_radius_m, tick_hz,
    odom_topic, scan_topic.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import Empty

from ..evolution import genome as g
from ..utils import sensors as sens


# CTF phases (for mode="ctf")
_PHASE_GO_ENEMY_BASE = 0
_PHASE_GO_ENEMY_FLAG = 1
_PHASE_RETURN_HOME   = 2
_PHASE_DONE          = 3


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

        self.create_service(Empty, "/gp_controller/reset", self._on_reset)

        scan_topic = self.get_parameter("scan_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        self.get_logger().info(
            f"subs: scan={scan_topic} odom={odom_topic}")
        self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)

        self._scan = None
        self._odom = None
        self._genome = self._load_genome()

        self._current_action = None
        self._action_started_ns = 0
        self._action_duration_ns = 0
        self._last_twist = Twist()
        self._holding_flag = False
        self._ctf_phase = _PHASE_GO_ENEMY_BASE
        self._last_missing_log_ns = 0
        self._publish_holding(False)

        tick_hz = self.get_parameter("tick_hz").value
        self.create_timer(1.0 / float(tick_hz), self._tick)

        self.add_on_set_parameters_callback(self._on_params_changed)
        self.get_logger().info(
            f"gp_controller up (mode={self.get_parameter('mode').value})")

    # ----------------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------------

    def _declare_params(self):
        self.declare_parameter("genome_json", "")
        self.declare_parameter("mode", "train")  # "train" | "ctf"
        # Target injected externally (train mode) or overwritten by the
        # state machine (ctf mode).
        self.declare_parameter("target_x", 0.0)
        self.declare_parameter("target_y", 0.0)
        # Used only in ctf mode.
        self.declare_parameter("enemy_flag_x", 0.0)
        self.declare_parameter("enemy_flag_y", 0.0)
        self.declare_parameter("enemy_base_x", 0.0)
        self.declare_parameter("enemy_base_y", 0.0)
        self.declare_parameter("own_base_x",   0.0)
        self.declare_parameter("own_base_y",   0.0)
        self.declare_parameter("phase_reach_radius_m", 1.0)
        # Motion & grip.
        self.declare_parameter("v_frente",  0.4)
        self.declare_parameter("v_re",     -0.3)
        self.declare_parameter("w_gira",    0.8)
        self.declare_parameter("grab_reach_m", 0.3)
        # Joint order per prm_2026/config/controller_config.yaml:
        #   [gripper_extension, right_gripper_joint, left_gripper_joint].
        # (0.02, -0.02, 0.02) OPENS the claw (confirmed empirically).
        self.declare_parameter("gripper_open",  [0.02, -0.02, 0.02])
        self.declare_parameter("gripper_close", [0.0,  0.0,  0.0])
        self.declare_parameter("tick_hz", 10.0)
        # Wheel odometry. prm_2026 publishes under
        # /diff_drive_base_controller/odom (the /odom relay is commented
        # out in carrega_robo.launch.py). Overridable for other worlds.
        self.declare_parameter("odom_topic", "/diff_drive_base_controller/odom")
        self.declare_parameter("scan_topic", "/scan")

    def _on_params_changed(self, params):
        from rcl_interfaces.msg import SetParametersResult
        for p in params:
            if p.name == "genome_json" and p.value:
                try:
                    self._genome = g.from_json(p.value)
                    self.get_logger().info("genome reloaded from parameter")
                except ValueError as e:
                    return SetParametersResult(successful=False, reason=str(e))
            elif p.name == "mode":
                if p.value not in ("train", "ctf"):
                    return SetParametersResult(
                        successful=False,
                        reason=f"mode must be 'train' or 'ctf', got {p.value!r}")
                self._ctf_phase = _PHASE_GO_ENEMY_BASE
                self.get_logger().info(f"mode switched to {p.value!r}")
        return SetParametersResult(successful=True)

    def _load_genome(self):
        raw = self.get_parameter("genome_json").value
        if not raw:
            self.get_logger().warn("no genome_json parameter set; idling")
            return None
        return g.from_json(raw)

    # ----------------------------------------------------------------------
    # Reset service — called by episode.py between episodes
    # ----------------------------------------------------------------------

    def _on_reset(self, _request, response):
        self._current_action = None
        self._action_started_ns = 0
        self._action_duration_ns = 0
        self._last_twist = Twist()
        self.cmd_pub.publish(self._last_twist)
        self._set_holding(False)
        self._ctf_phase = _PHASE_GO_ENEMY_BASE
        self.get_logger().info("gp_controller reset")
        return response

    # ----------------------------------------------------------------------
    # Subscriptions
    # ----------------------------------------------------------------------

    def _on_scan(self, msg):
        self._scan = msg

    def _on_odom(self, msg):
        self._odom = msg

    # ----------------------------------------------------------------------
    # Tick — the main control loop
    # ----------------------------------------------------------------------

    def _tick(self):
        if self._genome is None or self._odom is None or self._scan is None:
            self._log_missing_inputs()
            return

        # CTF state machine: may update target + trigger PEGAR/SOLTAR.
        if self.get_parameter("mode").value == "ctf":
            self._step_ctf_state_machine()

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
        self._begin_action(action, dur_ms, now)

    def _log_missing_inputs(self):
        now = self.get_clock().now().nanoseconds
        if now - self._last_missing_log_ns < 2_000_000_000:
            return
        self._last_missing_log_ns = now
        missing = []
        if self._genome is None: missing.append("genome")
        if self._odom is None:   missing.append("odom")
        if self._scan is None:   missing.append("/scan")
        self.get_logger().warn(f"idle — waiting for: {', '.join(missing)}")

    def _begin_action(self, action, dur_ms, now_ns):
        twist = self._action_to_twist(action)
        self._last_twist = twist
        self._current_action = action
        self._action_started_ns = now_ns
        self._action_duration_ns = int(dur_ms * 1_000_000)
        self.cmd_pub.publish(twist)
        self.get_logger().info(
            f"action={action} dur={dur_ms}ms "
            f"v=({twist.linear.x:+.2f}, {twist.angular.z:+.2f})")

    def _action_to_twist(self, action):
        v_f = float(self.get_parameter("v_frente").value)
        v_r = float(self.get_parameter("v_re").value)
        w_g = float(self.get_parameter("w_gira").value)
        t = Twist()
        if action == "FRENTE":
            t.linear.x = v_f
        elif action == "RE":
            t.linear.x = v_r
        elif action == "GIRA_ESQ":
            t.angular.z = w_g
        elif action == "GIRA_DIR":
            t.angular.z = -w_g
        return t

    # ----------------------------------------------------------------------
    # CTF state machine (only active when mode == "ctf")
    # ----------------------------------------------------------------------

    def _step_ctf_state_machine(self):
        rx, ry, _ = self._robot_pose_xyy()
        reach = float(self.get_parameter("phase_reach_radius_m").value)
        grab = float(self.get_parameter("grab_reach_m").value)

        if self._ctf_phase == _PHASE_GO_ENEMY_BASE:
            self._set_target_from("enemy_base")
            if self._near(rx, ry, "enemy_base", reach):
                self._ctf_phase = _PHASE_GO_ENEMY_FLAG
                self.get_logger().info("CTF phase → GO_ENEMY_FLAG")

        elif self._ctf_phase == _PHASE_GO_ENEMY_FLAG:
            self._set_target_from("enemy_flag")
            if self._near(rx, ry, "enemy_flag", grab):
                self._do_grab()
                self._ctf_phase = _PHASE_RETURN_HOME
                self.get_logger().info("CTF phase → RETURN_HOME (grabbed)")

        elif self._ctf_phase == _PHASE_RETURN_HOME:
            self._set_target_from("own_base")
            if self._near(rx, ry, "own_base", reach):
                self._do_release()
                self._ctf_phase = _PHASE_DONE
                self.get_logger().info("CTF phase → DONE (delivered)")

        # _PHASE_DONE: target stays at own_base; robot coasts to stop.

    def _set_target_from(self, prefix: str):
        """Copy `{prefix}_x/y` into `target_x/y` if they differ."""
        x = float(self.get_parameter(f"{prefix}_x").value)
        y = float(self.get_parameter(f"{prefix}_y").value)
        tx = float(self.get_parameter("target_x").value)
        ty = float(self.get_parameter("target_y").value)
        if (x, y) != (tx, ty):
            self.set_parameters([
                rclpy.parameter.Parameter(
                    "target_x", rclpy.parameter.Parameter.Type.DOUBLE, x),
                rclpy.parameter.Parameter(
                    "target_y", rclpy.parameter.Parameter.Type.DOUBLE, y),
            ])

    def _near(self, rx, ry, prefix: str, radius: float) -> bool:
        x = float(self.get_parameter(f"{prefix}_x").value)
        y = float(self.get_parameter(f"{prefix}_y").value)
        return math.hypot(x - rx, y - ry) < radius

    def _do_grab(self):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in
                    self.get_parameter("gripper_close").value]
        self.gripper_pub.publish(msg)
        self._set_holding(True)

    def _do_release(self):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in
                    self.get_parameter("gripper_open").value]
        self.gripper_pub.publish(msg)
        self._set_holding(False)

    # ----------------------------------------------------------------------
    # Holding flag latched publisher (observable by external tools)
    # ----------------------------------------------------------------------

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
        rx, ry, ryaw = self._robot_pose_xyy()
        twist = self._odom.twist.twist
        return sens.compute_features(
            scan_ranges=list(self._scan.ranges),
            scan_angle_min=self._scan.angle_min,
            scan_angle_increment=self._scan.angle_increment,
            robot_pose=(rx, ry, ryaw),
            linear_vel=twist.linear.x,
            angular_vel=twist.angular.z,
            target_pose=(
                float(self.get_parameter("target_x").value),
                float(self.get_parameter("target_y").value),
            ),
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
