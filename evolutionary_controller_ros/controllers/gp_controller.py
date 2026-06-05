"""ROS2 node: GP tree controller + mission state machine.

Loads a genome (JSON dict) and drives the robot toward a target pose
injected via ROS parameters (`target_x`, `target_y`). Each leaf of the
tree is `(action, duration_ms)`; the action is re-published at every
tick until it expires (keep-alive against diff_drive's
`cmd_vel_timeout=0.5s`), and the tree is re-evaluated only then.

Three operating modes (`mode` parameter):

    "train" (default) — target is whatever `target_x`/`target_y` say.
        `episode.py` sets them per scenario and the tree just navigates.
        No flag-grabbing / dropping happens here. This is the mode
        used during the evolutionary run.

    "mission" — Trabalho 1 deliverable. A 4-state machine drives the
        robot through:
            EXPLORANDO           — no flag in sight; reactive forward +
                                   obstacle-avoid bypasses the GP tree
                                   so the tree's narrow training is not
                                   misused here.
            BANDEIRA_DETECTADA   — transient: logs the detection and
                                   immediately advances.
            NAVEGANDO_PARA_BANDEIRA — flag visible; bearing+distance
                                   from the camera/LIDAR fusion is
                                   converted into a world-frame target,
                                   then the GP tree handles the
                                   navigation it was evolved for.
            POSICIONANDO_PARA_COLETA — close to the flag; bypass the
                                   tree, centre the camera bearing and
                                   stop at a known distance.

    "ctf" — Trabalho 2 placeholder: the older 3-phase CTF logic
        (GO_ENEMY_BASE → GO_ENEMY_FLAG → RETURN_HOME) using hardcoded
        coordinates and the gripper. Kept around as the scaffold for
        the next deliverable; not driven by the visual detector.

Sensing contract
    `/scan`                            LaserScan
    `/diff_drive_base_controller/odom` Odometry (wheel odom; /odom relay
                                       is disabled in prm_2026)
    `/robot_cam/labels_map`            Image (Gazebo segmentation labels;
                                       only used in mission mode)

Actuation
    `/cmd_vel`                         Twist
    `/gripper_controller/commands`     Float64MultiArray (3 joints)

Outbound observations
    `/gp_controller/holding_flag`      Bool — latched via transient_local
    `/flag/detected`                   Bool — latched; visual detection state
    `/flag/bearing_rad`                Float32 — robot-frame angle to flag
                                       (0 ahead, + left, − right)
    `/flag/area_px`                    Int32 — number of matching pixels
                                       (proxy for proximity)
    `/flag/distance_m`                 Float32 — LIDAR-fused distance,
                                       NaN if unavailable

Parameters (selected):
    genome_json, mode ("train" | "mission" | "ctf"),
    target_x, target_y (train mode: episode sets these; mission/ctf SM overwrites),
    enemy_flag_label, camera_horizontal_fov_rad, labels_topic,
    flag_min_area_px, flag_hysteresis_on/off, flag_max_distance_m,
    posicionamento_radius_m, posicionamento_target_distance_m,
    posicionamento_bearing_tolerance_rad,
    enemy_flag_x/y, enemy_base_x/y, own_base_x/y   (ctf only)
    v_frente, v_re, w_gira, grab_reach_m,
    phase_reach_radius_m, tick_hz,
    odom_topic, scan_topic.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Int32, Float64MultiArray
from std_srvs.srv import Empty

from ..evolution import genome as g
from ..utils import sensors as sens
from ..utils import flag_detection as fd
from ..evaluation import world_reset as wr


# CTF phases (for mode="ctf")
_PHASE_GO_ENEMY_BASE = 0
_PHASE_GO_ENEMY_FLAG = 1
_PHASE_RETURN_HOME   = 2
_PHASE_DONE          = 3

# Mission states (for mode="mission")
_MISSION_EXPLORANDO              = "EXPLORANDO"
_MISSION_BANDEIRA_DETECTADA      = "BANDEIRA_DETECTADA"
_MISSION_NAVEGANDO_PARA_BANDEIRA = "NAVEGANDO_PARA_BANDEIRA"
_MISSION_POSICIONANDO_PARA_COLETA = "POSICIONANDO_PARA_COLETA"

_MISSION_STATES = (
    _MISSION_EXPLORANDO,
    _MISSION_BANDEIRA_DETECTADA,
    _MISSION_NAVEGANDO_PARA_BANDEIRA,
    _MISSION_POSICIONANDO_PARA_COLETA,
)

_VALID_MODES = ("train", "mission", "ctf")


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

        # Detection topics — /flag/detected is latched so subscribers
        # that connect mid-run see the current state. The numeric
        # streams are unlatched (they only matter while subscribing).
        self.flag_detected_pub = self.create_publisher(
            Bool, "/flag/detected", latched)
        self.flag_bearing_pub  = self.create_publisher(
            Float32, "/flag/bearing_rad", 10)
        self.flag_area_pub     = self.create_publisher(
            Int32, "/flag/area_px", 10)
        self.flag_distance_pub = self.create_publisher(
            Float32, "/flag/distance_m", 10)

        self.create_service(Empty, "/gp_controller/reset", self._on_reset)

        scan_topic = self.get_parameter("scan_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        labels_topic = self.get_parameter("labels_topic").value
        self.get_logger().info(
            f"subs: scan={scan_topic} odom={odom_topic} labels={labels_topic}")
        self.create_subscription(LaserScan, scan_topic, self._on_scan, 10)
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(Image, labels_topic, self._on_labels, 10)

        # Sensor inputs.
        self._scan = None
        self._odom = None
        self._labels = None   # latest decoded numpy 2-D array of labels

        self._genome = self._load_genome()

        # Tree-driven cmd_vel timing state.
        self._current_action = None
        self._action_started_ns = 0
        self._action_duration_ns = 0
        self._last_twist = Twist()

        # CTF state machine.
        self._holding_flag = False
        self._ctf_phase = _PHASE_GO_ENEMY_BASE
        self._publish_holding(False)

        # Mission state machine.
        self._mission_state = _MISSION_EXPLORANDO
        # Used to detect transitions so the LED colour only updates once
        # per change instead of every tick (each update is an ign service
        # subprocess call ≈ 100-300 ms).
        self._prev_mission_state = None
        # Latest raw detection (after image processing, before hysteresis).
        self._raw_detected = False
        self._raw_bearing = 0.0
        self._raw_area = 0
        self._raw_distance = float("nan")
        # Hysteresis-filtered detection (what the SM consumes).
        self._flag_detected = False
        self._consec_seen = 0
        self._consec_lost = 0
        self._publish_flag_detected(False)

        self._last_missing_log_ns = 0

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
        self.declare_parameter("mode", "train")  # "train" | "mission" | "ctf"
        # Target injected externally (train mode) or overwritten by the
        # state machine (mission/ctf modes).
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
        # ---- Visual detection (mission mode) -----------------------
        # Gazebo semantic-segmentation image. Each pixel is the label
        # ID of the model it belongs to. From arena_cilindros.sdf:
        #     red_flag = 20, blue_flag = 25
        self.declare_parameter("labels_topic", "/robot_cam/labels_map")
        # Default 20 = chase red flag (we are blue). Set to 25 to play
        # red.
        self.declare_parameter("enemy_flag_label", 20)
        # From prm_2026/description/robot.urdf.xacro: <horizontal_fov>1.57</...>
        self.declare_parameter("camera_horizontal_fov_rad", 1.57)
        # Detection robustness.
        self.declare_parameter("flag_min_area_px", 30)
        self.declare_parameter("flag_hysteresis_on", 3)
        self.declare_parameter("flag_hysteresis_off", 5)
        self.declare_parameter("flag_max_distance_m", 10.0)
        # ---- Mission state-machine geometry ------------------------
        # Distance below which NAVEGANDO transitions to POSICIONANDO.
        self.declare_parameter("posicionamento_radius_m", 1.0)
        # Where POSICIONANDO tries to stop relative to the flag.
        self.declare_parameter("posicionamento_target_distance_m", 0.5)
        # How tightly POSICIONANDO must centre the bearing before
        # declaring "positioned" and stopping.
        self.declare_parameter("posicionamento_bearing_tolerance_rad", 0.1)
        # Exploration policy speeds (used while EXPLORANDO).
        self.declare_parameter("explore_obstacle_radius_m", 0.6)
        self.declare_parameter("explore_v_frente", 0.25)
        self.declare_parameter("explore_w_gira", 0.6)
        # ---- Cartola LED (visual feedback in Gazebo) --------------
        # The robot wears a top hat with a cylindrical "LED" on top
        # (see prm_2026/description/robot.urdf.xacro additions). When
        # mode=="mission" the LED's diffuse colour is updated on every
        # state transition via /world/<name>/visual_config so the
        # operator can read the SM state from the Gazebo viewport. The
        # ign service call runs in a background thread so it cannot
        # block the 10 Hz tick.
        self.declare_parameter("world_name", "capture_the_flag_world")
        self.declare_parameter("robot_model_name", "prm_robot")
        self.declare_parameter("led_link_name", "cartola_led_link")
        self.declare_parameter("led_visual_name", "led_visual")
        self.declare_parameter("led_update_enabled", True)
        # Per-state RGB triples in [0, 1]. Defaults map state semantics
        # onto natural traffic-light colours: yellow=searching, cyan=spotted,
        # green=going, red=arrived.
        self.declare_parameter("led_explorando_rgb",
                               [1.0, 1.0, 0.0])
        self.declare_parameter("led_bandeira_detectada_rgb",
                               [0.0, 1.0, 1.0])
        self.declare_parameter("led_navegando_rgb",
                               [0.0, 1.0, 0.0])
        self.declare_parameter("led_posicionando_rgb",
                               [1.0, 0.0, 0.0])

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
                if p.value not in _VALID_MODES:
                    return SetParametersResult(
                        successful=False,
                        reason=(f"mode must be one of {_VALID_MODES}, "
                                f"got {p.value!r}"))
                # Reset both state machines on mode change so we never
                # carry stale phase state into a fresh mode.
                self._ctf_phase = _PHASE_GO_ENEMY_BASE
                self._mission_state = _MISSION_EXPLORANDO
                self._consec_seen = 0
                self._consec_lost = 0
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
        # Mission state machine is also reset — the next episode starts
        # exploring from scratch with no carried-over detection memory.
        self._mission_state = _MISSION_EXPLORANDO
        self._flag_detected = False
        self._consec_seen = 0
        self._consec_lost = 0
        self._publish_flag_detected(False)
        self.get_logger().info("gp_controller reset")
        return response

    # ----------------------------------------------------------------------
    # Subscriptions
    # ----------------------------------------------------------------------

    def _on_scan(self, msg):
        self._scan = msg

    def _on_odom(self, msg):
        self._odom = msg

    def _on_labels(self, msg):
        """Decode Gazebo semantic-segmentation Image to a numpy 2-D array.

        The plugin can emit the labels in a few different encodings
        depending on Gazebo version and message-bridge settings:

            * `8UC1` / `mono8`  — single-channel grayscale, pixel value
                                  = label id. Easiest case.
            * `16UC1` / `mono16` — same idea, 16-bit. For worlds with
                                  more than 255 classes.
            * `rgb8` / `bgr8`   — RGB image where the *first* channel
                                  carries the label id (G and B carry
                                  instance/metadata that we don't use).
                                  This is what Gazebo Fortress publishes
                                  on `/robot_cam/labels_map` after the
                                  ros_gz_bridge mapping. We extract
                                  channel 0 (R for rgb8, B for bgr8 —
                                  but in practice the label is in the
                                  first colour byte either way).

        Decoding happens here so the per-tick `_update_detection` call
        stays cheap.
        """
        import numpy as np
        enc = msg.encoding
        try:
            if enc in ("8UC1", "mono8"):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width)
            elif enc in ("16UC1", "mono16"):
                arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    msg.height, msg.width)
            elif enc in ("rgb8", "bgr8"):
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)
                # Label is in the first byte of each pixel. For rgb8
                # that's R; for bgr8 it's B — same physical byte, just
                # different colour name. Taking channel 0 covers both.
                arr = img[:, :, 0]
            else:
                now = self.get_clock().now().nanoseconds
                if now - self._last_missing_log_ns >= 2_000_000_000:
                    self.get_logger().warn(
                        f"unknown labels_map encoding {enc!r}; ignoring frame")
                    self._last_missing_log_ns = now
                return
        except ValueError:
            return  # mismatched buffer size; drop frame quietly
        self._labels = arr

    # ----------------------------------------------------------------------
    # Tick — the main control loop
    # ----------------------------------------------------------------------

    def _tick(self):
        if self._genome is None or self._odom is None or self._scan is None:
            self._log_missing_inputs()
            return

        mode = self.get_parameter("mode").value

        # Visual detection runs in any mode that has labels data,
        # because the /flag/* topics are useful for debugging/teaching
        # even when the SM isn't consuming them. The mission SM is the
        # only place where the result drives behaviour.
        self._update_detection()

        # Mode-specific pre-processing. The mission SM may take direct
        # control of /cmd_vel for the EXPLORANDO / POSICIONANDO states;
        # the CTF SM only updates target params and never short-circuits.
        if mode == "ctf":
            self._step_ctf_state_machine()
        elif mode == "mission":
            override = self._step_mission_state_machine()
            # Detect mission state transitions and update the LED.
            # Doing it here (after the SM step) means we fire exactly
            # one update per change, regardless of how the SM moved.
            if self._mission_state != self._prev_mission_state:
                self._on_mission_state_changed(self._prev_mission_state,
                                               self._mission_state)
                self._prev_mission_state = self._mission_state
            if override is not None:
                # State machine took direct control; bypass the GP tree.
                self._last_twist = override
                self._current_action = None
                self.cmd_pub.publish(override)
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
    # Visual detection: process latest labels_map + LIDAR fusion + publish
    # ----------------------------------------------------------------------

    def _update_detection(self):
        """Process the latest segmentation frame and update detection state.

        Writes:
            self._raw_*           — single-frame detection (no hysteresis)
            self._flag_detected   — filtered (after on/off hysteresis)
            self._consec_seen / _consec_lost — hysteresis counters
        And publishes the four /flag/* topics every tick where we have
        a labels frame.
        """
        if self._labels is None:
            return

        label = int(self.get_parameter("enemy_flag_label").value)
        min_area = int(self.get_parameter("flag_min_area_px").value)

        detected, centroid_x, area = fd.detect_flag(
            self._labels, target_label=label, min_area_px=min_area)

        self._raw_area = area
        if detected:
            W = self._labels.shape[1]
            fov = float(self.get_parameter("camera_horizontal_fov_rad").value)
            bearing = fd.bearing_from_centroid(centroid_x, W, fov)
            self._raw_bearing = bearing
            self._raw_detected = True
            # LIDAR fusion: range at the bearing → distance.
            if self._scan is not None:
                max_d = float(self.get_parameter("flag_max_distance_m").value)
                d = fd.fuse_distance_from_scan(
                    bearing_rad=bearing,
                    scan_ranges=list(self._scan.ranges),
                    scan_angle_min=self._scan.angle_min,
                    scan_angle_increment=self._scan.angle_increment,
                    max_distance_m=max_d,
                )
                self._raw_distance = d if d is not None else float("nan")
            else:
                self._raw_distance = float("nan")
        else:
            self._raw_detected = False
            self._raw_bearing = 0.0
            self._raw_distance = float("nan")

        # Hysteresis: on requires N consecutive sees, off requires M
        # consecutive misses. Without this, a single dropped frame near
        # the flag's edge would oscillate the state machine.
        on_n  = int(self.get_parameter("flag_hysteresis_on").value)
        off_n = int(self.get_parameter("flag_hysteresis_off").value)
        if self._raw_detected:
            self._consec_seen += 1
            self._consec_lost = 0
            if not self._flag_detected and self._consec_seen >= on_n:
                self._flag_detected = True
                self._publish_flag_detected(True)
        else:
            self._consec_lost += 1
            self._consec_seen = 0
            if self._flag_detected and self._consec_lost >= off_n:
                self._flag_detected = False
                self._publish_flag_detected(False)

        # Publish numeric streams every tick (consumers can filter on
        # /flag/detected if they only want valid samples).
        self.flag_bearing_pub.publish(Float32(data=float(self._raw_bearing)))
        self.flag_area_pub.publish(Int32(data=int(self._raw_area)))
        self.flag_distance_pub.publish(Float32(data=float(self._raw_distance)))

    def _publish_flag_detected(self, val: bool):
        m = Bool()
        m.data = bool(val)
        self.flag_detected_pub.publish(m)

    # ----------------------------------------------------------------------
    # Mission state machine (only active when mode == "mission")
    # ----------------------------------------------------------------------

    def _step_mission_state_machine(self):
        """Step the mission SM and optionally return a Twist override.

        Return value contract:
            * `Twist`  — the SM is taking direct control of /cmd_vel;
                         the caller publishes this and skips the GP tree.
            * `None`   — the SM has updated target_x/y (or chose to do
                         nothing); the caller falls through to GP tree
                         evaluation.
        """
        s = self._mission_state

        if s == _MISSION_EXPLORANDO:
            if self._flag_detected:
                self._mission_state = _MISSION_BANDEIRA_DETECTADA
                # Defer the transition's behaviour to next tick so we
                # always log from a single place (BANDEIRA_DETECTADA).
                return self._twist_stop()
            return self._twist_explore()

        if s == _MISSION_BANDEIRA_DETECTADA:
            # Transient state: log + advance. Stopping for one tick
            # gives a clean visual cue that "we noticed the flag".
            self.get_logger().info(
                f"[mission] BANDEIRA_DETECTADA: "
                f"bearing={math.degrees(self._raw_bearing):+.1f}°, "
                f"distance={self._raw_distance:.2f}m, "
                f"area={self._raw_area}px"
            )
            self._mission_state = _MISSION_NAVEGANDO_PARA_BANDEIRA
            return self._twist_stop()

        if s == _MISSION_NAVEGANDO_PARA_BANDEIRA:
            if not self._flag_detected:
                self._mission_state = _MISSION_EXPLORANDO
                self.get_logger().info(
                    "[mission] flag lost → EXPLORANDO")
                return None
            # Convert bearing + distance into a world-frame target so
            # the GP tree's "go-to-goal" features remain meaningful.
            tx, ty = self._target_from_detection()
            if tx is not None:
                self._set_target_xy(tx, ty)
            # Close enough? Hand off to positioning.
            radius = float(
                self.get_parameter("posicionamento_radius_m").value)
            if (not math.isnan(self._raw_distance)
                    and self._raw_distance < radius):
                self._mission_state = _MISSION_POSICIONANDO_PARA_COLETA
                self.get_logger().info(
                    f"[mission] close to flag ({self._raw_distance:.2f}m) "
                    "→ POSICIONANDO_PARA_COLETA")
            return None    # GP tree handles motion

        if s == _MISSION_POSICIONANDO_PARA_COLETA:
            return self._twist_positioning()

        # Unknown state — defensive default.
        self._mission_state = _MISSION_EXPLORANDO
        return None

    # ---- Helper twists for the SM-controlled states ----------------------

    def _twist_stop(self):
        t = Twist()
        return t   # zeros

    def _twist_explore(self):
        """Reactive forward + obstacle-avoid, used while EXPLORANDO.

        We deliberately bypass the GP tree here because the tree was
        evolved to chase a known target; without one it would drive
        toward (0,0) by default, which is not exploration. A simple
        hand-coded behaviour is more honest about the state we are in
        and is what the assignment's `EXPLORANDO` description asks for
        ("aleatório ou sistemático").
        """
        obs_radius = float(
            self.get_parameter("explore_obstacle_radius_m").value)
        v = float(self.get_parameter("explore_v_frente").value)
        w = float(self.get_parameter("explore_w_gira").value)

        # Check front cone of the LIDAR. We reuse the same convention
        # as sensors.compute_features (60° centred cone).
        front_clear = self._front_is_clear(obs_radius)
        t = Twist()
        if front_clear:
            t.linear.x = v
        else:
            # Turn left by default; deterministic is fine for a small
            # arena and easier to reason about than randomized.
            t.angular.z = w
        return t

    def _front_is_clear(self, radius: float) -> bool:
        """True iff no LIDAR return inside `radius` in the front cone."""
        if self._scan is None:
            return False
        cone_half = math.radians(30)   # ±30° front cone
        amin = self._scan.angle_min
        inc = self._scan.angle_increment
        n = len(self._scan.ranges)
        if inc <= 0 or n == 0:
            return False
        lo = max(0, int((-cone_half - amin) / inc))
        hi = min(n, int((cone_half - amin) / inc) + 1)
        for r in self._scan.ranges[lo:hi]:
            if math.isfinite(r) and 0.0 < r < radius:
                return False
        return True

    def _twist_positioning(self):
        """Centre on the flag and stop at target distance."""
        bearing_tol = float(
            self.get_parameter("posicionamento_bearing_tolerance_rad").value)
        target_d = float(
            self.get_parameter("posicionamento_target_distance_m").value)
        w_gira = float(self.get_parameter("w_gira").value)
        v_slow = 0.5 * float(self.get_parameter("v_frente").value)

        t = Twist()
        if not self._flag_detected:
            # Lost sight while positioning: go back to EXPLORANDO. The
            # SM caller publishes the zero twist we return now.
            self._mission_state = _MISSION_EXPLORANDO
            self.get_logger().info(
                "[mission] lost flag during positioning → EXPLORANDO")
            return t

        bearing = self._raw_bearing
        distance = self._raw_distance

        if abs(bearing) > bearing_tol:
            # Centre first.
            t.angular.z = w_gira if bearing > 0 else -w_gira
            return t

        if math.isnan(distance) or distance > target_d + 0.05:
            # Close last centimetres slowly, straight ahead.
            t.linear.x = v_slow
            return t

        # Centred and at target distance: stop. Hold this state — the
        # robot has reached the deliverable goal of the assignment.
        return t

    # ---- Geometry helpers ------------------------------------------------

    def _target_from_detection(self):
        """World-frame target = robot pose + (bearing, distance).

        Used by NAVEGANDO_PARA_BANDEIRA to feed the GP tree a fresh
        target on every tick. Returns (None, None) if the LIDAR fusion
        failed (no usable range), in which case the previous target is
        left in place — the tree keeps driving toward where it last
        knew the flag was.
        """
        if math.isnan(self._raw_distance):
            return None, None
        rx, ry, ryaw = self._robot_pose_xyy()
        absolute_bearing = ryaw + self._raw_bearing
        tx = rx + self._raw_distance * math.cos(absolute_bearing)
        ty = ry + self._raw_distance * math.sin(absolute_bearing)
        return tx, ty

    def _set_target_xy(self, x: float, y: float):
        """Push target_x/y as ROS params (same path as the CTF SM)."""
        tx = float(self.get_parameter("target_x").value)
        ty = float(self.get_parameter("target_y").value)
        # Skip rclpy round-trip when the value is unchanged.
        if (x, y) == (tx, ty):
            return
        self.set_parameters([
            rclpy.parameter.Parameter(
                "target_x", rclpy.parameter.Parameter.Type.DOUBLE, x),
            rclpy.parameter.Parameter(
                "target_y", rclpy.parameter.Parameter.Type.DOUBLE, y),
        ])

    # ----------------------------------------------------------------------
    # Cartola LED — visual feedback in the Gazebo viewport
    # ----------------------------------------------------------------------

    def _on_mission_state_changed(self, prev, new):
        """Hook fired exactly once per mission-state transition."""
        if not bool(self.get_parameter("led_update_enabled").value):
            return
        rgb = self._led_rgb_for_state(new)
        if rgb is None:
            return
        # Run the ign service call on a background thread. Subprocess
        # roundtrip is ~100-300 ms; the 10 Hz tick budget is 100 ms;
        # blocking inline would chew up an entire tick worth of compute.
        import threading
        world  = str(self.get_parameter("world_name").value)
        model  = str(self.get_parameter("robot_model_name").value)
        link   = str(self.get_parameter("led_link_name").value)
        visual = str(self.get_parameter("led_visual_name").value)
        r, g, b = rgb
        def _worker():
            try:
                wr.set_visual_color(world, model, link, visual, r, g, b)
                self.get_logger().info(
                    f"[mission] LED → {new} (rgb={r:.2f},{g:.2f},{b:.2f})")
            except Exception as e:
                # Most common cause: gazebo not yet up, or the visual
                # path doesn't match the SDF. Log once and keep going —
                # the SM still works, only the visual feedback is lost.
                self.get_logger().warn(
                    f"[mission] LED update failed for {new}: {e}")
        threading.Thread(target=_worker, daemon=True).start()

    def _led_rgb_for_state(self, state):
        """Map a mission-state name to an RGB triple from the params."""
        if state == _MISSION_EXPLORANDO:
            param = "led_explorando_rgb"
        elif state == _MISSION_BANDEIRA_DETECTADA:
            param = "led_bandeira_detectada_rgb"
        elif state == _MISSION_NAVEGANDO_PARA_BANDEIRA:
            param = "led_navegando_rgb"
        elif state == _MISSION_POSICIONANDO_PARA_COLETA:
            param = "led_posicionando_rgb"
        else:
            return None
        triple = self.get_parameter(param).value
        if triple is None or len(triple) != 3:
            return None
        return float(triple[0]), float(triple[1]), float(triple[2])

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
