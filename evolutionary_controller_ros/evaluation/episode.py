"""Run one CTF episode and collect the `history` dict expected by fitness.py.

This module is a ROS2 helper (NOT a node) used by `orchestrator.py`. It
assumes:
    - a long-running `gp_controller` node is already up and idle.
    - a `/gp_controller/reset` service is available to clear per-episode
      state between runs.
    - the usual prm_2026 topics (`/odom_gt`, `/scan`, `/clock`) are alive.

Per-episode flow
    1. SetParameters on gp_controller: genome_json + target poses + our_team.
    2. Call `/gp_controller/reset`.
    3. `set_model_pose` to teleport robot to scenario start pose and the
       enemy flag back to its "initial" pose.
    4. Spin for scenario_time_s, collecting a tick-by-tick history:
         - robot pose (for exploration grid & colision debounce)
         - `/gp_controller/holding_flag` (for pegou/entregou events)
         - scan min (for collision detection)
         - camera flag visibility (latched from labels_map if available)
    5. Early stop when the `delivered` event fires.
    6. Return the history dict (shape defined by fitness.compute_fitness_cases).

The caller composes N scenarios' vectors into the 14-case vector the GA
consumes; this module owns one scenario.
"""
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from rcl_interfaces.srv import SetParameters

from . import world_reset as wr
from ..evolution import genome as g


@dataclass
class ScenarioConfig:
    name: str                        # e.g. "from_base"
    robot_start: tuple               # (x, y, yaw)
    enemy_flag_initial: tuple        # (x, y) — where the flag starts
    duration_s: float = 30.0
    tick_hz: float = 10.0


@dataclass
class WorldConfig:
    world_name: str                  # "capture_the_flag_world"
    robot_model_name: str            # e.g. "prm_robot"
    enemy_flag_model_name: str       # "blue_flag" or "red_flag"
    our_team: str                    # "blue" | "red"
    enemy_flag_pose: tuple           # initial (x, y) of the enemy flag
    own_base_pose: tuple
    enemy_base_pose: tuple
    deploy_zone_pose: tuple
    collision_radius_m: float = 0.25
    collision_debounce_s: float = 0.5
    deploy_zone_radius_m: float = 1.0
    exploration_cell_size_m: float = 0.5


def run_episode(
    node: Node,
    genome_tree: dict,
    scenario: ScenarioConfig,
    world: WorldConfig,
) -> dict:
    collector = _EpisodeCollector(node, scenario, world)
    try:
        collector.setup(genome_tree)
        return collector.run()
    finally:
        collector.teardown()


# ==========================================================================
# Internal collector
# ==========================================================================

class _EpisodeCollector:
    def __init__(self, node: Node, scenario: ScenarioConfig,
                 world: WorldConfig):
        self.node = node
        self.scenario = scenario
        self.world = world

        self._set_params_client = node.create_client(
            SetParameters, "/gp_controller/set_parameters")
        self._reset_client = node.create_client(
            Empty, "/gp_controller/reset")

        self._odom_sub = node.create_subscription(
            Odometry, "/odom_gt", self._on_odom, 10)
        self._scan_sub = node.create_subscription(
            LaserScan, "/scan", self._on_scan, 10)
        self._labels_sub = node.create_subscription(
            Image, "/robot_cam/labels_map", self._on_labels, 10)
        self._holding_sub = node.create_subscription(
            Bool, "/gp_controller/holding_flag", self._on_holding, 10)

        self._reset_state()

    def teardown(self):
        self.node.destroy_subscription(self._odom_sub)
        self.node.destroy_subscription(self._scan_sub)
        self.node.destroy_subscription(self._labels_sub)
        self.node.destroy_subscription(self._holding_sub)

    # ----------------------------------------------------------------------

    def _reset_state(self):
        self._odom_msg: Optional[Odometry] = None
        self._scan_msg: Optional[LaserScan] = None
        self._labels_frame: Optional[np.ndarray] = None
        self._holding_flag = False
        self._prev_holding = False

        self._positions_xy: list = []
        self._flag_visible_ticks = 0
        self._holding_any_tick = False
        self._delivered = False
        self._min_dist_deploy_holding = math.inf
        self._collision_events = 0
        self._last_collision_ts: Optional[float] = None
        self._time_to_deliver_s: Optional[float] = None

    # ----------------------------------------------------------------------
    # Subscription callbacks
    # ----------------------------------------------------------------------

    def _on_odom(self, msg):
        self._odom_msg = msg

    def _on_scan(self, msg):
        self._scan_msg = msg

    def _on_labels(self, msg):
        if msg.encoding not in ("mono8", "8UC1"):
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        self._labels_frame = buf.reshape(msg.height, msg.width)

    def _on_holding(self, msg):
        self._holding_flag = bool(msg.data)

    # ----------------------------------------------------------------------
    # Setup: load genome + target poses, reset, teleport world
    # ----------------------------------------------------------------------

    def setup(self, genome_tree: dict):
        self._reset_state()

        fx, fy = self.world.enemy_flag_pose
        ob_x, ob_y = self.world.own_base_pose
        eb_x, eb_y = self.world.enemy_base_pose
        dz_x, dz_y = self.world.deploy_zone_pose
        params = [
            _str_param("genome_json", g.to_json(genome_tree)),
            _str_param("our_team", self.world.our_team),
            _dbl_param("enemy_flag_x", fx),
            _dbl_param("enemy_flag_y", fy),
            _dbl_param("own_base_x",   ob_x),
            _dbl_param("own_base_y",   ob_y),
            _dbl_param("enemy_base_x", eb_x),
            _dbl_param("enemy_base_y", eb_y),
            _dbl_param("deploy_zone_x", dz_x),
            _dbl_param("deploy_zone_y", dz_y),
        ]
        self._call_set_parameters(params)
        self._call_reset()

        rx, ry, ryaw = self.scenario.robot_start
        wr.set_model_pose(self.world.world_name,
                          self.world.robot_model_name,
                          rx, ry, ryaw)
        flag_x0, flag_y0 = self.scenario.enemy_flag_initial
        wr.set_model_pose(self.world.world_name,
                          self.world.enemy_flag_model_name,
                          flag_x0, flag_y0, 0.0)

    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------

    def run(self) -> dict:
        dt = 1.0 / self.scenario.tick_hz
        t_start = time.monotonic()
        t_end = t_start + self.scenario.duration_s
        last_tick = t_start

        while time.monotonic() < t_end:
            rclpy.spin_once(self.node, timeout_sec=dt)
            now = time.monotonic()
            if now - last_tick >= dt:
                last_tick = now
                self._tick(now - t_start)
                if self._delivered:
                    break

        return self._build_history(elapsed=time.monotonic() - t_start)

    def _tick(self, elapsed_s: float):
        if self._odom_msg is None:
            return

        pos = self._odom_msg.pose.pose.position
        self._positions_xy.append((pos.x, pos.y))

        if self._holding_flag:
            self._holding_any_tick = True
            dx = pos.x - self.world.deploy_zone_pose[0]
            dy = pos.y - self.world.deploy_zone_pose[1]
            d = math.hypot(dx, dy)
            if d < self._min_dist_deploy_holding:
                self._min_dist_deploy_holding = d

            wr.set_model_pose(self.world.world_name,
                              self.world.enemy_flag_model_name,
                              pos.x, pos.y, 0.0)

        if self._prev_holding and not self._holding_flag:
            dx = pos.x - self.world.deploy_zone_pose[0]
            dy = pos.y - self.world.deploy_zone_pose[1]
            if math.hypot(dx, dy) < self.world.deploy_zone_radius_m:
                self._delivered = True
                self._time_to_deliver_s = elapsed_s
        self._prev_holding = self._holding_flag

        if self._labels_frame is not None:
            enemy_label = 20 if self.world.our_team == "blue" else 25
            if bool((self._labels_frame == enemy_label).any()):
                self._flag_visible_ticks += 1

        if self._scan_msg is not None:
            ranges = [r for r in self._scan_msg.ranges
                      if math.isfinite(r) and r > 0.0]
            if ranges and min(ranges) < self.world.collision_radius_m:
                now_mono = time.monotonic()
                if (self._last_collision_ts is None or
                        now_mono - self._last_collision_ts
                        >= self.world.collision_debounce_s):
                    self._collision_events += 1
                    self._last_collision_ts = now_mono

    def _build_history(self, elapsed: float) -> dict:
        return {
            "positions_xy": self._positions_xy,
            "dt_s": 1.0 / self.scenario.tick_hz,
            "flag_visible_ticks": self._flag_visible_ticks,
            "holding_any_tick": self._holding_any_tick,
            "delivered": self._delivered,
            "min_dist_deploy_holding": self._min_dist_deploy_holding,
            "collision_events": self._collision_events,
            "time_to_deliver_s": self._time_to_deliver_s,
            "scenario_time_s": self.scenario.duration_s,
            "elapsed_s": elapsed,
        }

    # ----------------------------------------------------------------------
    # Parameter and service plumbing
    # ----------------------------------------------------------------------

    def _call_set_parameters(self, params, timeout_s: float = 3.0):
        if not self._set_params_client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError("gp_controller /set_parameters not available")
        req = SetParameters.Request()
        req.parameters = params
        future = self._set_params_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future,
                                         timeout_sec=timeout_s)
        res = future.result()
        if res is None:
            raise RuntimeError("set_parameters call timed out")
        for r in res.results:
            if not r.successful:
                raise RuntimeError(f"set_parameters rejected: {r.reason}")

    def _call_reset(self, timeout_s: float = 3.0):
        if not self._reset_client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError("gp_controller /reset not available")
        future = self._reset_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self.node, future,
                                         timeout_sec=timeout_s)


# ==========================================================================
# Parameter helpers (for SetParameters service)
# ==========================================================================

def _str_param(name: str, value: str):
    p = Parameter(name, Parameter.Type.STRING, value)
    return p.to_parameter_msg()


def _dbl_param(name: str, value: float):
    p = Parameter(name, Parameter.Type.DOUBLE, float(value))
    return p.to_parameter_msg()
