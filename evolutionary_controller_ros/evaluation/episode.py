"""Run one go-to-goal episode and collect the `history` dict for fitness.py.

ROS2 helper (not a node) used by `orchestrator.py`. Assumes:
    - a long-running `gp_controller` node is up and subscribed to /odom,
      /scan, its `genome_json`, `target_x`, `target_y` ROS parameters.
    - `/gp_controller/reset` is available to clear per-episode state.
    - the prm_2026 topics (`/odom_gt`, `/scan`, `/clock`) are alive.

Per-episode flow
    1. SetParameters on gp_controller: genome_json + target_x/y.
    2. Call `/gp_controller/reset`.
    3. `set_model_pose` for the static drift-prone models, then for the
       robot itself (spawning slightly above ground so wheels settle).
    4. Spin for `duration_s`, collecting per-tick:
         - robot pose (for distance-to-target tracking)
         - scan min (for collision detection, debounced)
    5. Early stop when the robot is within `reach_radius_m` of the target.
    6. Return the history dict expected by `fitness.compute_fitness_cases`.

No camera, no flag, no holding_flag — those are owned by the CTF state
machine in `gp_controller` ("ctf" mode), which is out of scope for the
training episode.
"""
import math
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from rcl_interfaces.srv import SetParameters

from . import world_reset as wr
from ..evolution import genome as g


@dataclass
class ScenarioConfig:
    name: str                        # e.g. "from_base_to_center"
    robot_start: tuple               # (x, y, yaw)
    target_pose: tuple               # (x, y) — the goal for this scenario
    duration_s: float = 30.0
    tick_hz: float = 10.0


@dataclass
class WorldConfig:
    world_name: str                  # "capture_the_flag_world"
    robot_model_name: str            # e.g. "prm_robot"
    collision_radius_m: float = 0.25
    collision_debounce_s: float = 0.5
    reach_radius_m: float = 0.5      # early-stop + "alcancou_alvo" threshold
    robot_spawn_z_m: float = 0.3     # drop from this height so wheels settle
    # (model_name, x, y, yaw) of non-static world models that drift during
    # episodes. Teleported back to their snapshot pose on every reset.
    static_models_to_reset: tuple = ()


def run_episode(
    node: Node,
    genome_tree: dict,
    scenario: ScenarioConfig,
    world: WorldConfig,
    timing_log=None,
    timing_extras: dict | None = None,
) -> dict:
    """Run one episode and return its history dict.

    `timing_log` (optional `TimingLog`) collects fine-grained per-phase
    timings for offline analysis. `timing_extras` is a dict merged into
    every event emitted by this episode (e.g. `{"gen": 3, "individual":
    12}`) so rows can be grouped later.
    """
    collector = _EpisodeCollector(node, scenario, world,
                                  timing_log=timing_log,
                                  timing_extras=timing_extras or {})
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
                 world: WorldConfig, *,
                 timing_log=None, timing_extras: dict | None = None):
        self.node = node
        self.scenario = scenario
        self.world = world
        self._timing = timing_log
        self._timing_extras = dict(timing_extras or {})
        self._timing_extras.setdefault("scenario", scenario.name)

        self._set_params_client = node.create_client(
            SetParameters, "/gp_controller/set_parameters")
        self._reset_client = node.create_client(
            Empty, "/gp_controller/reset")

        self._odom_sub = node.create_subscription(
            Odometry, "/odom_gt", self._on_odom, 10)
        self._scan_sub = node.create_subscription(
            LaserScan, "/scan", self._on_scan, 10)

        self._reset_state()

    def _record(self, category: str, duration_s: float, **extras):
        if self._timing is None:
            return
        self._timing.event(category, duration_s,
                           **self._timing_extras, **extras)

    def teardown(self):
        self.node.destroy_subscription(self._odom_sub)
        self.node.destroy_subscription(self._scan_sub)

    # ----------------------------------------------------------------------

    def _reset_state(self):
        self._odom_msg: Optional[Odometry] = None
        self._scan_msg: Optional[LaserScan] = None

        self._positions_xy: list = []
        self._min_dist_target = math.inf
        self._reached_target = False
        self._collision_events = 0
        self._last_collision_ts: Optional[float] = None
        self._time_to_reach_s: Optional[float] = None

    # ----------------------------------------------------------------------
    # Subscription callbacks
    # ----------------------------------------------------------------------

    def _on_odom(self, msg):
        self._odom_msg = msg

    def _on_scan(self, msg):
        self._scan_msg = msg

    # ----------------------------------------------------------------------
    # Setup: load genome + target, reset, teleport world
    # ----------------------------------------------------------------------

    def setup(self, genome_tree: dict):
        self._reset_state()

        tx, ty = self.scenario.target_pose
        params = [
            _str_param("genome_json", g.to_json(genome_tree)),
            _dbl_param("target_x", tx),
            _dbl_param("target_y", ty),
        ]
        t0 = time.monotonic()
        self._call_set_parameters(params)
        self._record("setup_param_push", time.monotonic() - t0)

        t0 = time.monotonic()
        self._call_reset()
        self._record("setup_reset_call", time.monotonic() - t0)

        t0 = time.monotonic()
        for model_name, mx, my, myaw in self.world.static_models_to_reset:
            wr.set_model_pose(self.world.world_name, model_name,
                              mx, my, myaw)
        self._record("setup_teleport_models", time.monotonic() - t0,
                     models_count=len(self.world.static_models_to_reset))

        rx, ry, ryaw = self.scenario.robot_start
        t0 = time.monotonic()
        wr.set_model_pose(self.world.world_name,
                          self.world.robot_model_name,
                          rx, ry, ryaw,
                          z=self.world.robot_spawn_z_m)
        self._record("setup_teleport_robot", time.monotonic() - t0)

    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------

    def run(self) -> dict:
        dt = 1.0 / self.scenario.tick_hz
        t_start = time.monotonic()
        t_end = t_start + self.scenario.duration_s
        last_tick = t_start
        ticks = 0
        spin_total = 0.0

        while time.monotonic() < t_end:
            t_spin0 = time.monotonic()
            rclpy.spin_once(self.node, timeout_sec=dt)
            spin_total += time.monotonic() - t_spin0
            now = time.monotonic()
            if now - last_tick >= dt:
                last_tick = now
                self._tick(now - t_start)
                ticks += 1
                if self._reached_target:
                    break

        elapsed = time.monotonic() - t_start
        self._record(
            "episode_spin", elapsed,
            ticks_collected=ticks,
            early_stop=bool(self._reached_target),
            duration_cap_s=self.scenario.duration_s,
            spin_once_total_s=spin_total,
            collisions=self._collision_events,
            min_dist_target_m=(self._min_dist_target
                               if math.isfinite(self._min_dist_target)
                               else -1.0),
        )
        return self._build_history(elapsed=elapsed)

    def _tick(self, elapsed_s: float):
        if self._odom_msg is None:
            return

        pos = self._odom_msg.pose.pose.position
        self._positions_xy.append((pos.x, pos.y))

        tx, ty = self.scenario.target_pose
        d = math.hypot(pos.x - tx, pos.y - ty)
        if d < self._min_dist_target:
            self._min_dist_target = d
        if d < self.world.reach_radius_m and not self._reached_target:
            self._reached_target = True
            self._time_to_reach_s = elapsed_s

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
            "target_pose": self.scenario.target_pose,
            "min_dist_target_m": (self._min_dist_target
                                  if math.isfinite(self._min_dist_target)
                                  else self.world.reach_radius_m * 1e3),
            "reached_target": self._reached_target,
            "collision_events": self._collision_events,
            "time_to_reach_s": self._time_to_reach_s,
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
