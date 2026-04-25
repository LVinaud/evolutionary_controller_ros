"""HTTP server that exposes one machine as an evaluation worker.

Architecture
------------
A worker PC runs Gazebo + the prm_2026 robot stack + `gp_controller` (all
unchanged from the single-machine setup) AND this server. The server is a
single Python process that:

    * acts as a regular ROS2 node (so it can talk to the local
      `gp_controller` via SetParameters / `/reset`, and subscribe to the
      local `/odom_gt` and `/scan`),
    * exposes an HTTP API on a TCP port (default 8000) that accepts
      remote requests of the form "evaluate this genome on this scenario"
      and returns the lexicase case vector.

Requests are processed strictly serially (a `threading.Lock` makes
concurrent `/evaluate` calls return HTTP 503 instead of stepping on each
other). Each evaluation reuses `evaluation.episode.run_episode` end to
end — there is no behavioural difference vs. single-machine mode; only
the trigger source changes from "Python function call" to "HTTP POST".

Why one process and not two?
    `run_episode` needs an `rclpy.Node` to publish parameters and spin
    subscriptions. Sharing the same node across an HTTP boundary requires
    either two processes plus an IPC layer, or a single process where
    FastAPI request handlers run in a threadpool and reach into the
    rclpy node directly. The latter is materially simpler and is what
    we do here. FastAPI's default behaviour for synchronous handlers
    (defined with `def`, not `async def`) is to dispatch them onto a
    threadpool — that thread can safely call blocking rclpy primitives
    like `spin_until_future_complete` because the lock guarantees only
    one such call at a time.

CLI
    python3 -m evolutionary_controller_ros.evaluation.worker_server \\
        --host 0.0.0.0 --port 8000

When invoked through the colcon `console_scripts` entry point the same
script is exposed as `ros2 run evolutionary_controller_ros worker_server`.
"""
import argparse
import threading
import time
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError as e:
    raise ImportError(
        "worker_server requires fastapi, uvicorn, and pydantic — install with "
        "`pip install fastapi uvicorn`"
    ) from e

import rclpy
from rclpy.node import Node

from . import episode as ep
from ..evolution import fitness as fit
from ..evolution import genome as g


# --------------------------------------------------------------------------
# Request / response schemas (validated by Pydantic)
# --------------------------------------------------------------------------

class ScenarioRequest(BaseModel):
    name: str
    robot_start: list[float]   # [x, y, yaw]
    target_pose: list[float]   # [x, y]
    duration_s: float = 30.0
    tick_hz: float = 10.0


class WorldRequest(BaseModel):
    world_name: str = "capture_the_flag_world"
    robot_model_name: str = "prm_robot"
    collision_radius_m: float = 0.25
    collision_debounce_s: float = 0.5
    reach_radius_m: float = 0.5
    robot_spawn_z_m: float = 0.3
    # Each entry is [name, x, y, yaw] for a drift-prone model the worker
    # should teleport back into place at the start of every episode.
    static_models_to_reset: list[list[Any]] = []


class EvaluateRequest(BaseModel):
    genome_json: str
    scenario: ScenarioRequest
    world: WorldRequest
    enabled_metrics: list[str]


class EvaluateResponse(BaseModel):
    case_vector: list[float]
    history: dict[str, Any]
    wall_time_s: float


class HealthResponse(BaseModel):
    status: str           # "idle" | "busy"
    uptime_s: float
    last_eval_s: float    # wall time of the most recent /evaluate, 0 if none yet


# --------------------------------------------------------------------------
# Worker node
# --------------------------------------------------------------------------

class WorkerNode(Node):
    """rclpy.Node + state shared between FastAPI handlers."""

    def __init__(self):
        super().__init__("evaluator_worker")
        self._lock = threading.Lock()
        self._busy = False
        self._last_eval_s = 0.0
        self._t_boot = time.monotonic()
        self._eval_count = 0

    # ----------------------------------------------------------------------

    def evaluate(self, req: EvaluateRequest) -> EvaluateResponse:
        scenario = ep.ScenarioConfig(
            name=req.scenario.name,
            robot_start=tuple(req.scenario.robot_start),
            target_pose=tuple(req.scenario.target_pose),
            duration_s=req.scenario.duration_s,
            tick_hz=req.scenario.tick_hz,
        )
        world = ep.WorldConfig(
            world_name=req.world.world_name,
            robot_model_name=req.world.robot_model_name,
            collision_radius_m=req.world.collision_radius_m,
            collision_debounce_s=req.world.collision_debounce_s,
            reach_radius_m=req.world.reach_radius_m,
            robot_spawn_z_m=req.world.robot_spawn_z_m,
            static_models_to_reset=tuple(
                tuple(m) for m in req.world.static_models_to_reset),
        )
        tree = g.from_json(req.genome_json)

        t0 = time.monotonic()
        history = ep.run_episode(self, tree, scenario, world)
        elapsed = time.monotonic() - t0

        case_vector = fit.compute_fitness_cases(
            history, enabled_metrics=tuple(req.enabled_metrics))
        self._last_eval_s = elapsed
        self._eval_count += 1

        # `history` contains positions_xy (potentially huge) + non-JSONable
        # floats; we pick out the headline keys and ship them.
        return EvaluateResponse(
            case_vector=[float(c) for c in case_vector],
            history={
                "min_dist_target_m": float(history["min_dist_target_m"]),
                "reached_target": bool(history["reached_target"]),
                "collision_events": int(history["collision_events"]),
                "elapsed_s": float(history["elapsed_s"]),
            },
            wall_time_s=elapsed,
        )

    # ----------------------------------------------------------------------

    def health(self) -> HealthResponse:
        return HealthResponse(
            status=("busy" if self._busy else "idle"),
            uptime_s=time.monotonic() - self._t_boot,
            last_eval_s=self._last_eval_s,
        )


# --------------------------------------------------------------------------
# FastAPI app builder
# --------------------------------------------------------------------------

def make_app(node: WorkerNode) -> "FastAPI":
    app = FastAPI(title="evolutionary_controller_ros worker",
                  description="One machine = one evaluation worker.")

    @app.post("/evaluate", response_model=EvaluateResponse)
    def evaluate(req: EvaluateRequest):
        if not node._lock.acquire(blocking=False):
            raise HTTPException(503, detail="evaluator busy")
        try:
            node._busy = True
            return node.evaluate(req)
        except ValueError as e:
            raise HTTPException(400, detail=str(e))
        finally:
            node._busy = False
            node._lock.release()

    @app.get("/health", response_model=HealthResponse)
    def health():
        return node.health()

    return app


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HTTP-exposed fitness evaluator (one worker = one machine).")
    parser.add_argument("--host", default="0.0.0.0",
                        help="bind address (default: all interfaces)")
    parser.add_argument("--port", type=int, default=8000,
                        help="bind port (default: 8000)")
    parser.add_argument("--log-level", default="info",
                        help="uvicorn log level (default: info)")
    # rclpy parses any --ros-args style flags from sys.argv; argparse
    # handles only the ones we declare.
    args, _ros_args = parser.parse_known_args()

    rclpy.init()
    node = WorkerNode()
    node.get_logger().info(
        f"worker_server up on {args.host}:{args.port} — node is /evaluator_worker")

    app = make_app(node)

    try:
        uvicorn.run(app, host=args.host, port=args.port,
                    log_level=args.log_level)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
