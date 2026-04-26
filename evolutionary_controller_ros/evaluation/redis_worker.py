"""Redis-relayed evaluation worker.

Same job as `worker_server.py` (run `run_episode` against the local
Gazebo+gp_controller and return a fitness vector), but the trigger is a
**job pulled from Redis**, not an inbound HTTP request. Designed for
networks where peer-to-peer connectivity is blocked (eduroam client
isolation, locked-down corporate VLANs, etc.) but outbound HTTPS to a
hosted Redis is allowed.

Architecture
    coordinator                hosted Redis            worker(s)
    ───────────                ─────────────            ─────────
                                                        ┌──────────┐
        RPUSH evo:jobs ───────► [evo:jobs queue]        │ this node│
                                       │  BLPOP         │  (idle)  │
                                       │ ◄──────────────┤          │
                                       │  drain & run   │ run_     │
                                       │     episode    │ episode  │
                                       │ ◄──────────────┤          │
                                                        │ (busy)   │
        BLPOP evo:results ◄────[evo:results queue]      │          │
                                       │ ◄──────────────┤ RPUSH    │
                                                        └──────────┘

Wire format (JSON in both directions)
    job:    {"job_id", "individual", "scenario": {name, robot_start,
             target_pose, duration_s, tick_hz}, "world": {...},
             "enabled_metrics": [...]}
    result: {"job_id", "case_vector": [...], "history": {...},
             "wall_time_s", "error": null}  (or "error": "..." on failure)

Failure mode: if `run_episode` raises, the worker still pushes a result
with `error` populated and a sentinel case vector. The coordinator
treats that as a failed individual (lexicase-rejected).

CLI
    REDIS_URL=rediss://default:<token>@<host>:<port> \\
    ros2 run evolutionary_controller_ros redis_worker

The URL can also be passed as `--redis-url`. Workers loop forever; stop
with Ctrl+C. They are stateless — restart at will.
"""
import argparse
import json
import os
import signal
import sys
import time
import traceback

try:
    import redis
except ImportError as e:
    raise ImportError(
        "redis_worker requires the redis-py package — install with "
        "`pip install redis`"
    ) from e

import rclpy
from rclpy.node import Node

from . import episode as ep
from ..evolution import fitness as fit
from ..evolution import genome as g


# Queue names (shared across all coordinators / workers — keep simple).
# Per-run namespacing is done at the payload level via job_id prefix.
_JOBS_QUEUE = "evo:jobs"
_RESULTS_QUEUE = "evo:results"

# How long BLPOP blocks before returning None and looping. Short enough
# that Ctrl+C is responsive, long enough that we don't spam Redis.
_BLPOP_TIMEOUT_S = 5


# --------------------------------------------------------------------------
# Worker node
# --------------------------------------------------------------------------

class RedisWorkerNode(Node):
    def __init__(self, redis_url: str):
        super().__init__("redis_evaluator_worker")
        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._stopping = False
        try:
            self._redis.ping()
        except redis.RedisError as e:
            raise RuntimeError(f"could not connect to redis at {redis_url}: {e}") from e
        self.get_logger().info(
            f"connected to redis; waiting for jobs on '{_JOBS_QUEUE}'")

    # ----------------------------------------------------------------------

    def request_stop(self):
        self._stopping = True

    def run_forever(self):
        n_done = 0
        while not self._stopping:
            try:
                item = self._redis.blpop(_JOBS_QUEUE, timeout=_BLPOP_TIMEOUT_S)
            except redis.RedisError as e:
                self.get_logger().error(
                    f"redis blpop failed: {e}; sleeping 5s and retrying")
                time.sleep(5)
                continue

            if item is None:
                continue   # idle; loop and check _stopping
            _key, payload = item
            try:
                job = json.loads(payload)
            except json.JSONDecodeError as e:
                self.get_logger().error(f"bad job payload (skipped): {e}")
                continue

            job_id = job.get("job_id", "<no-id>")
            self.get_logger().info(f"picked job {job_id}")
            result = self._handle_job(job)
            try:
                self._redis.rpush(_RESULTS_QUEUE, json.dumps(result))
            except redis.RedisError as e:
                self.get_logger().error(
                    f"failed to push result for {job_id}: {e}; result LOST")
                continue
            n_done += 1
            self.get_logger().info(
                f"finished job {job_id} ({n_done} done since boot)")

        self.get_logger().info("worker loop exiting cleanly")

    # ----------------------------------------------------------------------

    def _handle_job(self, job: dict) -> dict:
        """Run one evaluation. Always returns a result dict (never raises)."""
        job_id = job.get("job_id", "<no-id>")
        t0 = time.monotonic()
        try:
            scenario = ep.ScenarioConfig(
                name=job["scenario"]["name"],
                robot_start=tuple(job["scenario"]["robot_start"]),
                target_pose=tuple(job["scenario"]["target_pose"]),
                duration_s=float(job["scenario"]["duration_s"]),
                tick_hz=float(job["scenario"]["tick_hz"]),
            )
            world = ep.WorldConfig(
                world_name=job["world"].get("world_name", "capture_the_flag_world"),
                robot_model_name=job["world"].get("robot_model_name", "prm_robot"),
                collision_radius_m=float(job["world"].get("collision_radius_m", 0.25)),
                collision_debounce_s=float(job["world"].get("collision_debounce_s", 0.5)),
                reach_radius_m=float(job["world"].get("reach_radius_m", 0.5)),
                robot_spawn_z_m=float(job["world"].get("robot_spawn_z_m", 0.3)),
                static_models_to_reset=tuple(
                    tuple(m) for m in job["world"].get("static_models_to_reset", [])),
            )
            tree = g.from_json(job["genome_json"])
            history = ep.run_episode(self, tree, scenario, world)
            case_vector = fit.compute_fitness_cases(
                history, enabled_metrics=tuple(job["enabled_metrics"]))
            return {
                "job_id": job_id,
                "case_vector": [float(c) for c in case_vector],
                "history": {
                    "min_dist_target_m": float(history["min_dist_target_m"]),
                    "reached_target": bool(history["reached_target"]),
                    "collision_events": int(history["collision_events"]),
                    "elapsed_s": float(history["elapsed_s"]),
                },
                "wall_time_s": time.monotonic() - t0,
                "error": None,
            }
        except Exception as e:
            self.get_logger().error(
                f"job {job_id} FAILED: {e!r}\n{traceback.format_exc()}")
            return {
                "job_id": job_id,
                "case_vector": [],
                "history": {},
                "wall_time_s": time.monotonic() - t0,
                "error": f"{type(e).__name__}: {e}",
            }


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Redis-relayed fitness evaluator worker.")
    parser.add_argument("--redis-url",
                        default=os.environ.get("REDIS_URL"),
                        help="Redis URL (rediss://user:pass@host:port). "
                             "Defaults to $REDIS_URL.")
    args, _ = parser.parse_known_args()

    if not args.redis_url:
        print("ERROR: pass --redis-url or set REDIS_URL env var",
              file=sys.stderr)
        sys.exit(2)

    rclpy.init()
    node = RedisWorkerNode(args.redis_url)

    def _on_signal(signum, _frame):
        node.get_logger().info(
            f"got signal {signum}, finishing current job and exiting")
        node.request_stop()
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        node.run_forever()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
