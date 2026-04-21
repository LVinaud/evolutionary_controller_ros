"""Gazebo-Fortress IPC helpers via `ign` CLI.

This module wraps `ign service` and `ign topic` calls in plain Python so we
don't depend on any ros_gz bridge configuration beyond what the professor
already ships in `prm_2026`. It is the only module in this package that
shells out.

Scope
    - `set_model_pose(world, name, x, y, yaw)` — teleports a model.
    - `query_model_poses(world, timeout_s)` — snapshots current poses of
      every model in the world (used by the orchestrator to discover the
      base/flag/deploy-zone positions without hardcoding anything).

Notes
    - Service name: `/world/<world>/set_pose`. Request type:
      `ignition.msgs.Pose`; reply: `ignition.msgs.Boolean`.
    - Topic for snapshots: `/world/<world>/pose/info`, type
      `ignition.msgs.Pose_V` (a repeated `pose` field).
"""
import math
import re
import subprocess


_POSE_RE = re.compile(
    r"pose\s*\{[^{}]*?name:\s*\"(?P<name>[^\"]+)\"[^{}]*?"
    r"position\s*\{[^{}]*?x:\s*(?P<x>-?[0-9.eE+-]+)[^{}]*?"
    r"y:\s*(?P<y>-?[0-9.eE+-]+)[^{}]*?"
    r"z:\s*(?P<z>-?[0-9.eE+-]+)[^{}]*?\}[^{}]*?"
    r"orientation\s*\{[^{}]*?x:\s*(?P<ox>-?[0-9.eE+-]+)[^{}]*?"
    r"y:\s*(?P<oy>-?[0-9.eE+-]+)[^{}]*?"
    r"z:\s*(?P<oz>-?[0-9.eE+-]+)[^{}]*?"
    r"w:\s*(?P<ow>-?[0-9.eE+-]+)[^{}]*?\}",
    re.DOTALL,
)


def set_model_pose(
    world: str,
    name: str,
    x: float,
    y: float,
    yaw: float = 0.0,
    z: float = 0.0,
    timeout_ms: int = 2000,
) -> None:
    """Teleport a model (robot, flag, ...) to (x, y, z, yaw)."""
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    req = (
        f'name: "{name}", '
        f"position: {{x: {x}, y: {y}, z: {z}}}, "
        f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}"
    )
    cmd = [
        "ign", "service",
        "-s", f"/world/{world}/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", str(timeout_ms),
        "--req", req,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
    if result.returncode != 0:
        raise RuntimeError(
            f"ign set_pose failed (rc={result.returncode}): {result.stderr}")


def query_model_poses(world: str, timeout_s: float = 3.0) -> dict:
    """Snapshot (x, y, yaw) of every model in the world.

    Subscribes to `/world/<world>/pose/info` for up to `timeout_s` and parses
    the first non-empty message. Returns {model_name: (x, y, yaw)}.
    """
    cmd = [
        "ign", "topic",
        "-e", "-t", f"/world/{world}/pose/info",
        "-n", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=timeout_s + 2.0)
    if result.returncode != 0:
        raise RuntimeError(
            f"ign topic echo failed (rc={result.returncode}): {result.stderr}")

    poses = {}
    for m in _POSE_RE.finditer(result.stdout):
        name = m.group("name")
        x = float(m.group("x"))
        y = float(m.group("y"))
        qz = float(m.group("oz"))
        qw = float(m.group("ow"))
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        poses[name] = (x, y, yaw)
    if not poses:
        raise RuntimeError(
            f"no poses parsed from /world/{world}/pose/info output")
    return poses
