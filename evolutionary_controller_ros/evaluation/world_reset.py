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
import subprocess


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


def set_physics(
    world: str,
    real_time_factor: float,
    max_step_size_s: float = 0.001,
    timeout_ms: int = 2000,
) -> None:
    """Set the world's real-time factor via `/world/<world>/set_physics`.

    Use values > 1 to run faster than wall time (trading simulation
    fidelity for throughput). Fortress clamps to whatever the CPU can
    actually sustain; if the requested RTF is above what the host can
    do, the true RTF degrades to ~max achievable.
    """
    req = (
        f"max_step_size: {max_step_size_s}, "
        f"real_time_factor: {real_time_factor}, "
        f"real_time_update_rate: {1.0 / max_step_size_s}"
    )
    cmd = [
        "ign", "service",
        "-s", f"/world/{world}/set_physics",
        "--reqtype", "ignition.msgs.Physics",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", str(timeout_ms),
        "--req", req,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5.0)
    if result.returncode != 0:
        raise RuntimeError(
            f"ign set_physics failed (rc={result.returncode}): {result.stderr}")


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

    poses = _parse_pose_v(result.stdout)
    if not poses:
        raise RuntimeError(
            f"no poses parsed from /world/{world}/pose/info output")
    return poses


def _parse_pose_v(text: str) -> dict:
    """Parse the text form of an `ignition.msgs.Pose_V` dump.

    The message encodes each model as a block:

        pose {
          name: "foo"
          id: 42
          position { x: 1  y: 2  z: 0 }
          orientation { x: 0 y: 0 z: 0.7 w: 0.7 }
          header { ... }
        }

    Top-level `pose {` blocks are collected with a brace counter (regex
    fails here because of the nested `position`/`orientation` groups).
    Inside each block a sub-state tracks whether we're inside
    `position { ... }` or `orientation { ... }` so we know which x/y/z
    assignment to consume.
    """
    poses: dict = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "pose {" or line.startswith("pose {"):
            i, block = _read_block(lines, i)
            parsed = _parse_pose_block(block)
            if parsed is not None:
                name, x, y, qz, qw = parsed
                yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
                poses[name] = (x, y, yaw)
        else:
            i += 1
    return poses


def _read_block(lines: list, start: int) -> tuple:
    """Return (next_index, body_lines) for the `{ ... }` block opened on `start`."""
    depth = 0
    body = []
    i = start
    while i < len(lines):
        line = lines[i]
        opens = line.count("{")
        closes = line.count("}")
        depth += opens - closes
        body.append(line)
        i += 1
        if depth == 0:
            break
    return i, body


def _parse_pose_block(block_lines: list) -> tuple | None:
    """Extract (name, x, y, qz, qw) from a single `pose { ... }` block."""
    name = None
    x = y = 0.0
    qz = 0.0
    qw = 1.0
    section = None  # None | "position" | "orientation"
    section_depth = 0

    for raw in block_lines:
        line = raw.strip()
        if line.startswith("name:"):
            name = line.split('"', 2)[1] if '"' in line else None
            continue
        if line.startswith("position {"):
            section = "position"
            section_depth = 1
            continue
        if line.startswith("orientation {"):
            section = "orientation"
            section_depth = 1
            continue
        if section is not None:
            section_depth += line.count("{") - line.count("}")
            if section_depth <= 0:
                section = None
                continue
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            try:
                v = float(val)
            except ValueError:
                continue
            if section == "position":
                if key == "x": x = v
                elif key == "y": y = v
            elif section == "orientation":
                if key == "z": qz = v
                elif key == "w": qw = v

    if name is None:
        return None
    return name, x, y, qz, qw
