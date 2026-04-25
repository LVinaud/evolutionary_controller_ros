"""Run the best evolved genome through the CTF mission.

The trained genome only knows reactive go-to-goal navigation. The CTF
sequence (go to enemy base → go to enemy flag → grab → return home →
release) is the hardcoded state machine inside `gp_controller` (mode
"ctf"). This launch wires both: it reads the genome JSON from disk and
sets the base/flag coordinates from the CTF world.

Usage:
    # 1) start Gazebo + robot in another terminal:
    #    ros2 launch prm_2026 carrega_robo.launch.py
    # 2) launch the CTF run:
    ros2 launch evolutionary_controller_ros demo_best.launch.py \\
        genome:=/abs/path/to/best.json
    # default: <pkg>/genomes/best.json (search order below)

Coordinates default to the `arena_cilindros` CTF world. Override any
of them with launch args (own_base_x, enemy_flag_y, etc.) to use a
different world.

Side: by default the robot is on the BLUE team — its own_base is
(6, 0), enemy_base is (-6, 0), enemy_flag is (-8, 0). To play as RED
swap the values via launch args.
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# CTF world coordinates from src/prm_2026/world/arena_cilindros.sdf
# (red and blue bases + flags are statically placed there).
_DEFAULTS = {
    # blue team (default)
    "own_base_x":   "6.0",
    "own_base_y":   "0.0",
    "enemy_base_x": "-6.0",
    "enemy_base_y": "0.0",
    "enemy_flag_x": "-8.0",
    "enemy_flag_y": "0.0",
    # mission tuning
    "phase_reach_radius_m": "1.0",
    "grab_reach_m":         "0.3",
}


def _resolve_genome_path(raw: str) -> Path:
    """Accept absolute paths or relative names; fall back to common locations.

    With `colcon build --symlink-install`, `Path(__file__).resolve()` lands in
    `src/evolutionary_controller_ros/launch/`, so walking 3 levels up reaches
    the workspace root — that's where `orchestrator` writes `genomes/`.
    """
    p = Path(raw).expanduser()
    if p.is_absolute() and p.exists():
        return p
    pkg_root = Path(__file__).resolve().parents[1]
    ws_root = pkg_root.parents[1]
    candidates = [
        Path.cwd() / raw,
        Path.cwd() / "genomes" / Path(raw).name,
        pkg_root / raw,
        pkg_root / "genomes" / Path(raw).name,
        ws_root / raw,
        ws_root / "genomes" / Path(raw).name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"genome file {raw!r} not found; tried: "
        + ", ".join(str(c) for c in [p, *candidates]))


def _launch_setup(context, *args, **kwargs):
    raw = LaunchConfiguration("genome").perform(context)
    path = _resolve_genome_path(raw)
    genome_json = path.read_text().strip()

    params = {
        "genome_json": genome_json,
        "mode": "ctf",
    }
    for name in _DEFAULTS:
        params[name] = float(LaunchConfiguration(name).perform(context))

    return [
        Node(
            package="evolutionary_controller_ros",
            executable="gp_controller",
            name="gp_controller",
            output="screen",
            parameters=[params],
        ),
    ]


def generate_launch_description():
    args = [DeclareLaunchArgument("genome", default_value="genomes/best.json",
                                  description="genome JSON path (abs or relative)")]
    for name, default in _DEFAULTS.items():
        args.append(DeclareLaunchArgument(name, default_value=default))
    return LaunchDescription([*args, OpaqueFunction(function=_launch_setup)])
