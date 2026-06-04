"""Bring up the whole Trabalho 1 stack with one command.

What it does
    1. Includes prm_2026's `inicia_simulacao.launch.py` to start Gazebo
       with the CTF arena world.
    2. Includes prm_2026's `carrega_robo.launch.py` (with a short delay
       so Gazebo has time to settle) to spawn the diff-drive robot,
       bridges, and ros2_control controllers.
    3. Starts `gp_controller` in `mode="mission"` with a saved genome
       loaded (default `genomes/best.json`), wired to the four-state
       mission machine (EXPLORANDO → BANDEIRA_DETECTADA →
       NAVEGANDO_PARA_BANDEIRA → POSICIONANDO_PARA_COLETA).

Usage
    ros2 launch evolutionary_controller_ros mission.launch.py
    ros2 launch evolutionary_controller_ros mission.launch.py \\
        genome:=genomes/gen_021_best.json team:=red

Arguments
    genome     — path to the genome JSON (default `genomes/best.json`,
                 resolved against cwd, the package directory, and the
                 workspace root).
    team       — `blue` (default) or `red`. Controls which flag label
                 the detector chases: blue chases red_flag (label 20),
                 red chases blue_flag (label 25).
    world      — Gazebo world file name from prm_2026 (default
                 `arena_cilindros.sdf`, which is the CTF map).
    sim_delay_s, robot_delay_s — staggered start times so the gp_controller
                 doesn't try to subscribe before /scan and /odom exist.

Headless Gazebo note
    On worker machines we patch prm_2026 to launch Gazebo server-only
    (`ign gazebo -s`). For demoing the mission run, you almost certainly
    want the GUI back so you can see the robot. Revert the patch with:
        sed -i "s|'gazebo', '-s', '-r'|'gazebo', '-r'|" \\
            ~/ros2_ws/src/prm_2026/launch/inicia_simulacao.launch.py
        cd ~/ros2_ws && colcon build --packages-select prm_2026
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


_DEFAULTS = {
    "genome":         "genomes/best.json",
    "team":           "blue",                 # → enemy_flag_label=20 (red flag)
    "world":          "arena_cilindros.sdf",
    "sim_delay_s":    "0.0",
    "robot_delay_s":  "5.0",
    "gp_delay_s":     "12.0",
}

# Mapping team name → which flag label to chase. blue chases red (20);
# red chases blue (25). Mirrors the convention used in demo_best.launch.
_TEAM_TO_ENEMY_LABEL = {"blue": 20, "red": 25}


def _resolve_genome_path(raw: str) -> Path:
    """Mirror of demo_best.launch.py's resolver — kept duplicated to
    avoid making launch files depend on package-level Python modules."""
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
    genome_raw = LaunchConfiguration("genome").perform(context)
    team       = LaunchConfiguration("team").perform(context).lower()
    world      = LaunchConfiguration("world").perform(context)
    sim_delay   = float(LaunchConfiguration("sim_delay_s").perform(context))
    robot_delay = float(LaunchConfiguration("robot_delay_s").perform(context))
    gp_delay    = float(LaunchConfiguration("gp_delay_s").perform(context))

    if team not in _TEAM_TO_ENEMY_LABEL:
        raise ValueError(
            f"team must be one of {list(_TEAM_TO_ENEMY_LABEL)}, got {team!r}")
    enemy_label = _TEAM_TO_ENEMY_LABEL[team]

    genome_path = _resolve_genome_path(genome_raw)
    genome_json = genome_path.read_text().strip()

    prm_share = FindPackageShare("prm_2026")

    sim_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([prm_share, "launch", "inicia_simulacao.launch.py"])
        ),
        launch_arguments={"world": world}.items(),
    )

    robot_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([prm_share, "launch", "carrega_robo.launch.py"])
        ),
    )

    gp_node = Node(
        package="evolutionary_controller_ros",
        executable="gp_controller",
        name="gp_controller",
        output="screen",
        parameters=[{
            "mode": "mission",
            "genome_json": genome_json,
            "enemy_flag_label": enemy_label,
            # everything else uses the controller's declared defaults
        }],
    )

    return [
        # Gazebo + bridges first.
        TimerAction(period=sim_delay, actions=[sim_include]),
        # Robot + ros2_control + RViz — needs Gazebo alive.
        TimerAction(period=robot_delay, actions=[robot_include]),
        # gp_controller subscribes /scan, /odom, /robot_cam/labels_map — all
        # of which only exist after the robot launch has loaded. Conservative
        # delay so we don't see "idle — waiting for: scan, odom" for the
        # first dozen ticks.
        TimerAction(period=gp_delay, actions=[gp_node]),
    ]


def generate_launch_description():
    args = []
    for name, default in _DEFAULTS.items():
        args.append(DeclareLaunchArgument(name, default_value=default))
    return LaunchDescription([*args, OpaqueFunction(function=_launch_setup)])
