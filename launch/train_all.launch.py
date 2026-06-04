"""Bring up the whole evolutionary training stack with one command.

What it does
    1. Includes prm_2026's `inicia_simulacao.launch.py` to start Gazebo.
    2. Includes prm_2026's `carrega_robo.launch.py` for the robot stack.
    3. Starts `gp_controller` in `mode="train"` (it just waits for the
       orchestrator to push genomes via SetParameters).
    4. Starts the `orchestrator` with `config/ga_params.yaml`. As soon
       as it has subscribers it begins the GA loop.

Use this for a self-contained training run. The lower-friction workflow
during heavy iteration on GA hyperparameters remains the four-terminal
setup (Gazebo + robot + gp_controller + orchestrator in separate
shells, so you can restart the orchestrator without re-spinning
Gazebo), but for the deliverable / for a fresh box, one command is
worth the extra startup time.

Usage
    ros2 launch evolutionary_controller_ros train_all.launch.py
    ros2 launch evolutionary_controller_ros train_all.launch.py \\
        params:=path/to/ga_params.yaml \\
        output_dir:=genomes/run42 \\
        world:=arena_cilindros.sdf

Arguments
    params        — ga_params.yaml path (default: package's installed copy)
    output_dir    — where the orchestrator writes best.json + per-gen
                    checkpoints + timing CSV (default `genomes/`)
    world         — Gazebo world (default `arena_cilindros.sdf`)
    sim_delay_s, robot_delay_s, gp_delay_s, orch_delay_s — staggered
                    startup so subscribers come up after publishers.

Headless Gazebo
    Training does not benefit from a GUI and the GUI eats wall-clock.
    Keep the prm_2026 `-s` patch applied during training (see README's
    `Headless Gazebo on WSL workers` subsection); revert it only for
    the demo via mission.launch.py.
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
    "params":         "",                       # empty → use package's installed copy
    "output_dir":     "genomes",
    "world":          "arena_cilindros.sdf",
    "sim_delay_s":    "0.0",
    "robot_delay_s":  "5.0",
    "gp_delay_s":     "12.0",
    "orch_delay_s":   "15.0",
}


def _default_params_path() -> str:
    """Locate the package's installed ga_params.yaml.

    Tries the workspace's install/ tree first (where colcon symlinks the
    YAML on `--symlink-install`), then falls back to the package source.
    """
    pkg_root = Path(__file__).resolve().parents[1]
    ws_root = pkg_root.parents[1]
    candidates = [
        ws_root / "install" / "evolutionary_controller_ros"
                / "share" / "evolutionary_controller_ros"
                / "config" / "ga_params.yaml",
        pkg_root / "config" / "ga_params.yaml",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "Could not find ga_params.yaml; pass --ros-args -p params:=<path> "
        "or run colcon build first.")


def _launch_setup(context, *args, **kwargs):
    params_arg = LaunchConfiguration("params").perform(context).strip()
    output_dir = LaunchConfiguration("output_dir").perform(context)
    world      = LaunchConfiguration("world").perform(context)
    sim_delay   = float(LaunchConfiguration("sim_delay_s").perform(context))
    robot_delay = float(LaunchConfiguration("robot_delay_s").perform(context))
    gp_delay    = float(LaunchConfiguration("gp_delay_s").perform(context))
    orch_delay  = float(LaunchConfiguration("orch_delay_s").perform(context))

    params_path = params_arg or _default_params_path()

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
        parameters=[{"mode": "train"}],
    )

    orchestrator_node = Node(
        package="evolutionary_controller_ros",
        executable="orchestrator",
        name="orchestrator",
        output="screen",
        parameters=[
            params_path,                       # ga_params.yaml
            {"output_dir": output_dir},        # override / set output
        ],
    )

    return [
        TimerAction(period=sim_delay,   actions=[sim_include]),
        TimerAction(period=robot_delay, actions=[robot_include]),
        TimerAction(period=gp_delay,    actions=[gp_node]),
        # Orchestrator after gp_controller so SetParameters and /reset
        # have somebody to talk to.
        TimerAction(period=orch_delay,  actions=[orchestrator_node]),
    ]


def generate_launch_description():
    args = [DeclareLaunchArgument(name, default_value=default)
            for name, default in _DEFAULTS.items()]
    return LaunchDescription([*args, OpaqueFunction(function=_launch_setup)])
