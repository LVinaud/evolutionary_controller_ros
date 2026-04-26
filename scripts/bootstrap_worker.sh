#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu 22.04 box as an evaluation worker for the
# evolutionary_controller_ros HTTP coordinator.
#
# What this script does:
#   1. Installs ROS2 Humble + Gazebo Fortress bridge if missing.
#   2. Installs the Python deps for worker_server (fastapi/uvicorn/pydantic).
#   3. Creates ~/ros2_ws and clones the two packages we need
#      (prm_2026 from the professor; this package from the student fork).
#   4. Builds with colcon.
#   5. Prints the four commands to bring the worker up in tmux/screen.
#
# Idempotent — running it twice is safe; existing files/repos are skipped.
#
# Usage:
#   bash bootstrap_worker.sh                  # uses defaults below
#   bash bootstrap_worker.sh --port 8001      # different worker port
#
# After the script completes, see the printed instructions for the four
# processes you must launch. The script does NOT start them — those need
# to live in long-running terminals (tmux recommended).

set -euo pipefail

PORT=8000
WS_DIR="${HOME}/ros2_ws"
# Both repos are public — using https:// so a fresh lab PC without an
# SSH key on GitHub still clones cleanly. Override at the command line
# (--evo-repo, --prm-repo) if a private fork is needed.
PRM_REPO="https://github.com/matheusbg8/prm_2026.git"
EVO_REPO="https://github.com/LVinaud/evolutionary_controller_ros.git"
EVO_BRANCH="parallelism-experiments"   # branch where worker_server lives

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --ws-dir) WS_DIR="$2"; shift 2 ;;
        --evo-branch) EVO_BRANCH="$2"; shift 2 ;;
        --evo-repo) EVO_REPO="$2"; shift 2 ;;
        --prm-repo) PRM_REPO="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

say() { printf "\n=== %s ===\n" "$*"; }

# ---------------------------------------------------------------------- ROS2
if ! command -v ros2 >/dev/null 2>&1; then
    say "installing ROS2 Humble"
    sudo apt-get update
    sudo apt-get install -y curl gnupg lsb-release
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt-get update
    # ros-humble-xacro is listed explicitly because some apt mirrors do
    # not pull it in transitively from ros-humble-desktop. prm_2026 ships
    # the robot as `.urdf.xacro`, so the launch fails with
    # "file not found: 'xacro'" if it's missing.
    sudo apt-get install -y ros-humble-desktop ros-humble-xacro \
        ros-humble-ros-gz-bridge ros-humble-ros-gz-sim ros-humble-ros-gz-image \
        python3-colcon-common-extensions python3-rosdep python3-pip
    if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
        sudo rosdep init || true
    fi
    rosdep update
fi

# ---------------------------------------------------------------------- pip deps
say "installing python deps for worker_server"
pip install --user --quiet fastapi uvicorn pydantic httpx pyyaml

# ---------------------------------------------------------------------- workspace
mkdir -p "${WS_DIR}/src"
cd "${WS_DIR}/src"

if [[ ! -d prm_2026 ]]; then
    say "cloning prm_2026"
    git clone "${PRM_REPO}"
fi

if [[ ! -d evolutionary_controller_ros ]]; then
    say "cloning evolutionary_controller_ros (branch ${EVO_BRANCH})"
    git clone -b "${EVO_BRANCH}" "${EVO_REPO}"
fi

# Patch the prm_2026 simulation launch to run Gazebo server-only.
# Background: prm_2026's inicia_simulacao.launch.py invokes
# `ign gazebo -r -v 3 <world>` and its own comment says "modo headless
# (sem GUI)" — but the actual command is missing the `-s` flag, so the
# GUI does start. On WSL2 (Windows 10 especially, which has no WSLg)
# the GUI process crashes inside Qt's XCB clipboard handler, taking
# the simulation server down with it. A worker has no use for the GUI
# anyway, so the cleanest fix is to honour the original intent: add
# `-s`. Idempotent: skip if already patched.
PRM_LAUNCH="${WS_DIR}/src/prm_2026/launch/inicia_simulacao.launch.py"
if [[ -f "${PRM_LAUNCH}" ]] && ! grep -q "'gazebo', '-s'" "${PRM_LAUNCH}"; then
    say "patching prm_2026 launch to run Gazebo server-only (no GUI)"
    sed -i "s|'gazebo', '-r'|'gazebo', '-s', '-r'|" "${PRM_LAUNCH}"
fi

# ---------------------------------------------------------------------- build
say "colcon build"
cd "${WS_DIR}"
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select prm_2026 evolutionary_controller_ros

# ---------------------------------------------------------------------- env hint
if ! grep -q "LIBGL_ALWAYS_SOFTWARE" "${HOME}/.bashrc"; then
    say "appending LIBGL_ALWAYS_SOFTWARE=1 to ~/.bashrc (WSL needs it)"
    echo 'export LIBGL_ALWAYS_SOFTWARE=1' >> "${HOME}/.bashrc"
fi

# ---------------------------------------------------------------------- print run instructions
cat <<EOF

=== bootstrap done ===

Worker port: ${PORT}
Workspace:   ${WS_DIR}

Bring up the four required processes in separate terminals (tmux strongly
recommended). They must all source the workspace first:

    source /opt/ros/humble/setup.bash
    source ${WS_DIR}/install/local_setup.bash
    export LIBGL_ALWAYS_SOFTWARE=1   # only on WSL

Terminal 1 — Gazebo + bridge:
    ros2 launch prm_2026 inicia_simulacao.launch.py

Terminal 2 — Robot + ros2_control + RViz:
    ros2 launch prm_2026 carrega_robo.launch.py

Terminal 3 — gp_controller (default mode "train"):
    ros2 run evolutionary_controller_ros gp_controller

Terminal 4 — worker_server (this is what the coordinator talks to):
    ros2 run evolutionary_controller_ros worker_server --port ${PORT}

To verify the worker is alive from any other machine on the LAN:

    curl http://\$(hostname -I | awk '{print \$1}'):${PORT}/health
    # → {"status":"idle","uptime_s":...}

Then on the COORDINATOR machine, add this worker's URL to workers.yaml:

    workers:
      - url: http://<this-machine-ip>:${PORT}

EOF
