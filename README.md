# evolutionary_controller_ros

Evolutionary controller for the diff-drive robot of the [`prm_2026`](https://github.com/matheusbg8/prm_2026) package, course SSC0712 (Mobile Robot Programming — USP São Carlos, 2026).

## What this package does

The `prm_2026` package (from the professor) provides the **platform**: Gazebo world, robot with sensors, `ros2_control` controllers, ROS↔Gazebo bridges. It defines the **Capture The Flag** scenario (world `arena_cilindros.sdf`, internally `capture_the_flag_world`) — two bases, two flags, a deploy zone, walls and cylinders — but does not implement the game logic. It only ships stubs in `controle_robo.py` and `robo_mapper.py`.

This package is the **brain**: it implements evolutionary algorithms that learn policies for the robot to solve the CTF task (leave the base, find the opponent's flag, grab it with the gripper, bring it to the deploy zone).

**Separation principle:** `prm_2026` stays untouched. We consume the topics it publishes (`/scan`, `/imu`, `/odom_gt`, `/robot_cam/*`) and publish to the topics it consumes (`/cmd_vel`, `/gripper_controller/commands`). Nothing more.

## Dependencies

- ROS 2 Humble
- Gazebo Fortress (`ign gazebo` 6.x)
- `prm_2026` package cloned into `src/` of the same workspace
- Python 3.10+, numpy, scipy, opencv (already shipped with ROS Humble)

## Build

```bash
cd ~/USP/Robos_moveis/ros2_ws
colcon build --symlink-install --packages-select evolutionary_controller_ros
source install/local_setup.bash
```

With `--symlink-install`, `install/` becomes a pointer to the source in `src/` — edits do not require rebuilding, only a fresh `source` (except when `setup.py` or `package.xml` change).

## Run

Three terminals. On WSL you need `LIBGL_ALWAYS_SOFTWARE=1` on the ones that launch Gazebo (the `rsim` and `rrobo` aliases in `~/.bashrc` already do this).

```bash
# 1) World + bridge (professor's package)
rsim     # = LIBGL_ALWAYS_SOFTWARE=1 ros2 launch prm_2026 inicia_simulacao.launch.py

# 2) Robot + controllers + RViz (professor's package)
rrobo    # = LIBGL_ALWAYS_SOFTWARE=1 ros2 launch prm_2026 carrega_robo.launch.py

# 3) Evolutionary controller (this package) — instead of the professor's controle_robo
ros2 launch evolutionary_controller_ros run_controller.launch.py
```

Training (evolutionary loop):

```bash
ros2 launch evolutionary_controller_ros train.launch.py
```

Demo of the best trained genome:

```bash
ros2 launch evolutionary_controller_ros demo_best.launch.py genome:=genomes/best.npy
```

## Full structure, file by file

### Package root

| File | Purpose |
|---|---|
| `package.xml` | ROS2 manifest — name, version, dependencies, build type (`ament_python`). `colcon` reads this file. |
| `setup.py` | Python setup + registration of **ROS2 executables** (`entry_points` section). This is where you declare "this script becomes a `ros2 run` command". |
| `setup.cfg` | setuptools config — tells it where to install the scripts. |
| `resource/evolutionary_controller_ros` | Empty file. Marker that `ament_index` uses to discover the package. Never edit. |
| `LICENSE` | MIT. |
| `README.md` | This file. |
| `.gitignore` | What does not go into git (see "What is in git" section). |

### `launch/` — launch files

Use with `ros2 launch evolutionary_controller_ros <file>`.

| File | Purpose |
|---|---|
| `run_controller.launch.py` | Brings up only the `gp_controller` node. Use when the world and the robot are already running in the other terminals. |
| `train.launch.py` | Brings up the `orchestrator` with parameters from `config/ga_params.yaml`. This is the training command. |
| `demo_best.launch.py` | Brings up the controller loading a specific genome (`genome:=path/to/file.npy`). For demoing the best individual after training. |

### `config/` — parameters

| File | Purpose |
|---|---|
| `ga_params.yaml` | GA parameters read by the `orchestrator` (population size, generations, mutation/crossover rates, elite, seed). |
| `neat_config.ini` | Placeholder. Only fill in if we decide to use the `neat-python` library. |

### `evolutionary_controller_ros/` — Python code

The folder name equals the package name (`ament_python` convention).

```
evolutionary_controller_ros/
├── __init__.py             (empty — Python marker)
├── controllers/            ROS2 nodes (the "body" that drives the robot)
├── evolution/              GA core (pure Python, no ROS)
├── evaluation/             glue between ROS and the GA
└── utils/                  shared helpers
```

#### `controllers/` — the policies that drive the robot

Each file here is an **executable ROS2 node**. All of them subscribe to `/scan`, `/odom_gt`, `/robot_cam/colored_map` and publish to `/cmd_vel`. The difference between them is **how** they decide the `cmd_vel`.

| File | Purpose |
|---|---|
| `__init__.py` | Empty. |
| `gp_controller.py` | Genetic-programming tree controller. Evaluates a typed decision tree at each tick; the selected leaf `(action, duration_ms)` runs for that duration. Genome is a JSON dict. |

#### `evolution/` — the pure genetic algorithm

**This folder imports nothing from ROS.** It is plain Python, testable with `pytest` without Gazebo. That separation is what lets you iterate fast on the evolutionary part — you test a crossover in seconds, no need to bring up simulation.

| File | Purpose |
|---|---|
| `__init__.py` | Empty. |
| `genome.py` | `Genome` class — representation (float vector? NEAT graph? tree?) and serialization (`to_bytes`, `from_bytes`). |
| `population.py` | `Population` class — selection (tournament/ranking/roulette), crossover, mutation. |
| `algorithm.py` | `run_ga(population, evaluator, n_generations)` — main GA loop: evaluate → select → crossover → mutate → repeat. |
| `fitness.py` | `ctf_fitness(...)` — takes episode data and returns a score. Defines what "good" means (got flag? time? distance? collisions?). |

#### `evaluation/` — where ROS and the GA talk to each other

| File | Purpose |
|---|---|
| `__init__.py` | Empty. |
| `episode.py` | `run_episode(controller, genome, max_time)` — loads the genome into the controller, lets it run for N seconds of sim_time, collects metrics. |
| `world_reset.py` | `reset_robot(pose)` — calls the Ignition service to teleport the robot back to its initial pose. Without fast reset, training is infeasible. |
| `orchestrator.py` | Executable ROS2 node (registered as `ros2 run ... orchestrator`). Loads params from YAML, instantiates `Population`, runs the generations via `run_ga`, saves the champion to `genomes/`. |

#### `utils/` — shared helpers

| File | Purpose |
|---|---|
| `__init__.py` | Empty. |
| `sensors.py` | Turns sensor messages into normalized features. `scan_to_features` bins the lidar angularly; `image_to_flag_mask` detects the flag color blob in the segmented camera image. |
| `logger.py` | `CSVLogger` — records `(generation, best, mean)` to CSV for later plotting. |

### `genomes/` — training artifacts

Here is where the `orchestrator` saves genomes during training. The `.gitkeep` keeps the folder in git even when empty. The `.gitignore` excludes everything in it **except** `best.npy` — you commit only the champion (intermediate genomes can total many MB/GB after several generations).

### `scripts/` — tools outside ROS

Standalone scripts that run with `python3 scripts/<name>.py`. They do not become `ros2 run` commands.

| File | Purpose |
|---|---|
| `plot_fitness.py` | Reads the `CSVLogger` CSV and plots the evolution curve with matplotlib. |

### `test/` — unit tests

Run with `colcon test --packages-select evolutionary_controller_ros` or directly with `pytest`.

| File | Purpose |
|---|---|
| `test_genome.py` | Placeholder. Tests for the `Genome` class will live here. |

## Data flow

```
During normal execution (controller running in the world):

  Gazebo (prm_2026)                     This package
  ─────────────────                     ────────────
  /scan              ───────→  controller ──→  /cmd_vel
  /imu               ───────→  controller     (published at 10 Hz)
  /odom_gt           ───────→  controller
  /robot_cam/        ───────→  controller
    colored_map
                                               → /gripper_controller/commands
                                                 (when grabbing the flag)

During training:

  orchestrator
     │
     │ creates
     ▼
  population ──→ genome ──→ controller (loads the weights)
                              │
                              │ runs 1 episode
                              ▼
                         (robot runs in Gazebo for N seconds)
                              │
                              │ at episode end
                              ▼
  orchestrator ←── fitness ←── collected data
     │
     │ uses the fitness to select/crossover/mutate
     ▼
  new generation ...

  Between episodes:
  orchestrator ──→ world_reset ──→ ign service (teleports robot to initial pose)
```

## What is in git

**Everything tracked**, except what `.gitignore` excludes:

| Pattern | Why exclude |
|---|---|
| `__pycache__/`, `*.py[cod]`, `*.egg-info/` | Python bytecode — regenerable. |
| `.pytest_cache/` | pytest cache — regenerable. |
| `build/`, `install/`, `log/` | colcon artifacts (in case you run `colcon build` from the package folder by mistake). Never commit — they are regenerated. |
| `.vscode/`, `.idea/`, `*.swp` | Editor files — personal, do not belong in the repo. |
| `logs/` | Training CSV logs — can grow a lot. |
| `genomes/*` except `.gitkeep` and `best.npy` | Intermediate genomes. Commit only the champion. |

**Remote:** `git@github.com:LVinaud/evolutionary_controller_ros.git` — branch `main`.

**First commit:** `0ef606f` — initial scaffold.

To see what is tracked right now: `git ls-files` inside the package folder.

## Package conventions

- Code in English (variable names, comments, docstrings).
- Each ROS2 executable is a single `.py` file with one class + a `main()` function.
- `evolution/` is pure Python, no `import rclpy` — testable without a simulator.
- `controllers/` and `evaluation/` speak ROS.
- Genomes are serializable to `.npy` (numpy) or raw bytes via `to_bytes`/`from_bytes`.
- Stubs are marked with `raise NotImplementedError` (evolutionary part) or `# TODO:` (controllers that need to run without crashing).

## License

MIT — see [LICENSE](LICENSE).
