# evolutionary_controller_ros

Evolutionary controller for the diff-drive robot of the [`prm_2026`](https://github.com/matheusbg8/prm_2026) package, course SSC0712 (Mobile Robot Programming — USP São Carlos, 2026).

## What this package does

The `prm_2026` package (from the professor) provides the **platform**: Gazebo world, robot with sensors, `ros2_control` controllers, ROS↔Gazebo bridges. It defines the **Capture The Flag** scenario (world `arena_cilindros.sdf`, internally `capture_the_flag_world`) — two bases, two flags, a deploy zone, walls and cylinders — but does not implement the game logic. It only ships stubs in `controle_robo.py` and `robo_mapper.py`.

This package is the **brain**: it implements evolutionary algorithms that learn policies for the robot to solve the CTF task (leave the base, find the opponent's flag, grab it with the gripper, bring it to the deploy zone).

**Separation principle:** `prm_2026` stays untouched. We consume the topics it publishes (`/scan`, `/odom`, `/odom_gt`, `/robot_cam/labels_map`) and publish to the topics it consumes (`/cmd_vel`, `/gripper_controller/commands`). Nothing more.

## Design decisions

This section documents the choices that shape the evolutionary part. The rationale behind each one lives here so the repo is self-describing; the same notes are mirrored in the agent's persistent memory (`memory/project_gp_approach.md`).

### Learning approach: typed Genetic Programming

- **Representation:** strongly-typed GP tree (types `Bool`, `Float`, `Action`), **not** a neural network, NEAT graph, or potential fields. Chosen for interpretability (we can read the champion) and because an IC on evolutionary algorithms makes GP the most natural extension.
- **Execution model:** *reactive and stateless* — no memory between ticks. Every decision comes from the current sensor reading.
- **Function set:** `AND/OR/NOT` (Bool→Bool), `LT/GT` (Float→Bool), `IF(Bool, Action, Action)→Action`. No arithmetic for now; we add it if a champion is bottlenecked by the lack of it.
- **Terminal set:**
  - **Bool terminals** pre-cooked with YAML-tunable thresholds: `obstaculo_frente/esq/dir`, `bandeira_inimiga_visivel`, `bandeira_esquerda/direita/centralizada`, `segurando_bandeira`, `na_base_propria`, `na_zona_deploy`, `base_inimiga_visivel`.
  - **Float terminals** raw (normalized to `[-1, 1]`): cone distances, target angles/distances, linear/angular velocity.
  - **ERC** (ephemeral random constants), `Float` values spawned at tree-creation time and mutable later.
- **Action set (7):** `FRENTE`, `RE`, `GIRA_ESQ`, `GIRA_DIR`, `PARAR`, `PEGAR`, `SOLTAR`.
- **Leaf duration (E3 style):** each leaf is `(Action, duration_ms)`. The controller becomes a mini state machine — the current action runs for its declared duration; the tree is re-evaluated only when it expires. Three leaf mutations: swap action, swap duration, swap whole leaf.

### Selection and bloat control

- **Selection:** ε-lexicase with adaptive ε via MAD (La Cava, 2016). For each case, `ε = median(|xᵢ − median(x)|)` computed across the population in the current generation. Preserves specialists — champions that excel on one hard scenario are not averaged out.
- **Bloat control:** *lexicase parsimony* — `−size(tree)` is appended as an extra case. No α hyperparameter; lexicase itself decides when tree size is the discriminating case. Scales naturally with MAD.
- **Reproduction:** Koza-style — each child is **either** crossover (`crossover_rate=0.9`) **or** mutation (`0.1`), never both. Type-respecting crossover; five mutation operators (subtree, point, leaf-action swap, leaf-duration swap, ERC perturbation).
- **Elitism:** `elite_k=1` by default; the elitism code path is parametric so we can raise K without a code change.

### Fitness: 7 metrics × N scenarios, no aggregation

Fitness is **not** a scalar. Each episode produces 7 sub-metrics:

| # | Metric | Direction |
|---|---|---|
| a | `espaco_explorado` (unique 0.5 m grid cells visited) | MAX |
| b | `tempo_bandeira_visivel` (ticks enemy flag is on camera) | MAX |
| c | `pegou_bandeira` (0/1) | MAX |
| d | `dist_min_zona_deploy_com_bandeira` (min distance to deploy while holding) | MIN |
| e | `entregou_bandeira` (0/1) | MAX |
| f | `colisoes` (debounced collision events) | MIN |
| g | `tempo_ate_entrega` (`tempo_max` if never delivered) | MIN |

MIN metrics are negated (`-x`) so every case follows the "larger is better" convention lexicase expects. We do **not** rescale to `[0,1]`: MAD adapts to each case's natural scale.

With `len(scenarios)` scenarios, the final fitness vector has `7 × len(scenarios)` entries — **not** averaged. A champion that fails scenario A but dominates B is not collapsed into a mediocre mean.

### Delivery is pressure, not a gift

`episode.py` does not teach the tree to deliver. It only **observes**: a transition `holding_flag: True → False` *inside the deploy-zone radius* counts as delivery (and triggers early-stop for the episode). The tree has to learn to emit `SOLTAR` at the right place; if we gave free delivery, the tree would never evolve that skill.

While `holding_flag=True`, `episode.py` teleports the enemy flag model to the robot's pose each tick via `ign service set_pose`. On release the flag stays where it was dropped.

### /odom vs /odom_gt

- `gp_controller` (the robot's "brain") uses only `/odom` — wheel odometry, with drift. That is what the robot *knows*.
- `episode.py` (the trainer, external) uses `/odom_gt` (ground truth) for fitness metrics that need the true pose (exploration grid, collision radius, deploy distance).

Never mix: feeding `/odom_gt` into the tree's features would give the policy information it cannot access at deploy time.

### Target poses: discovered, never hardcoded

The orchestrator queries `/world/<name>/pose/info` **once** at startup (via `ign topic -e -n 1`) and extracts the positions of `blue_base`, `red_base`, `blue_flag`, `red_flag`, `flag_deploy_zone`. Those poses are passed to `gp_controller` as ROS parameters — "the robot knows where its base is" (a priori knowledge, realistic for CTF).

Adding a new scenario means editing the SDF; no code changes. Same for switching teams — `our_team: red` flips the label lookup tables and the base assignments, and `query_model_poses` supplies the right coordinates automatically.

### Collision detection

Event-based, not integral. An event fires when `min(cone distances in all 4 directions) < collision_radius_m`, with a debounce (`collision_debounce_s`) to prevent double-counting a single contact.

`collision_radius_m = 0.25` — roughly half the chassis diagonal of `prm_2026` (base 0.42 × 0.31 m, wheels stretching width to ~0.44 m; half-diagonal ~0.30 m, with some slack). Tunable in the YAML.

### Single continuous `gp_controller` node (training and demo)

One ROS node serves both modes. The orchestrator never restarts it; instead:

1. The orchestrator updates `genome_json` + target poses + `our_team` via `SetParameters`.
2. It calls `/gp_controller/reset` (`std_srvs/Empty`) — the node zeros its current action and `holding_flag`.
3. `ign service set_pose` teleports the robot and flag to their scenario start.
4. The orchestrator spins for `scenario_duration_s`, collecting history through its own subscriptions. The controller keeps re-publishing the latest `Twist` at every tick (keep-alive against `cmd_vel_timeout=0.5s` of `diff_drive`).

`holding_flag` lives **inside** the controller (not the trainer): the controller knows when it emitted `PEGAR` and whether the robot was within `grab_reach_m` of the enemy flag. It publishes to `/gp_controller/holding_flag` with `TRANSIENT_LOCAL` QoS so `episode.py` sees the latched state.

### Budget

- `pop_size = 20`, `n_generations = 30`, `2 scenarios × 30 s` per episode, `tick_hz = 10`.
- Target RTF 3–5× via physics override in the SDF; one generation ≈ 20 individuals × 2 scenarios × ≤30 s real = ~20–40 s wall time.
- All of the above is in `config/ga_params.yaml`.

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

On WSL, `export LIBGL_ALWAYS_SOFTWARE=1` is required for any terminal that brings up Gazebo (the `rsim` / `rrobo` aliases in `~/.bashrc` set it; on a fresh machine, export it manually).

There are two end-to-end flows: **training** (evolve a genome) and **CTF mission** (run the trained genome through the real game).

### Training the genome

Three terminals — Gazebo + robot + the GA orchestrator.

```bash
# Terminal 1 — Gazebo world (professor's package)
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch prm_2026 inicia_simulacao.launch.py

# Terminal 2 — Robot + controllers + RViz (professor's package)
LIBGL_ALWAYS_SOFTWARE=1 ros2 launch prm_2026 carrega_robo.launch.py

# Terminal 3 — gp_controller in TRAIN mode + orchestrator
ros2 launch evolutionary_controller_ros run_controller.launch.py   # in one terminal
ros2 launch evolutionary_controller_ros train.launch.py            # in another
```

Per-generation checkpoints are written to `genomes/gen_NNN_best.json`; the all-time best ends up in `genomes/best.json`. To override pop / generations:

```bash
ros2 run evolutionary_controller_ros orchestrator --ros-args \
    --params-file install/evolutionary_controller_ros/share/evolutionary_controller_ros/config/ga_params.yaml \
    -p pop_size:=10 -p n_generations:=8
```

### Running the trained genome through the CTF mission

`demo_best.launch.py` boots `gp_controller` in `mode="ctf"`, loads the genome JSON from disk, and injects the base/flag coordinates of `arena_cilindros.sdf`. The state machine then drives `genome → enemy_base → enemy_flag (PEGAR) → own_base (SOLTAR)`.

```bash
# Terminals 1 and 2 — same as training (world + robot)

# Terminal 3 — run the best genome in CTF mode
ros2 launch evolutionary_controller_ros demo_best.launch.py
# default: loads genomes/best.json, blue team
```

Custom genome path or RED-team coords:

```bash
ros2 launch evolutionary_controller_ros demo_best.launch.py \
    genome:=genomes/gen_021_best.json \
    own_base_x:=-6.0 enemy_base_x:=6.0 enemy_flag_x:=8.0
```

All overridable args: `genome`, `own_base_x/y`, `enemy_base_x/y`, `enemy_flag_x/y`, `phase_reach_radius_m` (≤ this distance from a phase target → advance), `grab_reach_m` (≤ this distance from the flag → PEGAR).

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
| `run_controller.launch.py` | Brings up only the `gp_controller` node in default `mode="train"`. Used during training (the orchestrator pushes `genome_json` and `target_x/y` per episode). |
| `train.launch.py` | Brings up the `orchestrator` with parameters from `config/ga_params.yaml`. This is the training command. |
| `demo_best.launch.py` | Brings up `gp_controller` in `mode="ctf"`, loads a saved genome from disk into `genome_json`, and sets the CTF base/flag coordinates. `genome:=path/to/file.json` (default `genomes/best.json`). This is the **CTF mission runner** for an evolved genome. |

### `config/` — parameters

| File | Purpose |
|---|---|
| `ga_params.yaml` | Orchestrator parameters: GA settings (pop, generations, crossover/elite/seed), episode settings (duration, tick), collision/deploy radii, exploration grid, scenarios JSON. |
| `neat_config.ini` | Legacy placeholder; not used by the GP pipeline. Delete if it stays unused. |

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

#### `controllers/` — the policy that drives the robot

| File | Purpose |
|---|---|
| `__init__.py` | Empty. |
| `gp_controller.py` | Genetic-programming tree controller. At each tick, either keeps publishing the current `(action, duration_ms)` leaf's `cmd_vel` until it expires, or re-evaluates the tree with fresh features. Maintains `holding_flag` internally and publishes it latched. Exposes `/gp_controller/reset` for per-episode state cleanup. |

#### `evolution/` — the pure genetic algorithm

**This folder imports nothing from ROS.** It is plain Python, testable with `pytest` without Gazebo. That separation is what lets you iterate fast on the evolutionary part — you test a crossover in seconds, no need to bring up simulation.

| File | Purpose |
|---|---|
| `__init__.py` | Empty. |
| `genome.py` | Typed GP grammar: Bool/Float/Action types, `grow` initializer, `validate`, `evaluate`, JSON serialization (`to_json`/`from_json`), `size`. |
| `population.py` | `init_population`, type-respecting `crossover`, and the five mutation operators (subtree, point, leaf-action, leaf-duration, ERC perturbation). |
| `algorithm.py` | `compute_mad_epsilons`, `epsilon_lexicase_select`, and `run_ga`: the Koza-style GA loop with lexicase parsimony and parametric elitism. |
| `fitness.py` | `compute_fitness_cases(history)` — pure transformation of a history dict into the 7-entry fitness vector (in "larger is better" convention). No ROS, no episode running. |

#### `evaluation/` — where ROS and the GA talk to each other

| File | Purpose |
|---|---|
| `__init__.py` | Empty. |
| `world_reset.py` | Thin wrappers around `ign service` / `ign topic`: `set_model_pose(world, name, x, y, yaw)` for teleports, `query_model_poses(world)` to snapshot every model's pose from `/world/<name>/pose/info`. |
| `episode.py` | `run_episode(node, genome_tree, scenario, world) → history` — drives one scenario through the (long-running) `gp_controller`. Pushes genome+poses via `SetParameters`, calls `/gp_controller/reset`, teleports robot+flag, spins, collects the history dict that `fitness.py` consumes. |
| `orchestrator.py` | ROS2 node (`ros2 run ... orchestrator`). Queries target poses from the world, builds `WorldConfig`+scenarios, runs the GA with a per-tree evaluator that concatenates fitness vectors across all scenarios, saves the champion. |

#### `utils/` — shared helpers

| File | Purpose |
|---|---|
| `__init__.py` | Empty. |
| `sensors.py` | Pure function `compute_features(...)` — takes raw primitives (lidar list, labels-map uint8 array, robot pose, target poses, holding flag) and returns the normalized feature dict that `genome.evaluate()` expects. No `rclpy` import; `gp_controller` decomposes the ROS messages before calling it. |

### `genomes/` — training artifacts

The `orchestrator` writes the champion here (`best.json`). The `.gitkeep` keeps the folder in git even when empty.

### `scripts/` — tools outside ROS

Standalone scripts that run with `python3 scripts/<name>.py`. They do not become `ros2 run` commands.

| File | Purpose |
|---|---|
| `plot_fitness.py` | (Planned) Reads the training log and plots the evolution curve with matplotlib. |

### `test/` — unit tests

Run with `colcon test --packages-select evolutionary_controller_ros` or directly with `pytest`.

| File | Purpose |
|---|---|
| `test_genome.py` | Covers grammar construction, grow/validate, evaluate, serialization. |
| `test_population.py` | Crossover type-matching, each of the five mutations, input-immutability guarantees. |
| `test_algorithm.py` | MAD-epsilon math, lexicase survival, parsimony ordering, `run_ga` smoke test. |
| `test_fitness.py` | History-dict → case vector under each scenario shape (delivered, timeout, no-grab, collisions). |
| `test_sensors.py` | Lidar cone reduction, label centroid → left/center/right, normalization bounds. |

## Data flow

```
During demo (a saved genome runs in the world):

  Gazebo (prm_2026)                     This package
  ─────────────────                     ────────────
  /scan                ────────→  gp_controller ──→  /cmd_vel
  /odom                ────────→  gp_controller      (re-published at 10 Hz)
  /robot_cam/          ────────→  gp_controller      /gripper_controller/commands
    labels_map                                       (PEGAR/SOLTAR leaves)
                                                     /gp_controller/holding_flag
                                                     (latched — for observers)

During training:

  orchestrator (once)
     │
     │ ign topic /world/.../pose/info  → target poses
     │
     │ per generation:
     │   population ─┐
     │               │
     │               ▼  per individual, per scenario:
     │      SetParameters(genome_json, our_team, poses) → gp_controller
     │      /gp_controller/reset                        → gp_controller
     │      ign set_pose                                → robot + flag
     │      spin 30s collecting /odom_gt, /scan,        ← episode.py
     │         /robot_cam/labels_map, /holding_flag
     │      fitness.compute_fitness_cases(history)      → 7 cases
     │
     │   concatenate over N scenarios                   → 7×N case vector
     │   ε-lexicase select → crossover/mutate → next pop
     │
     └── save champion to genomes/best.json
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
| `genomes/*` except `.gitkeep` and `best.json` | Intermediate genomes. Commit only the champion. |

**Remote:** `git@github.com:LVinaud/evolutionary_controller_ros.git` — branch `main`.

## Package conventions

- Code in English (variable names, comments, docstrings); Portuguese is fine in design notes / memory / this README's narrative sections.
- Each ROS2 executable is a single `.py` file with one class + a `main()` function.
- `evolution/` and `utils/sensors.py` are pure Python, no `import rclpy` — testable without a simulator.
- `controllers/` and `evaluation/` speak ROS.
- Genomes are serialized as JSON (a tree dict): human-readable, diffable, and big enough that size is a non-issue.

## License

MIT — see [LICENSE](LICENSE).
