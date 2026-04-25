# Where the wall-clock goes — timing audit (2026-04-25)

Run instrumentation produced on `parallelism-experiments` branch by adding a
`TimingLog` to `algorithm.run_ga`, `evaluation.episode.run_episode`, and
`evaluation.orchestrator`. Each phase emits one CSV row with category +
duration + identifying metadata (gen, individual, scenario).

Raw data: `genomes/timing_20260425_114957.csv` (318 events, 21 columns).

## Run configuration

| Parameter | Value |
|---|---|
| `pop_size` | 5 |
| `n_generations` | 2 |
| `scenarios` | 5 (`base_to_center`, `base_to_far`, `corner_to_base`, `behind_obstacles`, `center_to_blue`) |
| `scenario_duration_s` | 30 |
| `tick_hz` | 10 |
| `enabled_metrics` | 4 (`dist_min_alvo_neg`, `dist_mean_alvo_neg`, `alcancou_alvo`, `colisoes_neg`) |
| Total episodes run | 50 (10 individuals × 5 scenarios) |
| Total wall time | **1648 s ≈ 27 min 28 s** |
| Hardware | `lazaroNote` — WSL2, Ubuntu 22.04, software-rendered Gazebo Fortress |

## Headline numbers

|  | Wall time |
|---|---:|
| Whole run (2 gens, pop 5) | 1648 s |
| One generation (avg) | 824 s ≈ 13.7 min |
| One individual (5 scenarios, avg) | 165 s ≈ 2.7 min |
| One scenario (avg) | 33.0 s |

## Breakdown per scenario

| Phase | Mean per scenario | Per gen (× 25 scenarios) | % of wall |
|---|---:|---:|---:|
| `episode_spin` (Gazebo simulating) | 29.43 s | 736 s | **88.7 %** |
| `setup_teleport_models` (drift-prone reset) | 3.10 s | 77 s | 9.4 % |
| `setup_teleport_robot` | 0.38 s | 9.5 s | 1.1 % |
| `setup_param_push` (genome + target_x/y) | 0.034 s | 0.86 s | 0.1 % |
| `setup_reset_call` (`/gp_controller/reset`) | 0.004 s | 0.10 s | <0.01 % |

Pure-Python GA work (population init, case-matrix assembly, lexicase
selection, crossover/mutation) ran in **<10 ms total across the entire run**.
The GA is not a bottleneck and never will be at this pop size.

## Behavioural metrics during the run

| | Value |
|---|---:|
| Scenarios where the robot reached the target (`reach_radius=0.5 m`) | 1 / 50 |
| Total collision events (debounced) | 788 |
| Average collisions per scenario | 15.8 |
| Episodes that terminated by early-stop | 1 / 50 |
| Median ticks collected per 30 s episode | 178 (out of an ideal 300 at 10 Hz) |

The 1/50 reach rate explains why every scenario consumes the full 30 s cap: 
early-stop almost never fires. The 178/300 tick ratio shows the sampling 
loop in `episode.py` is being throttled by `rclpy.spin_once`'s 100 ms 
timeout — i.e. messages from Gazebo are arriving slower than 10 Hz under 
software rendering. The simulator is the throttle, not the GA.

## Per-generation comparison

|  | gen 0 | gen 1 |
|---|---:|---:|
| Total scenario-level wall time | 834.6 s | 813.1 s |
| `episode_spin` share | 90.0 % | 88.7 % |
| `setup_teleport_models` share | 8.8 % | 10.0 % |
| `eval_individual_total` mean | 167.0 s | 162.7 s |

The two generations are essentially identical in wall-clock cost. Without
early-stops to reward better individuals, gen 1 has no opportunity to be
faster; the difference here is noise in the model-teleport time.

## Implications for parallelism

The data implies a clean upper bound for what cross-machine parallelism
can buy:

* **100 % of wall time** is spent in operations bound to a single Gazebo
  process (`episode_spin` + the four `setup_*` phases all hit the same
  simulator IPC). Nothing material runs on the orchestrator-CPU side.
* `eval_individual_total` is the unit of work that can be dispatched
  to a worker — independent across individuals within a generation,
  identical input contract (genome + scenarios), identical output
  contract (case vector).
* Per-individual cost: **~165 s**. So with `N` workers and `pop_size`
  individuals, a generation should take roughly
  `ceil(pop_size / N) × 165 s`, modulo network round-trip (negligible).

Extrapolating to the realistic training config (`pop_size=20`,
`n_generations=30`):

| Workers (N) | Per-gen wall | 30 gens (full run) |
|---:|---:|---:|
| 1 (current) | 55 min | 27.5 h |
| 2 | 28 min | 14 h |
| 5 | 11 min | 5.5 h |
| 10 | 6 min | 2.8 h |
| 20 | 3 min | 1.4 h |

(Assumes one worker = one PC running Gazebo + robot + `gp_controller`. On
WSL with software rendering, two Gazebo instances per machine saturate
CPU; on a workstation with a real GPU, 2–3 instances per machine become
viable, multiplying the effective worker count.)

## Cheap wins independent of parallelism

These are **single-machine optimizations** worth considering before adding
the network/coordination overhead of multi-machine workers:

1. **Drop `setup_teleport_models` cost.** Currently 3.1 s/scenario × 50 =
   155 s/run (9.4 %). Each call is one `ign service set_pose` per
   drift-prone model. A re-snapshot of the world only needs to fix
   genuinely-drifted models, not all of them every episode. A check 
   "did this model move by more than ε since the snapshot?" before 
   teleporting could trim most of those calls.
2. **Tighten `scenario_duration_s` once individuals start reaching.** With
   the current 1/50 reach rate, 30 s is necessary to give a fitness
   signal. Once a non-trivial share of the population is reaching, 15–20 s
   would cut wall time significantly with no loss of signal.
3. **`real_time_factor > 1`.** The orchestrator already exposes this; it
   was capped at 1.0 because WSL software-rendering is CPU-bound at 1×.
   On a workstation with a GPU, RTF 3–5× should give a near-linear
   speedup. Worth measuring on lab hardware before parallelizing.

## Recommendation for the parallel implementation

Given that 100 % of wall time is in the per-individual simulation, and
that individuals within a generation are independent, a **stateless
worker-pool** architecture matches the workload:

* Coordinator (one machine): runs the orchestrator GA loop only — does
  selection, crossover, mutation. No Gazebo.
* Worker (each lab machine): runs Gazebo + robot + `gp_controller` and
  exposes a single RPC, `evaluate(genome_json, scenarios) → case_vector`.
* Coordinator dispatches `pop_size` evaluations per generation across
  the worker pool, blocks until all return, runs selection, repeats.

Both ROS2 multi-host (DDS over LAN) and a thin gRPC/HTTP layer can
implement this. The choice is mostly about how much of the existing
ROS plumbing we want to reuse on the coordinator side. That decision
is the next discussion.
