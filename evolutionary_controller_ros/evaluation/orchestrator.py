"""ROS2 executable: runs the evolutionary loop.

The orchestrator is a ROS2 node but does *not* drive the robot — it only:
    1. Discovers drift-prone model poses from /world/<name>/pose/info
       (so they can be reset between episodes without hardcoding).
    2. Iterates the GA (`evolution/algorithm.run_ga`), with each fitness
       evaluation running N scenario episodes via `evaluation/episode.py`.
    3. Logs per-generation metrics and saves the champion genome to disk.

The robot is driven by a separately-launched `gp_controller` node; the
orchestrator talks to it only via ROS params (`genome_json`, `target_x`,
`target_y`) and the `/gp_controller/reset` service.

Training drives pure go-to-goal: each scenario declares a `target` pose,
and the episode measures how well the genome navigates the robot there.
The CTF mission — what target to pick, when to grab/drop the flag — is
the `gp_controller`'s "ctf" mode concern, not the training loop's.
"""
import json
import random
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from . import episode as ep
from . import timing as tlog
from ..evolution import algorithm as alg
from ..evolution import fitness as fit
from ..evolution import genome as g
from . import world_reset as wr


_ROBOT_MODEL_NAME = "prm_robot"

# Non-static world models that drift under robot contact. Snapshotted on
# startup and teleported back at every reset. With arena_cilindros.sdf
# now marking paredes_arena as <static>true</static>, we only reset the
# remaining loose groups.
_DRIFTY_MODELS = (
    "obstaculos_azul",
    "obstaculos_vermelho",
    "center_zone",
    "blue_base",
    "red_base",
    "flag_deploy_zone",
    "blue_flag",
    "red_flag",
)


class Orchestrator(Node):
    def __init__(self):
        super().__init__("orchestrator")
        self._declare_params()

    def _declare_params(self):
        # --- World & GA ---
        self.declare_parameter("world_name", "capture_the_flag_world")
        self.declare_parameter("pop_size", 20)
        self.declare_parameter("n_generations", 30)
        self.declare_parameter("init_max_depth", 5)
        self.declare_parameter("init_op_prob", 0.7)
        self.declare_parameter("init_erc_prob", 0.3)
        self.declare_parameter("crossover_rate", 0.9)
        self.declare_parameter("elite_k", 1)
        self.declare_parameter("seed", 42)
        # --- Episode ---
        self.declare_parameter("scenario_duration_s", 30.0)
        self.declare_parameter("tick_hz", 10.0)
        self.declare_parameter("collision_radius_m", 0.25)
        self.declare_parameter("collision_debounce_s", 0.5)
        self.declare_parameter("reach_radius_m", 0.5)
        self.declare_parameter("robot_spawn_z_m", 0.3)
        # --- Fitness (lexicase) ---
        # Subset of fit.ALL_METRICS to emit per scenario, preserving order.
        self.declare_parameter(
            "enabled_metrics",
            list(fit.ALL_METRICS),
        )
        # --- Scenarios ---
        # JSON list of {name, start: [x,y,yaw], target: [x,y]}.
        self.declare_parameter(
            "scenarios_json",
            json.dumps([
                {"name": "base_to_center",   "start": [6.0, 0.0, 3.14159],
                 "target": [0.0, 0.0]},
                {"name": "base_to_far",      "start": [6.0, 0.0, 3.14159],
                 "target": [-6.0, 0.0]},
                {"name": "corner_to_base",   "start": [0.0, 3.0, -1.5708],
                 "target": [-6.0, 0.0]},
                {"name": "behind_obstacles", "start": [6.0, 0.0, 3.14159],
                 "target": [-3.0, 2.0]},
            ]),
        )
        self.declare_parameter("output_dir", "genomes")
        self.declare_parameter("real_time_factor", 1.0)
        # --- Seeding (optional) ---
        # List of genome JSON strings injected as the first individuals of
        # gen 0. Empty strings are filtered out, so a list like [""] means
        # "fully random init" (historical behavior). The `[""]` sentinel
        # exists because rclpy infers BYTE_ARRAY from an empty-list default.
        self.declare_parameter("seeds_json", [""])

    # ----------------------------------------------------------------------

    def run(self):
        world_name = self.get_parameter("world_name").value
        rng = random.Random(int(self.get_parameter("seed").value))

        rtf = float(self.get_parameter("real_time_factor").value)
        if rtf != 1.0:
            self.get_logger().info(
                f"setting real_time_factor={rtf} on world '{world_name}'")
            try:
                wr.set_physics(world_name, real_time_factor=rtf)
            except RuntimeError as e:
                self.get_logger().warn(
                    f"set_physics failed, continuing at default RTF: {e}")

        self.get_logger().info(
            f"querying world '{world_name}' to snapshot drift-prone model poses...")
        poses = wr.query_model_poses(world_name)

        world_cfg = self._build_world_config(poses)
        scenarios = self._build_scenarios()

        self.get_logger().info(
            f"drift reset list: "
            f"{[name for name, *_ in world_cfg.static_models_to_reset]}")
        self.get_logger().info(
            f"scenarios: {[(s.name, s.target_pose) for s in scenarios]}")

        enabled_metrics = tuple(
            self.get_parameter("enabled_metrics").value)
        self.get_logger().info(
            f"fitness cases per scenario: {enabled_metrics} "
            f"({len(enabled_metrics) * len(scenarios)} total)")

        pop_size = int(self.get_parameter("pop_size").value)
        out_dir = Path(self.get_parameter("output_dir").value)
        out_dir.mkdir(parents=True, exist_ok=True)
        counter = {"gen": 0, "ind": 0, "t_gen_start": time.monotonic()}

        timing = tlog.TimingLog()
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        timing_path = out_dir / f"timing_{run_stamp}.csv"
        self.get_logger().info(f"timing log will be written to {timing_path}")

        def evaluator(tree: dict) -> list:
            counter["ind"] += 1
            tree_json = g.to_json(tree)
            tree_size = g.size(tree)
            tree_depth = g.depth(tree)
            self.get_logger().info(
                f"  ind {counter['ind']}/{pop_size} size={tree_size} "
                f"depth={tree_depth}")
            self.get_logger().info(f"    genome: {tree_json}")
            t0 = time.monotonic()
            cases = _score_tree(self, tree, scenarios, world_cfg,
                                enabled_metrics, timing,
                                gen=counter["gen"],
                                individual=counter["ind"])
            timing.event(
                "eval_individual_total", time.monotonic() - t0,
                gen=counter["gen"], individual=counter["ind"],
                tree_size=tree_size, tree_depth=tree_depth,
                n_scenarios=len(scenarios),
            )
            timing.write_csv(timing_path)  # flush after each individual — survives crash
            return cases

        def on_gen(gen, pop, case_matrix, best_idx):
            elapsed = time.monotonic() - counter["t_gen_start"]
            counter["gen"] = gen + 1
            counter["ind"] = 0
            counter["t_gen_start"] = time.monotonic()
            means = [sum(r) / len(r) for r in case_matrix]
            self.get_logger().info(
                f"=== gen {gen} DONE in {elapsed:.1f}s: "
                f"best_mean={means[best_idx]:.3f} "
                f"avg_mean={sum(means) / len(means):.3f} "
                f"best_size={g.size(pop[best_idx])} ===")
            # Checkpoint: save champion of this generation (survives crashes).
            (out_dir / f"gen_{gen:03d}_best.json").write_text(g.to_json(pop[best_idx]))
            (out_dir / "best.json").write_text(g.to_json(pop[best_idx]))

        seeds = [g.from_json(s) for s in
                 self.get_parameter("seeds_json").value if s]
        if seeds:
            self.get_logger().info(
                f"seeding gen 0 with {len(seeds)} hand-crafted genome(s)")

        best = alg.run_ga(
            rng,
            evaluator=evaluator,
            pop_size=int(self.get_parameter("pop_size").value),
            n_generations=int(self.get_parameter("n_generations").value),
            init_max_depth=int(self.get_parameter("init_max_depth").value),
            init_op_prob=float(self.get_parameter("init_op_prob").value),
            init_erc_prob=float(self.get_parameter("init_erc_prob").value),
            crossover_rate=float(self.get_parameter("crossover_rate").value),
            elite_k=int(self.get_parameter("elite_k").value),
            seeds=seeds or None,
            on_generation=on_gen,
            timing_log=timing,
        )

        (out_dir / "best.json").write_text(g.to_json(best))
        self.get_logger().info(f"saved final champion to {out_dir / 'best.json'}")
        timing.write_csv(timing_path)
        self.get_logger().info(
            f"timing log: {len(timing.events)} events → {timing_path}")

    # ----------------------------------------------------------------------

    def _build_world_config(self, poses: dict) -> ep.WorldConfig:
        return ep.WorldConfig(
            world_name=self.get_parameter("world_name").value,
            robot_model_name=_ROBOT_MODEL_NAME,
            collision_radius_m=float(
                self.get_parameter("collision_radius_m").value),
            collision_debounce_s=float(
                self.get_parameter("collision_debounce_s").value),
            reach_radius_m=float(
                self.get_parameter("reach_radius_m").value),
            robot_spawn_z_m=float(
                self.get_parameter("robot_spawn_z_m").value),
            static_models_to_reset=tuple(
                (name, *poses[name]) for name in _DRIFTY_MODELS if name in poses
            ),
        )

    def _build_scenarios(self) -> list:
        raw = self.get_parameter("scenarios_json").value
        specs = json.loads(raw)
        dur = float(self.get_parameter("scenario_duration_s").value)
        hz = float(self.get_parameter("tick_hz").value)
        out = []
        for s in specs:
            sx, sy, syaw = s["start"]
            tx, ty = s["target"]
            out.append(ep.ScenarioConfig(
                name=s["name"],
                robot_start=(float(sx), float(sy), float(syaw)),
                target_pose=(float(tx), float(ty)),
                duration_s=dur,
                tick_hz=hz,
            ))
        return out


# ==========================================================================
# Evaluator helper — 1 tree → case vector (concatenated over scenarios)
# ==========================================================================

def _score_tree(node, tree, scenarios, world_cfg, enabled_metrics,
                timing=None, *, gen=None, individual=None) -> list:
    cases = []
    for sc in scenarios:
        t0 = time.monotonic()
        history = ep.run_episode(
            node, tree, sc, world_cfg,
            timing_log=timing,
            timing_extras={"gen": gen, "individual": individual,
                           "scenario": sc.name},
        )
        sc_cases = fit.compute_fitness_cases(
            history, enabled_metrics=enabled_metrics)
        if timing is not None:
            timing.event(
                "scenario_total", time.monotonic() - t0,
                gen=gen, individual=individual, scenario=sc.name,
                reached=int(history["reached_target"]),
                min_dist_target_m=float(history["min_dist_target_m"]),
                collisions=int(history["collision_events"]),
                elapsed_s=float(history["elapsed_s"]),
            )
        node.get_logger().info(
            f"    scenario '{sc.name}': "
            f"reached={history['reached_target']} "
            f"min_dist={history['min_dist_target_m']:.2f}m "
            f"collisions={history['collision_events']} "
            f"elapsed={history['elapsed_s']:.1f}s")
        cases.extend(sc_cases)
    return cases


# ==========================================================================
# Entry point
# ==========================================================================

def main():
    rclpy.init()
    node = Orchestrator()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
