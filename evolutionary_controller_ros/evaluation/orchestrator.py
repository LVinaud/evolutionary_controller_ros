"""ROS2 executable: runs the evolutionary loop.

The orchestrator is a ROS2 node but does *not* drive the robot — it only:
    1. Discovers target poses from /world/<name>/pose/info (no hardcoding).
    2. Iterates the GA (`evolution/algorithm.run_ga`), with each fitness
       evaluation running N scenario episodes via `evaluation/episode.py`.
    3. Logs per-generation metrics and saves the champion genome to disk.

The robot is driven by a separately-launched `gp_controller` node; the
orchestrator talks to it only via ROS params and the `/gp_controller/reset`
service (see `evaluation/episode.py`).
"""
import json
import os
import random
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from . import episode as ep
from ..evolution import algorithm as alg
from ..evolution import genome as g
from . import world_reset as wr


_DEFAULT_MODEL_NAMES = {
    "robot":        "prm_robot",
    "own_base":     {"blue": "blue_base",  "red": "red_base"},
    "enemy_base":   {"blue": "red_base",   "red": "blue_base"},
    "enemy_flag":   {"blue": "red_flag",   "red": "blue_flag"},
    "deploy_zone":  "flag_deploy_zone",
}

# Non-static world models that drift under robot contact (the professor's SDF
# does not mark them `<static>true</static>`). We snapshot their poses on
# startup and teleport them back at every episode reset.
_DRIFTY_MODELS = (
    "paredes_arena",
    "obstaculos_azul",
    "obstaculos_vermelho",
    "center_zone",
    "blue_base",
    "red_base",
    "flag_deploy_zone",
)


class Orchestrator(Node):
    def __init__(self):
        super().__init__("orchestrator")
        self._declare_params()

    def _declare_params(self):
        self.declare_parameter("world_name", "capture_the_flag_world")
        self.declare_parameter("our_team", "blue")
        self.declare_parameter("pop_size", 20)
        self.declare_parameter("n_generations", 30)
        self.declare_parameter("init_max_depth", 5)
        self.declare_parameter("init_op_prob", 0.7)
        self.declare_parameter("init_erc_prob", 0.3)
        self.declare_parameter("crossover_rate", 0.9)
        self.declare_parameter("elite_k", 1)
        self.declare_parameter("seed", 42)
        self.declare_parameter("scenario_duration_s", 30.0)
        self.declare_parameter("tick_hz", 10.0)
        self.declare_parameter("collision_radius_m", 0.25)
        self.declare_parameter("collision_debounce_s", 0.5)
        self.declare_parameter("deploy_zone_radius_m", 1.0)
        self.declare_parameter("exploration_cell_size_m", 0.5)
        self.declare_parameter("robot_spawn_z_m", 0.3)
        self.declare_parameter(
            "scenarios_json",
            json.dumps([
                {"name": "from_base", "x": 6.0, "y": 0.0, "yaw": 3.14159},
                {"name": "north_center", "x": 0.0, "y": 3.0, "yaw": -1.5708},
            ]),
        )
        self.declare_parameter("output_dir", "genomes")
        self.declare_parameter("real_time_factor", 1.0)

    # ----------------------------------------------------------------------

    def run(self):
        world_name = self.get_parameter("world_name").value
        team = self.get_parameter("our_team").value
        rng = random.Random(int(self.get_parameter("seed").value))

        rtf = float(self.get_parameter("real_time_factor").value)
        if rtf != 1.0:
            self.get_logger().info(f"setting real_time_factor={rtf} on world '{world_name}'")
            try:
                wr.set_physics(world_name, real_time_factor=rtf)
            except RuntimeError as e:
                self.get_logger().warn(f"set_physics failed, continuing at default RTF: {e}")

        self.get_logger().info(
            f"querying world '{world_name}' for target poses...")
        poses = wr.query_model_poses(world_name)

        world_cfg = self._build_world_config(poses, team)
        scenarios = self._build_scenarios(world_cfg.enemy_flag_pose)

        self.get_logger().info(
            f"poses: own_base={world_cfg.own_base_pose}, "
            f"enemy_base={world_cfg.enemy_base_pose}, "
            f"enemy_flag={world_cfg.enemy_flag_pose}, "
            f"deploy={world_cfg.deploy_zone_pose}")
        self.get_logger().info(f"scenarios: {[s.name for s in scenarios]}")

        pop_size = int(self.get_parameter("pop_size").value)
        out_dir = Path(self.get_parameter("output_dir").value)
        out_dir.mkdir(parents=True, exist_ok=True)
        counter = {"gen": 0, "ind": 0, "t_gen_start": time.monotonic()}

        def evaluator(tree: dict) -> list:
            counter["ind"] += 1
            tree_json = g.to_json(tree)
            self.get_logger().info(
                f"  ind {counter['ind']}/{pop_size} size={g.size(tree)} "
                f"depth={g.depth(tree)}")
            self.get_logger().info(f"    genome: {tree_json}")
            return _score_tree(self, tree, scenarios, world_cfg,
                               self.get_parameter("exploration_cell_size_m").value)

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
            on_generation=on_gen,
        )

        (out_dir / "best.json").write_text(g.to_json(best))
        self.get_logger().info(f"saved final champion to {out_dir / 'best.json'}")

    # ----------------------------------------------------------------------

    def _build_world_config(self, poses: dict, team: str) -> ep.WorldConfig:
        def pick(key):
            target = _DEFAULT_MODEL_NAMES[key]
            if isinstance(target, dict):
                target = target[team]
            if target not in poses:
                raise RuntimeError(
                    f"model {target!r} not found in /pose/info snapshot")
            x, y, _yaw = poses[target]
            return target, (x, y)

        _, own_base  = pick("own_base")
        _, enemy_base = pick("enemy_base")
        ef_name, enemy_flag = pick("enemy_flag")
        _, deploy_zone = pick("deploy_zone")

        return ep.WorldConfig(
            world_name=self.get_parameter("world_name").value,
            robot_model_name=_DEFAULT_MODEL_NAMES["robot"],
            enemy_flag_model_name=ef_name,
            our_team=team,
            enemy_flag_pose=enemy_flag,
            own_base_pose=own_base,
            enemy_base_pose=enemy_base,
            deploy_zone_pose=deploy_zone,
            collision_radius_m=float(
                self.get_parameter("collision_radius_m").value),
            collision_debounce_s=float(
                self.get_parameter("collision_debounce_s").value),
            deploy_zone_radius_m=float(
                self.get_parameter("deploy_zone_radius_m").value),
            exploration_cell_size_m=float(
                self.get_parameter("exploration_cell_size_m").value),
            robot_spawn_z_m=float(
                self.get_parameter("robot_spawn_z_m").value),
            static_models_to_reset=tuple(
                (name, *poses[name]) for name in _DRIFTY_MODELS if name in poses
            ),
        )

    def _build_scenarios(self, enemy_flag_initial: tuple) -> list:
        raw = self.get_parameter("scenarios_json").value
        specs = json.loads(raw)
        dur = float(self.get_parameter("scenario_duration_s").value)
        hz = float(self.get_parameter("tick_hz").value)
        return [
            ep.ScenarioConfig(
                name=s["name"],
                robot_start=(float(s["x"]), float(s["y"]), float(s["yaw"])),
                enemy_flag_initial=enemy_flag_initial,
                duration_s=dur,
                tick_hz=hz,
            )
            for s in specs
        ]


# ==========================================================================
# Evaluator helper — 1 tree → case vector (concatenated over scenarios)
# ==========================================================================

def _score_tree(node, tree, scenarios, world_cfg, cell_size_m) -> list:
    from ..evolution import fitness as fit
    cases = []
    for sc in scenarios:
        history = ep.run_episode(node, tree, sc, world_cfg)
        sc_cases = fit.compute_fitness_cases(
            history, exploration_cell_size_m=cell_size_m)
        node.get_logger().info(
            f"    scenario '{sc.name}': "
            f"held={history['holding_any_tick']} "
            f"delivered={history['delivered']} "
            f"collisions={history['collision_events']} "
            f"elapsed={history['elapsed_s']:.1f}s "
            f"cells={len(set((int(x/cell_size_m), int(y/cell_size_m)) for x,y in history['positions_xy']))}")
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
