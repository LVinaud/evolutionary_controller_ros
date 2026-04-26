"""GA driver that dispatches evaluations through a hosted Redis instance.

Sister of `coordinator.py` (HTTP) — same shape, different transport.
Use this when the lab network blocks peer-to-peer connections (eduroam
client isolation, locked-down VLANs) but outbound HTTPS to a hosted
Redis is allowed.

Coordinator pushes jobs onto `evo:jobs` (RPUSH). Workers pull from that
list with `BLPOP` and push results onto `evo:results`. The coordinator
drains `evo:results` with another `BLPOP`, matches them back to the
right population slot via `job_id`, runs the GA loop, repeats.

Per-run namespacing
    `job_id = "<run_id>:<individual>:<scenario>"`. The coordinator
    generates a fresh random `run_id` each invocation so stale results
    from a crashed previous run can be filtered out (silently ignored).
    This means a worker that picked up a stale job from a previous run
    will return a result the new coordinator does not care about — the
    cycle is wasted but correctness is preserved.

Failure handling
    A worker that runs into an exception inside `run_episode` still
    pushes a result with `error` populated (and an empty case vector).
    The coordinator treats any such result as a failed individual and
    substitutes a sentinel "very bad" case vector that lexicase will
    reject. Jobs that never come back at all (worker died mid-execution,
    Redis loss, etc.) eventually time out per `--per-job-timeout-s` and
    the population slot is filled with the same sentinel.

Static config
    `--redis-url` (or $REDIS_URL): rediss://default:<token>@<host>:<port>
    `--params`: same `ga_params.yaml` consumed by orchestrator/coordinator
"""
import argparse
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path

try:
    import redis
    import yaml
except ImportError as e:
    raise ImportError(
        "redis_coordinator requires redis-py and pyyaml — install with "
        "`pip install redis pyyaml`"
    ) from e

from ..evolution import algorithm as alg
from ..evolution import genome as g
from . import timing as tlog


_JOBS_QUEUE = "evo:jobs"
_RESULTS_QUEUE = "evo:results"

# Same sentinel as the HTTP coordinator. -1e6 is "extremely bad" under
# lexicase's max-is-better convention; the median-based MAD epsilon
# handles outliers gracefully so the rest of the population still
# differentiates normally.
_FAILURE_SENTINEL = -1e6


# --------------------------------------------------------------------------

def _build_payload(run_id: str, individual: int, scenario_idx: int,
                   tree: dict, scenario_dict: dict, world: dict,
                   enabled_metrics: list,
                   duration_s: float, tick_hz: float) -> tuple[str, dict]:
    job_id = f"{run_id}:{individual}:{scenario_idx}"
    return job_id, {
        "job_id": job_id,
        "individual": individual,
        "genome_json": g.to_json(tree),
        "scenario": {
            "name": scenario_dict["name"],
            "robot_start": list(scenario_dict["start"]),
            "target_pose": list(scenario_dict["target"]),
            "duration_s": duration_s,
            "tick_hz": tick_hz,
        },
        "world": world,
        "enabled_metrics": enabled_metrics,
    }


def _evaluate_population(
    r: "redis.Redis", run_id: str, population: list, scenarios: list,
    world: dict, enabled_metrics: list, duration_s: float, tick_hz: float,
    n_metrics_per_scenario: int, per_job_timeout_s: float,
    timing: tlog.TimingLog | None, gen: int,
) -> list[list[float]]:
    """Submit `len(population) * len(scenarios)` jobs, collect by job_id."""
    failure_vector = [_FAILURE_SENTINEL] * (
        n_metrics_per_scenario * len(scenarios))

    # Submit jobs.
    pending: dict[str, tuple[int, int]] = {}     # job_id → (individual, scenario_idx)
    pipe = r.pipeline()
    for i, tree in enumerate(population):
        for j, sc in enumerate(scenarios):
            job_id, payload = _build_payload(
                run_id, i, j, tree, sc, world, enabled_metrics,
                duration_s, tick_hz)
            pending[job_id] = (i, j)
            pipe.rpush(_JOBS_QUEUE, json.dumps(payload))
    pipe.execute()

    # Pre-allocate per-individual case vectors (filled scenario-by-scenario).
    per_indiv: list[list[float | None]] = [
        [None] * (n_metrics_per_scenario * len(scenarios))
        for _ in population
    ]

    # Drain results.
    deadline = time.monotonic() + per_job_timeout_s * len(pending)
    while pending and time.monotonic() < deadline:
        item = r.blpop(_RESULTS_QUEUE, timeout=int(per_job_timeout_s))
        if item is None:
            continue
        _key, raw = item
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        job_id = data.get("job_id")
        if job_id not in pending:
            # Stale result from another run, or a job we already accounted for.
            continue
        i, j = pending.pop(job_id)
        if data.get("error"):
            print(f"  [coord] gen={gen} ind={i} scenario={j} FAILED on worker: "
                  f"{data['error']}", file=sys.stderr)
            slot = failure_vector[j * n_metrics_per_scenario:
                                  (j + 1) * n_metrics_per_scenario]
        else:
            slot = data["case_vector"]
        per_indiv[i][j * n_metrics_per_scenario:
                     (j + 1) * n_metrics_per_scenario] = slot
        if timing is not None:
            timing.event(
                "scenario_total", float(data.get("wall_time_s", 0.0)),
                gen=gen, individual=i, scenario_idx=j, job_id=job_id,
                **data.get("history", {}),
            )

    # Anything still pending = timed out → sentinel.
    if pending:
        print(f"  [coord] {len(pending)} jobs timed out, marking individuals as failed",
              file=sys.stderr)
        for job_id, (i, j) in pending.items():
            per_indiv[i][j * n_metrics_per_scenario:
                         (j + 1) * n_metrics_per_scenario] = (
                failure_vector[j * n_metrics_per_scenario:
                               (j + 1) * n_metrics_per_scenario])

    # Convert any leftover Nones (shouldn't happen) to sentinel and return.
    out: list[list[float]] = []
    for vec in per_indiv:
        out.append([(_FAILURE_SENTINEL if v is None else float(v)) for v in vec])
    return out


# --------------------------------------------------------------------------

def _load_params(path: Path) -> dict:
    return yaml.safe_load(path.read_text())["orchestrator"]["ros__parameters"]


def main():
    parser = argparse.ArgumentParser(
        description="Redis-relayed GA coordinator. Use when the lab "
                    "network blocks peer-to-peer connections.")
    parser.add_argument("--redis-url",
                        default=os.environ.get("REDIS_URL"),
                        help="rediss://default:<token>@<host>:<port>. "
                             "Defaults to $REDIS_URL.")
    parser.add_argument("--params", required=True,
                        help="path to ga_params.yaml (same as the "
                             "single-host orchestrator uses)")
    parser.add_argument("--output-dir", default="genomes")
    parser.add_argument("--per-job-timeout-s", type=float, default=600.0,
                        help="how long to wait for any one result before giving up")
    parser.add_argument("--clear-queues", action="store_true",
                        help="DEL evo:jobs and evo:results before starting "
                             "(cleans up after a crashed run)")
    args = parser.parse_args()

    if not args.redis_url:
        print("ERROR: pass --redis-url or set REDIS_URL", file=sys.stderr)
        sys.exit(2)

    r = redis.from_url(args.redis_url, decode_responses=True)
    try:
        r.ping()
    except redis.RedisError as e:
        print(f"ERROR: cannot reach redis: {e}", file=sys.stderr)
        sys.exit(2)

    if args.clear_queues:
        r.delete(_JOBS_QUEUE, _RESULTS_QUEUE)
        print(f"[coord] cleared {_JOBS_QUEUE} and {_RESULTS_QUEUE}")

    params = _load_params(Path(args.params))
    pop_size = int(params["pop_size"])
    n_generations = int(params["n_generations"])
    duration_s = float(params["scenario_duration_s"])
    tick_hz = float(params["tick_hz"])
    enabled_metrics = list(params["enabled_metrics"])
    scenarios = json.loads(params["scenarios_json"])
    seeds = [g.from_json(s) for s in params.get("seeds_json", []) if s]

    world = {
        "world_name": params.get("world_name", "capture_the_flag_world"),
        "collision_radius_m": float(params.get("collision_radius_m", 0.25)),
        "collision_debounce_s": float(params.get("collision_debounce_s", 0.5)),
        "reach_radius_m": float(params.get("reach_radius_m", 0.5)),
        "robot_spawn_z_m": float(params.get("robot_spawn_z_m", 0.3)),
        "static_models_to_reset": [],
    }

    rng = random.Random(int(params.get("seed", 42)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = tlog.TimingLog()
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    timing_path = out_dir / f"timing_redis_{run_stamp}.csv"
    run_id = uuid.uuid4().hex[:8]
    print(f"[coord] run_id={run_id} pop={pop_size} gens={n_generations}")
    print(f"[coord] timing log: {timing_path}")

    counter = {"gen": 0}

    def evaluator_batch(population: list) -> list:
        return _evaluate_population(
            r, run_id, population, scenarios, world, enabled_metrics,
            duration_s, tick_hz,
            n_metrics_per_scenario=len(enabled_metrics),
            per_job_timeout_s=args.per_job_timeout_s,
            timing=timing, gen=counter["gen"],
        )

    def on_gen(gen, pop, case_matrix, best_idx):
        counter["gen"] = gen + 1
        means = [sum(row) / len(row) for row in case_matrix]
        print(f"[coord] gen {gen} done: best_mean={means[best_idx]:.3f} "
              f"avg={sum(means) / len(means):.3f} "
              f"best_size={g.size(pop[best_idx])}")
        (out_dir / f"gen_{gen:03d}_best.json").write_text(g.to_json(pop[best_idx]))
        (out_dir / "best.json").write_text(g.to_json(pop[best_idx]))
        timing.write_csv(timing_path)

    best = alg.run_ga(
        rng,
        evaluator_batch=evaluator_batch,
        pop_size=pop_size,
        n_generations=n_generations,
        init_max_depth=int(params["init_max_depth"]),
        init_op_prob=float(params["init_op_prob"]),
        init_erc_prob=float(params["init_erc_prob"]),
        crossover_rate=float(params["crossover_rate"]),
        elite_k=int(params["elite_k"]),
        seeds=seeds or None,
        on_generation=on_gen,
        timing_log=timing,
    )

    (out_dir / "best.json").write_text(g.to_json(best))
    timing.write_csv(timing_path)
    print(f"[coord] champion saved to {out_dir / 'best.json'}")
    print(f"[coord] timing log: {len(timing.events)} events → {timing_path}")


if __name__ == "__main__":
    main()
