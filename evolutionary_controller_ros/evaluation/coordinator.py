"""Multi-machine GA driver: dispatches evaluations over HTTP to a worker pool.

This is the **alternative** entry point for training. The default,
single-machine flow remains `orchestrator.py`, which evaluates each
individual locally by calling `gp_controller` through ROS in-process.
This script does the same job, but instead of evaluating in-process it
POSTs to one or more `worker_server` instances running on remote PCs.

Each worker = one PC running Gazebo + the prm_2026 stack + `gp_controller`
+ `worker_server`. The coordinator does **not** run Gazebo at all and
does **not** even need rclpy on the host — pure Python plus
`fastapi`/`httpx`/`pyyaml`.

Workflow per generation
    1. The GA proposes a population of trees.
    2. All N individuals go into a shared pending deque. One thread per
       worker spins up; each thread is tied to ONE worker URL for the
       entire generation. Each thread loops: pop an individual from the
       deque, evaluate it on its own worker, store the result.
    3. If a worker fails an individual (HTTP error, timeout), the
       individual goes back to the deque. The next thread to become
       free — driving a *different* worker — picks it up. After
       `max_attempts` failures across any combination of workers, the
       individual is marked failed (sentinel vector).
    4. The generation completes when the deque is empty AND no worker
       thread is mid-evaluation.

Wire format is the EvaluateRequest/EvaluateResponse pair from
`worker_server.py`. The coordinator's `_evaluate_one_individual` packages
each scenario as a separate POST so per-scenario timing is preserved on
the client side too.

Why "thread tied to one worker" matters
    The earlier round-robin-with-retry scheme caused a 503 cascade: when
    a worker hiccupped, the retry got dispatched to the *next* worker
    URL, which was already busy with its own assigned individual. Both
    requests competed for the same worker_server lock, one got 503,
    triggered another retry, which collided with yet another busy
    worker, and so on. Pinning each thread to one URL eliminates that
    cross-thread contention entirely; failures are redistributed via
    the shared deque, not via concurrent requests to the same server.

Failure model
    A worker that returns non-2xx, times out, or refuses connection is
    counted as one *attempt* against the individual. The individual is
    re-queued and tried by the next free worker. After `max_attempts`
    failures (default 5), the individual gets a sentinel "very bad"
    case vector so lexicase rejects it — the GA continues without that
    indi but does not crash.

Static config: `--workers` (yaml with `workers: [{url: ...}, ...]`),
`--params` (the same `ga_params.yaml` consumed by the single-machine
orchestrator; only the GA fields are read, not the ROS plumbing).
"""
import argparse
import collections
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import httpx
    import yaml
except ImportError as e:
    raise ImportError(
        "coordinator requires httpx and pyyaml — install with "
        "`pip install httpx pyyaml`"
    ) from e

from ..evolution import algorithm as alg
from ..evolution import fitness as fit
from ..evolution import genome as g
from . import timing as tlog


# Sentinel case-vector handed back when a worker hard-fails on an
# individual. Each entry is "extremely bad" under lexicase's max-is-best
# convention; all four metrics in our default vector are negated/0-or-1,
# so -1e6 sinks the individual without polluting the MAD epsilon (median
# is robust to extreme outliers).
_FAILURE_SENTINEL = -1e6


# --------------------------------------------------------------------------
# Worker pool
# --------------------------------------------------------------------------

class WorkerPool:
    """Round-robin assignment of individuals to worker URLs.

    Holds a single httpx.Client per worker for connection reuse. No
    persistent state about dead workers — each request is independent;
    a transient failure does not blacklist the worker.
    """

    def __init__(self, urls: list[str], request_timeout_s: float):
        if not urls:
            raise ValueError("worker pool is empty")
        self.urls = list(urls)
        self._timeout = request_timeout_s
        self._clients = {u: httpx.Client(base_url=u,
                                         timeout=request_timeout_s)
                         for u in urls}

    def close(self):
        for c in self._clients.values():
            c.close()

    def health_check(self) -> dict[str, dict | None]:
        """One GET /health per worker; returns dict url -> response or None."""
        out = {}
        for u in self.urls:
            try:
                r = self._clients[u].get("/health", timeout=5.0)
                r.raise_for_status()
                out[u] = r.json()
            except Exception as e:
                out[u] = {"error": repr(e)}
        return out

    def evaluate(self, url: str, payload: dict) -> dict:
        """One POST /evaluate. Raises on transport / HTTP error."""
        r = self._clients[url].post("/evaluate", json=payload,
                                    timeout=self._timeout)
        r.raise_for_status()
        return r.json()


# --------------------------------------------------------------------------
# Population evaluation
# --------------------------------------------------------------------------

def _evaluate_one_individual(
    pool: WorkerPool, worker_url: str, tree: dict,
    scenarios: list, world: dict, enabled_metrics: list,
    duration_s: float, tick_hz: float,
    timing: tlog.TimingLog | None, gen: int, individual: int,
) -> tuple[list[float], list[dict]]:
    """Run all scenarios for one tree on one worker. Serial within the worker."""
    case_vector: list[float] = []
    histories: list[dict] = []
    genome_json = g.to_json(tree)
    for sc in scenarios:
        payload = {
            "genome_json": genome_json,
            "scenario": {
                "name": sc["name"],
                "robot_start": list(sc["start"]),
                "target_pose": list(sc["target"]),
                "duration_s": duration_s,
                "tick_hz": tick_hz,
            },
            "world": world,
            "enabled_metrics": enabled_metrics,
        }
        t0 = time.monotonic()
        data = pool.evaluate(worker_url, payload)
        dt = time.monotonic() - t0
        case_vector.extend(data["case_vector"])
        histories.append(data["history"])
        if timing is not None:
            timing.event(
                "scenario_total", dt,
                gen=gen, individual=individual, scenario=sc["name"],
                worker=worker_url, **data["history"],
                worker_wall_time_s=data["wall_time_s"],
            )
    return case_vector, histories


def _evaluate_population(
    pool: WorkerPool, population: list, scenarios: list, world: dict,
    enabled_metrics: list, duration_s: float, tick_hz: float,
    timing: tlog.TimingLog | None, gen: int,
    n_metrics_per_scenario: int, max_attempts: int = 5,
    gen_timeout_s: float | None = None,
) -> list[list[float]]:
    """Dispatch the population across the worker pool via a shared deque.

    Architecture
        - One thread per worker URL, pinned for the whole generation.
          Each thread loops: pop an individual from the shared pending
          deque, evaluate it on its tied worker, write the result. By
          construction, no two threads ever POST to the same worker
          simultaneously, so the worker_server lock never sees
          concurrent attempts from this coordinator.
        - On failure, the individual returns to the deque (any worker
          can pick it up next). The deque drains as fast as the slowest
          worker — fast workers naturally pick up extra work.
        - After `max_attempts` failures across any combination of
          workers, the individual is marked failed (sentinel vector).
        - Optional `gen_timeout_s` is a wall-clock cap on the whole
          generation. None = wait indefinitely.

    Why this replaces the old "round-robin + retry on next worker"
        The previous design dispatched retries to `urls[(idx + 1) %
        n_workers]`, which collided with whichever individual was
        already assigned to that worker, causing a 503 cascade. Pinning
        each thread to one URL makes failures redistribute via the
        deque, not via concurrent requests to the same server.
    """
    n_workers = len(pool.urls)
    failure_vector = [_FAILURE_SENTINEL] * (
        n_metrics_per_scenario * len(scenarios))

    pending: collections.deque = collections.deque(enumerate(population))
    results: list[list[float] | None] = [None] * len(population)
    attempts: list[int] = [0] * len(population)

    state_lock = threading.Lock()
    state_cond = threading.Condition(state_lock)
    in_flight = [0]   # mutable container so closures can write
    deadline = (time.monotonic() + gen_timeout_s) if gen_timeout_s else None

    def _worker_loop(worker_url: str):
        while True:
            if deadline is not None and time.monotonic() > deadline:
                return
            with state_cond:
                # Wait while pending is empty but some other thread is
                # still in flight (it may re-queue on failure).
                while not pending and in_flight[0] > 0:
                    remaining = (deadline - time.monotonic()) if deadline else None
                    if remaining is not None and remaining <= 0:
                        return
                    state_cond.wait(timeout=(remaining if remaining else 1.0))
                if not pending:
                    state_cond.notify_all()  # let other waiters realize we're done
                    return
                idx, tree = pending.popleft()
                in_flight[0] += 1

            try:
                cv, _ = _evaluate_one_individual(
                    pool, worker_url, tree, scenarios, world,
                    enabled_metrics, duration_s, tick_hz,
                    timing, gen, idx)
                with state_cond:
                    results[idx] = cv
                    in_flight[0] -= 1
                    state_cond.notify_all()
            except Exception as e:
                with state_cond:
                    attempts[idx] += 1
                    print(f"  [coord] gen={gen} ind={idx} worker={worker_url} "
                          f"attempt {attempts[idx]}/{max_attempts} FAILED: "
                          f"{e!r}; re-queued",
                          file=sys.stderr)
                    if attempts[idx] < max_attempts:
                        pending.append((idx, tree))
                    else:
                        print(f"  [coord] gen={gen} ind={idx} hit max_attempts; "
                              "sentinel applied", file=sys.stderr)
                        results[idx] = list(failure_vector)
                    in_flight[0] -= 1
                    state_cond.notify_all()

    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        futures = [exe.submit(_worker_loop, url) for url in pool.urls]
        for f in as_completed(futures):
            f.result()   # propagate any thread-level crash

    # If gen_timeout_s fired with individuals still unfinished, fill in.
    for i in range(len(population)):
        if results[i] is None:
            print(f"  [coord] gen={gen} ind={i} did not complete by deadline; "
                  "sentinel applied", file=sys.stderr)
            results[i] = list(failure_vector)
    return results  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Top-level driver
# --------------------------------------------------------------------------

def _load_params(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text())
    return raw["orchestrator"]["ros__parameters"]


def _load_workers(path: Path) -> list[str]:
    raw = yaml.safe_load(path.read_text())
    urls = [w["url"] for w in raw["workers"]]
    if not urls:
        raise ValueError(f"no workers listed in {path}")
    return urls


def main():
    parser = argparse.ArgumentParser(
        description="HTTP-coordinated GA: dispatches evaluations to "
                    "remote worker_server instances.")
    parser.add_argument("--workers", required=True,
                        help="path to workers.yaml (list of {url: http://host:port})")
    parser.add_argument("--params", required=True,
                        help="path to ga_params.yaml (same file as the single-host orchestrator uses)")
    parser.add_argument("--output-dir", default="genomes",
                        help="where to write best.json + per-gen checkpoints + timing CSV")
    parser.add_argument("--timeout-s", type=float, default=600.0,
                        help="HTTP timeout per /evaluate request (default 10 min)")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="how many times to attempt one individual across "
                             "the worker pool before marking it failed "
                             "(default 5)")
    parser.add_argument("--gen-timeout-s", type=float, default=None,
                        help="wall-clock cap for one generation; any unfinished "
                             "individual gets the sentinel vector. None = wait "
                             "indefinitely.")
    args = parser.parse_args()

    params = _load_params(Path(args.params))
    worker_urls = _load_workers(Path(args.workers))

    pop_size = int(params["pop_size"])
    n_generations = int(params["n_generations"])
    init_max_depth = int(params["init_max_depth"])
    init_op_prob = float(params["init_op_prob"])
    init_erc_prob = float(params["init_erc_prob"])
    crossover_rate = float(params["crossover_rate"])
    elite_k = int(params["elite_k"])
    seed = int(params.get("seed", 42))

    duration_s = float(params["scenario_duration_s"])
    tick_hz = float(params["tick_hz"])
    enabled_metrics = list(params["enabled_metrics"])
    scenarios = json.loads(params["scenarios_json"])

    world = {
        "world_name": params.get("world_name", "capture_the_flag_world"),
        "collision_radius_m": float(params.get("collision_radius_m", 0.25)),
        "collision_debounce_s": float(params.get("collision_debounce_s", 0.5)),
        "reach_radius_m": float(params.get("reach_radius_m", 0.5)),
        "robot_spawn_z_m": float(params.get("robot_spawn_z_m", 0.3)),
        # static_models_to_reset is host-local on the worker (it knows its
        # own world's drift list), so the coordinator does not push one.
        "static_models_to_reset": [],
    }

    seeds = [g.from_json(s) for s in params.get("seeds_json", []) if s]

    rng = random.Random(seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = tlog.TimingLog()
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    timing_path = out_dir / f"timing_http_{run_stamp}.csv"

    pool = WorkerPool(worker_urls, request_timeout_s=args.timeout_s)
    print(f"[coord] {len(worker_urls)} workers, pop={pop_size} gens={n_generations}")
    print(f"[coord] timing log: {timing_path}")

    print("[coord] health check...")
    for url, h in pool.health_check().items():
        print(f"  {url}  →  {h}")

    counter = {"gen": 0}

    def evaluator_batch(population: list) -> list:
        return _evaluate_population(
            pool, population, scenarios, world, enabled_metrics,
            duration_s, tick_hz, timing, counter["gen"],
            n_metrics_per_scenario=len(enabled_metrics),
            max_attempts=args.max_attempts,
            gen_timeout_s=args.gen_timeout_s,
        )

    def on_gen(gen, pop, case_matrix, best_idx):
        counter["gen"] = gen + 1
        means = [sum(r) / len(r) for r in case_matrix]
        print(f"[coord] gen {gen} done: best_mean={means[best_idx]:.3f} "
              f"avg={sum(means) / len(means):.3f} "
              f"best_size={g.size(pop[best_idx])}")
        (out_dir / f"gen_{gen:03d}_best.json").write_text(g.to_json(pop[best_idx]))
        (out_dir / "best.json").write_text(g.to_json(pop[best_idx]))
        timing.write_csv(timing_path)

    try:
        best = alg.run_ga(
            rng,
            evaluator_batch=evaluator_batch,
            pop_size=pop_size,
            n_generations=n_generations,
            init_max_depth=init_max_depth,
            init_op_prob=init_op_prob,
            init_erc_prob=init_erc_prob,
            crossover_rate=crossover_rate,
            elite_k=elite_k,
            seeds=seeds or None,
            on_generation=on_gen,
            timing_log=timing,
        )
    finally:
        pool.close()

    (out_dir / "best.json").write_text(g.to_json(best))
    timing.write_csv(timing_path)
    print(f"[coord] champion saved to {out_dir / 'best.json'}")
    print(f"[coord] timing log: {len(timing.events)} events → {timing_path}")


if __name__ == "__main__":
    main()
