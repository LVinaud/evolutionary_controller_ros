"""Tests for the HTTP coordinator — pure Python, no Gazebo, no ROS.

Strategy: stand up a fake "worker" via httpx.MockTransport that returns
synthetic case vectors. The coordinator should dispatch the population
across the fake workers, retry on failure, and respect lexicase output
ordering (results aligned by index even though dispatch is concurrent).

Auto-skips if httpx / pyyaml are not installed.
"""
import json
import random

import pytest

# Optional deps — both must be present for any of these tests to run.
# importlib instead of importorskip because the latter raises a
# pytest.skip *exception* at module load time, which older pytest
# versions (6.2 ships with ROS2 Humble) propagate up to the collector
# and abort discovery of other test files in the same run.
import importlib.util
_HAVE_DEPS = (importlib.util.find_spec("httpx") is not None
              and importlib.util.find_spec("yaml") is not None)
pytestmark = pytest.mark.skipif(
    not _HAVE_DEPS,
    reason="parallel-mode tests need httpx + pyyaml — install with "
           "`pip install httpx pyyaml`")

if _HAVE_DEPS:
    import httpx
    from evolutionary_controller_ros.evaluation import coordinator as coord
    from evolutionary_controller_ros.evolution import genome as g


# --------------------------------------------------------------------------
# Fixtures: minimal scenario / world / metrics for a coordinator request
# --------------------------------------------------------------------------

_SCENARIOS = [
    {"name": "to_origin",  "start": [3.0, 0.0, 3.14], "target": [0.0, 0.0]},
    {"name": "to_corner",  "start": [0.0, 0.0, 0.0],  "target": [3.0, 3.0]},
]
_WORLD = {
    "world_name": "test_world",
    "collision_radius_m": 0.25,
    "collision_debounce_s": 0.5,
    "reach_radius_m": 0.5,
    "robot_spawn_z_m": 0.3,
    "static_models_to_reset": [],
}
_METRICS = ["dist_min_alvo_neg", "alcancou_alvo"]


def _trivial_tree():
    return {"leaf": "FRENTE", "dur_ms": 200}


def _build_pool_with_handler(urls, handler):
    """WorkerPool with each url's httpx.Client wired to MockTransport(handler)."""
    pool = coord.WorkerPool.__new__(coord.WorkerPool)  # bypass __init__
    pool.urls = list(urls)
    pool._timeout = 5.0
    pool._clients = {
        u: httpx.Client(transport=httpx.MockTransport(handler), base_url=u)
        for u in urls
    }
    return pool


# --------------------------------------------------------------------------
# Happy path: one worker, two individuals, both complete fine
# --------------------------------------------------------------------------

def test_evaluate_population_dispatches_to_single_worker():
    received = []

    def handler(request):
        received.append(json.loads(request.content))
        return httpx.Response(200, json={
            "case_vector": [-1.0, 0.0, -2.0, 1.0],   # 2 metrics × 2 scenarios
            "history": {"min_dist_target_m": 1.0, "reached_target": False,
                        "collision_events": 2, "elapsed_s": 30.0},
            "wall_time_s": 30.5,
        })

    pool = _build_pool_with_handler(["http://w1:8000"], handler)
    pop = [_trivial_tree(), _trivial_tree()]
    out = coord._evaluate_population(
        pool, pop, _SCENARIOS, _WORLD, _METRICS,
        duration_s=30.0, tick_hz=10.0, timing=None, gen=0,
        n_metrics_per_scenario=len(_METRICS), retries=0,
    )
    assert len(out) == 2
    assert all(len(r) == 4 for r in out)
    # 2 individuals × 2 scenarios = 4 POSTs
    assert len(received) == 4
    pool.close()


# --------------------------------------------------------------------------
# Round-robin: 4 individuals across 2 workers should split 2-2
# --------------------------------------------------------------------------

def test_round_robin_split():
    counts = {"http://w1:8000": 0, "http://w2:8000": 0}

    def handler(request):
        # Identify which worker by host
        url = f"http://{request.url.host}:{request.url.port or 8000}"
        counts[url] += 1
        return httpx.Response(200, json={
            "case_vector": [-1.0, 0.0, -2.0, 1.0],
            "history": {"min_dist_target_m": 1.0, "reached_target": False,
                        "collision_events": 0, "elapsed_s": 30.0},
            "wall_time_s": 1.0,
        })

    pool = _build_pool_with_handler(["http://w1:8000", "http://w2:8000"],
                                    handler)
    pop = [_trivial_tree() for _ in range(4)]
    coord._evaluate_population(
        pool, pop, _SCENARIOS, _WORLD, _METRICS,
        duration_s=30.0, tick_hz=10.0, timing=None, gen=0,
        n_metrics_per_scenario=len(_METRICS), retries=0,
    )
    # Each individual = 2 scenarios = 2 POSTs. 4 inds × 2 = 8 POSTs.
    # Two individuals per worker → 4 POSTs each.
    assert counts["http://w1:8000"] == 4
    assert counts["http://w2:8000"] == 4
    pool.close()


# --------------------------------------------------------------------------
# Failure & retry: w1 always 500s, retry should succeed on w2
# --------------------------------------------------------------------------

def test_failure_retries_on_other_worker():
    def handler(request):
        if request.url.host == "w1":
            return httpx.Response(500, json={"detail": "boom"})
        return httpx.Response(200, json={
            "case_vector": [-1.0, 0.0, -2.0, 1.0],
            "history": {"min_dist_target_m": 1.0, "reached_target": False,
                        "collision_events": 0, "elapsed_s": 30.0},
            "wall_time_s": 1.0,
        })

    pool = _build_pool_with_handler(["http://w1:8000", "http://w2:8000"],
                                    handler)
    pop = [_trivial_tree()]
    out = coord._evaluate_population(
        pool, pop, _SCENARIOS, _WORLD, _METRICS,
        duration_s=30.0, tick_hz=10.0, timing=None, gen=0,
        n_metrics_per_scenario=len(_METRICS), retries=1,
    )
    assert out[0] == [-1.0, 0.0, -2.0, 1.0]   # rescued by w2
    pool.close()


def test_persistent_failure_yields_sentinel_vector():
    def handler(request):
        return httpx.Response(500, json={"detail": "always broken"})

    pool = _build_pool_with_handler(["http://w1:8000", "http://w2:8000"],
                                    handler)
    pop = [_trivial_tree()]
    out = coord._evaluate_population(
        pool, pop, _SCENARIOS, _WORLD, _METRICS,
        duration_s=30.0, tick_hz=10.0, timing=None, gen=0,
        n_metrics_per_scenario=len(_METRICS), retries=2,
    )
    # Sentinel = -1e6 in every slot, length n_metrics × n_scenarios
    assert out[0] == [coord._FAILURE_SENTINEL] * 4
    pool.close()


# --------------------------------------------------------------------------
# Result alignment: parallel dispatch must not scramble the output indices
# --------------------------------------------------------------------------

def test_results_aligned_by_index_under_concurrency():
    """Workers reply with different vectors; output[i] must match population[i]."""
    def handler(request):
        body = json.loads(request.content)
        # Encode the genome's leaf action into the case vector so we can
        # check alignment after concurrent dispatch.
        tree = json.loads(body["genome_json"])
        action = tree["leaf"]
        signature = {"FRENTE": 1.0, "RE": 2.0, "GIRA_ESQ": 3.0,
                     "GIRA_DIR": 4.0}[action]
        return httpx.Response(200, json={
            "case_vector": [signature, signature],   # 2 metrics × 1 scenario
            "history": {"min_dist_target_m": 1.0, "reached_target": False,
                        "collision_events": 0, "elapsed_s": 1.0},
            "wall_time_s": 0.1,
        })

    pool = _build_pool_with_handler(["http://w1:8000", "http://w2:8000"],
                                    handler)
    pop = [
        {"leaf": "FRENTE",   "dur_ms": 200},
        {"leaf": "RE",       "dur_ms": 200},
        {"leaf": "GIRA_ESQ", "dur_ms": 200},
        {"leaf": "GIRA_DIR", "dur_ms": 200},
    ]
    out = coord._evaluate_population(
        pool, pop, [{"name": "x", "start": [0, 0, 0], "target": [1, 1]}],
        _WORLD, _METRICS,
        duration_s=1.0, tick_hz=10.0, timing=None, gen=0,
        n_metrics_per_scenario=len(_METRICS), retries=0,
    )
    assert out[0][0] == 1.0   # FRENTE
    assert out[1][0] == 2.0   # RE
    assert out[2][0] == 3.0   # GIRA_ESQ
    assert out[3][0] == 4.0   # GIRA_DIR
    pool.close()
