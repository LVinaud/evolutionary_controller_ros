"""Tests for the Redis-relayed coordinator — pure Python, no Gazebo, no ROS.

Strategy: replace `redis.Redis` with `fakeredis.FakeRedis`, queue some
synthetic results in advance (or after a short delay), and assert the
coordinator's `_evaluate_population` matches them back to the right
population slots and applies the failure sentinel where needed.

Auto-skips if redis-py / fakeredis are not installed.
"""
import importlib.util
import json
import threading
import time

import pytest

_HAVE_DEPS = (importlib.util.find_spec("redis") is not None
              and importlib.util.find_spec("fakeredis") is not None)
pytestmark = pytest.mark.skipif(
    not _HAVE_DEPS,
    reason="redis-relay tests need redis-py + fakeredis — "
           "`pip install redis fakeredis`")

if _HAVE_DEPS:
    import fakeredis
    from evolutionary_controller_ros.evaluation import redis_coordinator as rc


# --------------------------------------------------------------------------

_SCENARIOS = [
    {"name": "to_origin", "start": [3.0, 0.0, 3.14], "target": [0.0, 0.0]},
    {"name": "to_corner", "start": [0.0, 0.0, 0.0],  "target": [3.0, 3.0]},
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


def _fake_redis():
    return fakeredis.FakeRedis(decode_responses=True)


# --------------------------------------------------------------------------
# Worker-side simulation: drain evo:jobs, push fake results
# --------------------------------------------------------------------------

def _fake_worker_loop(r, stop_event, response_fn):
    """Drain evo:jobs and push synthetic results until stop_event is set."""
    while not stop_event.is_set():
        item = r.blpop("evo:jobs", timeout=1)
        if item is None:
            continue
        _key, raw = item
        job = json.loads(raw)
        result = response_fn(job)
        r.rpush("evo:results", json.dumps(result))


# --------------------------------------------------------------------------
# Happy path: every job comes back with a synthetic case vector
# --------------------------------------------------------------------------

def test_evaluate_population_happy_path():
    r = _fake_redis()
    stop = threading.Event()

    def respond(job):
        return {
            "job_id": job["job_id"],
            "case_vector": [-1.0, 0.0],   # 2 metrics
            "history": {"min_dist_target_m": 1.0, "reached_target": False,
                        "collision_events": 0, "elapsed_s": 1.0},
            "wall_time_s": 0.1,
            "error": None,
        }

    t = threading.Thread(target=_fake_worker_loop, args=(r, stop, respond))
    t.start()
    try:
        out = rc._evaluate_population(
            r, run_id="abc123", population=[_trivial_tree(), _trivial_tree()],
            scenarios=_SCENARIOS, world=_WORLD, enabled_metrics=_METRICS,
            duration_s=1.0, tick_hz=10.0, n_metrics_per_scenario=2,
            per_job_timeout_s=2.0, timing=None, gen=0,
        )
    finally:
        stop.set()
        t.join(timeout=2.0)

    # 2 individuals × 2 scenarios × 2 metrics = 4 floats per individual
    assert len(out) == 2
    for vec in out:
        assert vec == [-1.0, 0.0, -1.0, 0.0]


# --------------------------------------------------------------------------
# Worker reports an error → coordinator substitutes sentinel for that scenario
# --------------------------------------------------------------------------

def test_evaluate_population_error_yields_sentinel_slot():
    r = _fake_redis()
    stop = threading.Event()

    def respond(job):
        # Fail any job whose scenario index is 1 (second scenario per indiv).
        scenario_idx = int(job["job_id"].rsplit(":", 1)[-1])
        if scenario_idx == 1:
            return {"job_id": job["job_id"], "case_vector": [],
                    "history": {}, "wall_time_s": 0.0,
                    "error": "RuntimeError: simulated"}
        return {
            "job_id": job["job_id"],
            "case_vector": [-2.0, 1.0],
            "history": {"min_dist_target_m": 2.0, "reached_target": True,
                        "collision_events": 0, "elapsed_s": 1.0},
            "wall_time_s": 0.1,
            "error": None,
        }

    t = threading.Thread(target=_fake_worker_loop, args=(r, stop, respond))
    t.start()
    try:
        out = rc._evaluate_population(
            r, run_id="run42", population=[_trivial_tree()],
            scenarios=_SCENARIOS, world=_WORLD, enabled_metrics=_METRICS,
            duration_s=1.0, tick_hz=10.0, n_metrics_per_scenario=2,
            per_job_timeout_s=2.0, timing=None, gen=0,
        )
    finally:
        stop.set()
        t.join(timeout=2.0)

    # First scenario succeeded, second fell back to sentinel.
    assert out[0][0:2] == [-2.0, 1.0]
    assert out[0][2:4] == [rc._FAILURE_SENTINEL, rc._FAILURE_SENTINEL]


# --------------------------------------------------------------------------
# Stale results from a different run_id are ignored (don't break the match)
# --------------------------------------------------------------------------

def test_stale_results_from_other_run_are_ignored():
    r = _fake_redis()

    # Pre-seed a stale result from "old_run" for the same individual/scenario
    # indices that the new run will use. The stale job_id has a different
    # run prefix, so the coordinator must NOT consume it as a real result.
    r.rpush("evo:results", json.dumps({
        "job_id": "old_run:0:0", "case_vector": [-99.0, -99.0],
        "history": {}, "wall_time_s": 0.0, "error": None,
    }))

    stop = threading.Event()

    def respond(job):
        return {"job_id": job["job_id"], "case_vector": [-1.0, 0.0],
                "history": {}, "wall_time_s": 0.0, "error": None}

    t = threading.Thread(target=_fake_worker_loop, args=(r, stop, respond))
    t.start()
    try:
        out = rc._evaluate_population(
            r, run_id="new_run", population=[_trivial_tree()],
            scenarios=_SCENARIOS, world=_WORLD, enabled_metrics=_METRICS,
            duration_s=1.0, tick_hz=10.0, n_metrics_per_scenario=2,
            per_job_timeout_s=2.0, timing=None, gen=0,
        )
    finally:
        stop.set()
        t.join(timeout=2.0)

    # Real result was consumed; stale result was filtered out (it would
    # have given -99 if it were used).
    assert out[0] == [-1.0, 0.0, -1.0, 0.0]


# --------------------------------------------------------------------------
# No worker ever pulls jobs → all individuals time out and get sentinels
# --------------------------------------------------------------------------

def test_timeout_marks_unresponded_individuals_as_failed():
    r = _fake_redis()
    # No worker thread → results never come back.
    out = rc._evaluate_population(
        r, run_id="solo", population=[_trivial_tree()],
        scenarios=_SCENARIOS, world=_WORLD, enabled_metrics=_METRICS,
        duration_s=1.0, tick_hz=10.0, n_metrics_per_scenario=2,
        per_job_timeout_s=1.0,        # tight: 1s × 2 jobs = 2s total budget
        timing=None, gen=0,
    )
    assert out[0] == [rc._FAILURE_SENTINEL] * 4
