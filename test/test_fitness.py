"""Unit tests for evolution.fitness — go-to-goal case vector."""
import pytest

from evolutionary_controller_ros.evolution import fitness as fit


def _history(**over) -> dict:
    h = dict(
        positions_xy=[(0.0, 0.0)],
        dt_s=0.1,
        target_pose=(5.0, 0.0),
        min_dist_target_m=3.0,
        reached_target=False,
        collision_events=0,
        scenario_time_s=30.0,
    )
    h.update(over)
    return h


# --------------------------------------------------------------------------
# Case vector structure
# --------------------------------------------------------------------------

def test_returns_one_case_per_enabled_metric():
    v = fit.compute_fitness_cases(_history())
    assert len(v) == len(fit.ALL_METRICS)


def test_enabled_subset_filters_cases_and_preserves_order():
    v = fit.compute_fitness_cases(
        _history(),
        enabled_metrics=("alcancou_alvo", "colisoes_neg"),
    )
    assert len(v) == 2


def test_unknown_metric_name_raises():
    with pytest.raises(ValueError):
        fit.compute_fitness_cases(_history(), enabled_metrics=("nope",))


# --------------------------------------------------------------------------
# Per-metric semantics
# --------------------------------------------------------------------------

def test_dist_min_is_negated():
    v = fit.compute_fitness_cases(_history(min_dist_target_m=2.5))
    idx = fit.ALL_METRICS.index("dist_min_alvo_neg")
    assert v[idx] == -2.5


def test_closer_min_dist_yields_higher_case():
    close = fit.compute_fitness_cases(_history(min_dist_target_m=0.5))
    far   = fit.compute_fitness_cases(_history(min_dist_target_m=8.0))
    idx = fit.ALL_METRICS.index("dist_min_alvo_neg")
    assert close[idx] > far[idx]


def test_alcancou_alvo_binary():
    hit = fit.compute_fitness_cases(_history(reached_target=True))
    miss = fit.compute_fitness_cases(_history(reached_target=False))
    idx = fit.ALL_METRICS.index("alcancou_alvo")
    assert hit[idx] == 1.0
    assert miss[idx] == 0.0


def test_collision_count_is_negated():
    v = fit.compute_fitness_cases(_history(collision_events=5))
    idx = fit.ALL_METRICS.index("colisoes_neg")
    assert v[idx] == -5.0


def test_fewer_collisions_yields_higher_case():
    clean  = fit.compute_fitness_cases(_history(collision_events=0))
    crashy = fit.compute_fitness_cases(_history(collision_events=10))
    idx = fit.ALL_METRICS.index("colisoes_neg")
    assert clean[idx] > crashy[idx]
