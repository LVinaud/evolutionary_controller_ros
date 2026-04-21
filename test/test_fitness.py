"""Unit tests for evolution.fitness — pure transformation of an history dict."""
import math

import pytest

from evolutionary_controller_ros.evolution import fitness as f


def _base_history(**overrides):
    """Build a minimal valid history dict; override keys in overrides."""
    h = {
        "positions_xy": [(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)],
        "dt_s": 0.1,
        "flag_visible_ticks": 0,
        "holding_any_tick": False,
        "delivered": False,
        "min_dist_deploy_holding": math.inf,
        "collision_events": 0,
        "time_to_deliver_s": None,
        "scenario_time_s": 30.0,
    }
    h.update(overrides)
    return h


# --------------------------------------------------------------------------
# shape / names
# --------------------------------------------------------------------------

def test_vector_has_seven_metrics():
    cases = f.compute_fitness_cases(_base_history())
    assert len(cases) == 7
    assert len(f.METRIC_NAMES) == 7


# --------------------------------------------------------------------------
# espaco_explorado
# --------------------------------------------------------------------------

def test_exploration_counts_unique_cells():
    # 3 points, all in different 0.5m cells.
    h = _base_history(positions_xy=[(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)])
    cases = f.compute_fitness_cases(h, exploration_cell_size_m=0.5)
    assert cases[0] == 3.0


def test_exploration_dedups_repeated_cell():
    # Six points all within the same 0.5m cell → count = 1.
    h = _base_history(positions_xy=[(0.0, 0.0), (0.1, 0.1),
                                    (0.2, 0.2), (0.3, 0.3),
                                    (0.4, 0.4), (0.49, 0.49)])
    cases = f.compute_fitness_cases(h, exploration_cell_size_m=0.5)
    assert cases[0] == 1.0


def test_exploration_handles_negative_coords():
    # Points straddling the origin should still count correctly.
    h = _base_history(positions_xy=[(-0.1, -0.1), (-0.6, 0.0),
                                    (0.4, 0.4), (0.6, 0.6)])
    cases = f.compute_fitness_cases(h, exploration_cell_size_m=0.5)
    # Cells: (-1,-1), (-2,0), (0,0), (1,1) → 4 unique.
    assert cases[0] == 4.0


def test_exploration_rejects_zero_cell_size():
    h = _base_history()
    with pytest.raises(ValueError):
        f.compute_fitness_cases(h, exploration_cell_size_m=0.0)


# --------------------------------------------------------------------------
# tempo_bandeira_visivel
# --------------------------------------------------------------------------

def test_flag_visible_time_multiplies_ticks_by_dt():
    h = _base_history(flag_visible_ticks=50, dt_s=0.1)
    cases = f.compute_fitness_cases(h)
    assert cases[1] == pytest.approx(5.0)


# --------------------------------------------------------------------------
# pegou / entregou (0/1 booleans)
# --------------------------------------------------------------------------

def test_picked_flag_maps_to_one_zero():
    assert f.compute_fitness_cases(_base_history(holding_any_tick=True))[2] == 1.0
    assert f.compute_fitness_cases(_base_history(holding_any_tick=False))[2] == 0.0


def test_delivered_flag_maps_to_one_zero():
    assert f.compute_fitness_cases(_base_history(delivered=True))[4] == 1.0
    assert f.compute_fitness_cases(_base_history(delivered=False))[4] == 0.0


# --------------------------------------------------------------------------
# dist_min_deploy_holding (MIN → negated; 0 when inf)
# --------------------------------------------------------------------------

def test_min_dist_deploy_returns_zero_when_never_held():
    cases = f.compute_fitness_cases(
        _base_history(min_dist_deploy_holding=math.inf))
    assert cases[3] == 0.0


def test_min_dist_deploy_is_negated_when_finite():
    cases = f.compute_fitness_cases(
        _base_history(min_dist_deploy_holding=1.7))
    assert cases[3] == pytest.approx(-1.7)


# --------------------------------------------------------------------------
# colisoes (MIN → negated)
# --------------------------------------------------------------------------

def test_collision_count_is_negated():
    cases = f.compute_fitness_cases(_base_history(collision_events=4))
    assert cases[5] == -4.0


def test_collision_larger_is_better_after_negation():
    # Fewer collisions → larger (less negative) case.
    a = f.compute_fitness_cases(_base_history(collision_events=2))[5]
    b = f.compute_fitness_cases(_base_history(collision_events=10))[5]
    assert a > b


# --------------------------------------------------------------------------
# tempo_ate_entrega (MIN → negated; uses scenario_time_s when never delivered)
# --------------------------------------------------------------------------

def test_time_to_deliver_uses_scenario_time_when_none():
    cases = f.compute_fitness_cases(
        _base_history(time_to_deliver_s=None, scenario_time_s=30.0))
    assert cases[6] == -30.0


def test_time_to_deliver_is_negated_when_known():
    cases = f.compute_fitness_cases(
        _base_history(time_to_deliver_s=12.5, scenario_time_s=30.0))
    assert cases[6] == -12.5


def test_faster_delivery_yields_higher_case():
    fast = f.compute_fitness_cases(
        _base_history(time_to_deliver_s=5.0, scenario_time_s=30.0))[6]
    slow = f.compute_fitness_cases(
        _base_history(time_to_deliver_s=20.0, scenario_time_s=30.0))[6]
    assert fast > slow
