"""Fitness decomposition for the go-to-goal task — pure, ROS-free.

Given an `history` dict pre-computed by `evaluation/episode.py`, this
module produces the per-scenario case vector in lexicase convention
(larger is better).

Three per-scenario metrics by default:
    (a) dist_min_alvo_neg   MAX — negated min distance to target during episode
                                  (always finite, always different — dense signal)
    (b) alcancou_alvo       MAX — 0/1, ever got within reach_radius of target
    (c) colisoes_neg        MAX — negated debounced collision count

Any subset of the metrics can be selected via `enabled_metrics` — useful
if Lazaro wants to simplify even further during tuning.

Conversion to lexicase ("larger is better")
    alcancou_alvo is already MAX.
    dist_min and colisoes are MIN — returned as `-x`.

Multi-scenario composition is done by the caller (`orchestrator.py`):
per-scenario case vectors are concatenated into a single length
(n_metrics × n_scenarios) vector before being handed to the GA.

Expected `history` keys (populated by `episode.py`):
    positions_xy        : list[(x, y)]   robot pose samples (odom_gt)
    dt_s                : float          sampling interval of positions_xy
    target_pose         : (x, y)         scenario target
    min_dist_target_m   : float          smallest d(robot, target) observed
    reached_target      : bool           min_dist_target_m < reach_radius
    collision_events    : int            debounced collisions (see episode.py)
    scenario_time_s     : float          episode cap
"""
ALL_METRICS = (
    "dist_min_alvo_neg",
    "alcancou_alvo",
    "colisoes_neg",
)


def compute_fitness_cases(
    history: dict,
    *,
    enabled_metrics: tuple = ALL_METRICS,
) -> list:
    """Return the per-scenario case vector under lexicase convention.

    `enabled_metrics` filters which of the `ALL_METRICS` are emitted (in
    the same order). Order is preserved so downstream lexicase caseness
    stays consistent across scenarios.
    """
    cases = []
    for name in enabled_metrics:
        if name not in ALL_METRICS:
            raise ValueError(f"unknown metric: {name!r}")
        cases.append(_metric(history, name))
    return cases


def _metric(history: dict, name: str) -> float:
    if name == "dist_min_alvo_neg":
        return -float(history["min_dist_target_m"])
    if name == "alcancou_alvo":
        return 1.0 if history["reached_target"] else 0.0
    if name == "colisoes_neg":
        return -float(history["collision_events"])
    raise ValueError(f"unhandled metric: {name!r}")
