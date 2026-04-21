"""Fitness decomposition for the CTF task — pure, ROS-free.

Given an `history` dict pre-computed by `evaluation/episode.py`, this
module produces the 7-metric case vector for a single scenario, already
in lexicase convention (larger is better).

Seven per-scenario metrics:
    (a) espaco_explorado              MAX — unique 0.5m cells visited
    (b) tempo_bandeira_visivel_s      MAX — seconds with enemy flag in camera
    (c) pegou_bandeira                MAX — 0/1, ever held the flag
    (d) dist_min_zona_deploy_holding  MIN — min dist to deploy while holding
                                            (inf if never picked up)
    (e) entregou_bandeira             MAX — 0/1, deploy event happened
    (f) colisoes                      MIN — count of distinct collision events
    (g) tempo_ate_entrega_s           MIN — seconds until deploy (scenario_time
                                            if never delivered)

Conversion to lexicase ("larger is better")
    MAX metrics are returned as-is.
    MIN metrics are negated:
        colisoes           -> -colisoes
        tempo_ate_entrega  -> -tempo_ate_entrega
        dist_min_deploy    -> -dist_min_deploy     (or 0 if inf — never held)

Multi-scenario composition is done by the caller (e.g. `orchestrator.py`):
per-scenario case vectors of length 7 are concatenated into a single
length-(7 × n_scenarios) vector before being handed to the GA.

Expected `history` keys (populated by `episode.py`):
    positions_xy        : list[(x, y)]   robot pose samples (odom_gt)
    dt_s                : float          sampling interval of positions_xy
    flag_visible_ticks  : int            number of samples with flag in view
    holding_any_tick    : bool           ever segurando_bandeira == True
    delivered           : bool           deploy event happened
    min_dist_deploy_holding : float      smallest dist to deploy zone while
                                         holding the flag (math.inf if never)
    collision_events    : int            debounced collisions (see episode.py)
    time_to_deliver_s   : float | None   seconds until deploy; None if never
    scenario_time_s     : float          episode cap

Grid cell size for exploration is configurable (default 0.5m, matching the
design decision).
"""
import math


METRIC_NAMES = (
    "espaco_explorado",
    "tempo_bandeira_visivel_s",
    "pegou_bandeira",
    "dist_min_deploy_holding_neg",
    "entregou_bandeira",
    "colisoes_neg",
    "tempo_ate_entrega_neg",
)


def compute_fitness_cases(
    history: dict,
    *,
    exploration_cell_size_m: float = 0.5,
) -> list:
    """Return the 7-metric case vector (lexicase convention: larger is better)."""
    explored = _count_unique_cells(history["positions_xy"],
                                   exploration_cell_size_m)
    flag_time_s = history["flag_visible_ticks"] * history["dt_s"]
    picked = 1.0 if history["holding_any_tick"] else 0.0
    delivered = 1.0 if history["delivered"] else 0.0

    min_dist = history["min_dist_deploy_holding"]
    dist_case = -min_dist if math.isfinite(min_dist) else 0.0

    collisions_case = -float(history["collision_events"])

    t_deliver = history["time_to_deliver_s"]
    if t_deliver is None:
        t_deliver = history["scenario_time_s"]
    deliver_case = -float(t_deliver)

    return [
        float(explored),
        float(flag_time_s),
        picked,
        dist_case,
        delivered,
        collisions_case,
        deliver_case,
    ]


def _count_unique_cells(positions_xy: list, cell_size_m: float) -> int:
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be > 0")
    seen = set()
    for x, y in positions_xy:
        seen.add((int(x // cell_size_m), int(y // cell_size_m)))
    return len(seen)
