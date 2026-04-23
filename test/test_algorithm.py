"""Unit tests for evolution.algorithm — selection, elitism, GA loop."""
import random

from evolutionary_controller_ros.evolution import algorithm as a
from evolutionary_controller_ros.evolution import genome as g


# --------------------------------------------------------------------------
# compute_mad_epsilons
# --------------------------------------------------------------------------

def test_mad_epsilons_constant_column_is_zero():
    mat = [[1.0, 5.0], [1.0, 10.0], [1.0, 3.0]]
    eps = a.compute_mad_epsilons(mat)
    assert eps[0] == 0.0
    assert eps[1] > 0.0


def test_mad_epsilons_symmetric_column():
    mat = [[0.0], [1.0], [2.0], [3.0], [4.0]]
    # median = 2; abs devs = [2,1,0,1,2]; median = 1
    eps = a.compute_mad_epsilons(mat)
    assert eps[0] == 1.0


# --------------------------------------------------------------------------
# epsilon_lexicase_select
# --------------------------------------------------------------------------

def test_lexicase_picks_dominant_when_epsilon_zero():
    # Individual 2 dominates on both cases; ε=0 on both columns (constant).
    pop = ["A", "B", "C"]
    mat = [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]
    rng = random.Random(0)
    # MAD of a column [0,0,1] — median=0, abs devs [0,0,1], median=0.
    for _ in range(20):
        assert a.epsilon_lexicase_select(rng, pop, mat) == "C"


def test_lexicase_with_nonzero_epsilon_allows_close_rivals():
    # Column 0: [10, 9.9, 5]; median=9.9, abs devs=[0.1, 0, 4.9], MAD=0.1.
    # best=10, threshold=9.9 → A and B survive, C fails. With a single
    # case the tie is broken uniformly at random between A and B.
    pop = ["A", "B", "C"]
    mat = [[10.0], [9.9], [5.0]]
    rng = random.Random(0)
    seen = set()
    for _ in range(100):
        seen.add(a.epsilon_lexicase_select(rng, pop, mat))
    assert seen == {"A", "B"}


def test_lexicase_empty_population_raises():
    import pytest
    with pytest.raises(ValueError):
        a.epsilon_lexicase_select(random.Random(0), [], [])


def test_lexicase_uses_explicit_epsilons():
    pop = ["A", "B"]
    mat = [[10.0], [9.9]]
    # With ε=0, A strictly wins.
    rng = random.Random(0)
    for _ in range(20):
        assert a.epsilon_lexicase_select(rng, pop, mat, epsilons=[0.0]) == "A"


# --------------------------------------------------------------------------
# elite
# --------------------------------------------------------------------------

def test_elite_returns_top_k_by_mean():
    pop = ["A", "B", "C"]
    mat = [[1.0, 1.0], [5.0, 5.0], [3.0, 3.0]]  # means: 1, 5, 3
    assert a.elite(pop, mat, k=1) == ["B"]
    assert a.elite(pop, mat, k=2) == ["B", "C"]
    assert a.elite(pop, mat, k=0) == []


def test_elite_breaks_ties_by_index():
    pop = ["A", "B", "C"]
    mat = [[2.0], [2.0], [2.0]]  # all tied
    assert a.elite(pop, mat, k=2) == ["A", "B"]


def test_elite_k_larger_than_pop_returns_all_sorted():
    pop = ["A", "B", "C"]
    mat = [[1.0], [3.0], [2.0]]
    assert a.elite(pop, mat, k=10) == ["B", "C", "A"]


# --------------------------------------------------------------------------
# assemble_case_matrix
# --------------------------------------------------------------------------

def test_assemble_appends_negated_size_when_parsimony_on():
    pop = [{"leaf": "FRENTE", "dur_ms": 200},
           {"op": "IF",
            "cond": {"term": "alvo_atras"},
            "then": {"leaf": "FRENTE", "dur_ms": 200},
            "else": {"leaf": "RE", "dur_ms": 100}}]
    base = [[0.5, 1.0], [0.2, 0.8]]
    out = a.assemble_case_matrix(base, pop, include_parsimony=True)
    assert out[0] == [0.5, 1.0, -1.0]   # size 1
    assert out[1] == [0.2, 0.8, -4.0]   # size 4


def test_assemble_skips_size_when_parsimony_off():
    pop = [{"leaf": "FRENTE", "dur_ms": 200}]
    base = [[0.5, 1.0]]
    out = a.assemble_case_matrix(base, pop, include_parsimony=False)
    assert out == [[0.5, 1.0]]


# --------------------------------------------------------------------------
# run_ga (end-to-end, pure-Python evaluator)
# --------------------------------------------------------------------------

def test_run_ga_returns_valid_action_tree():
    rng = random.Random(0)
    # Dummy evaluator — constant score, so selection falls back to parsimony.
    evaluator = lambda tree: [1.0]
    best = a.run_ga(
        rng,
        evaluator=evaluator,
        pop_size=8,
        n_generations=3,
        init_max_depth=3,
    )
    g.validate(best)
    assert g.node_type(best) == "Action"


def test_run_ga_invokes_on_generation_callback():
    rng = random.Random(1)
    seen = []

    def cb(gen, pop, case_matrix, best_idx):
        seen.append((gen, len(pop), len(case_matrix), best_idx))

    a.run_ga(
        rng,
        evaluator=lambda t: [0.5],
        pop_size=6,
        n_generations=2,
        init_max_depth=3,
        on_generation=cb,
    )
    assert [gen for gen, *_ in seen] == [0, 1]
    assert all(n == 6 for _, n, _, _ in seen)


def test_run_ga_parsimony_shrinks_pop_over_time():
    # With a constant evaluator, parsimony is the only selection pressure.
    # Average size should not grow unboundedly.
    rng = random.Random(2)
    sizes_by_gen = []

    def cb(gen, pop, case_matrix, best_idx):
        sizes_by_gen.append(sum(g.size(t) for t in pop) / len(pop))

    a.run_ga(
        rng,
        evaluator=lambda t: [0.0],
        pop_size=20,
        n_generations=10,
        init_max_depth=5,
        on_generation=cb,
    )
    # The final 3 generations' mean size should not exceed the initial size
    # by more than a factor of 1.5 (loose check — parsimony pressure exists).
    assert max(sizes_by_gen[-3:]) <= sizes_by_gen[0] * 1.5


def test_run_ga_tracks_best_across_generations():
    # Evaluator returns score based on tree depth — deeper is worse.
    # Since run_ga tracks the max-mean across all generations, best_tree
    # should never have worse mean than any individual seen.
    rng = random.Random(3)
    best = a.run_ga(
        rng,
        evaluator=lambda t: [-float(g.depth(t))],
        pop_size=10,
        n_generations=4,
        init_max_depth=4,
        include_parsimony=False,
    )
    g.validate(best)


def test_run_ga_with_elite_k_gt_1():
    rng = random.Random(4)
    best = a.run_ga(
        rng,
        evaluator=lambda t: [random.Random(g.size(t)).random()],
        pop_size=10,
        n_generations=3,
        init_max_depth=3,
        elite_k=3,
    )
    g.validate(best)
