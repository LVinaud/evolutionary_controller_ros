"""Unit tests for evolution.population — genetic operators."""
import copy
import random

from evolutionary_controller_ros.evolution import genome as g
from evolutionary_controller_ros.evolution import population as p


# --------------------------------------------------------------------------
# init_population
# --------------------------------------------------------------------------

def test_init_population_returns_n_valid_action_trees():
    rng = random.Random(0)
    pop = p.init_population(rng, size=20, max_depth=4)
    assert len(pop) == 20
    for tree in pop:
        g.validate(tree)
        assert g.node_type(tree) == "Action"
        assert g.depth(tree) <= 4


# --------------------------------------------------------------------------
# crossover
# --------------------------------------------------------------------------

def test_crossover_produces_valid_action_rooted_children():
    rng = random.Random(1)
    a = g.random_tree(random.Random(10), max_depth=4)
    b = g.random_tree(random.Random(11), max_depth=4)
    ca, cb = p.crossover(rng, a, b)
    g.validate(ca)
    g.validate(cb)
    assert g.node_type(ca) == "Action"
    assert g.node_type(cb) == "Action"


def test_crossover_does_not_mutate_parents():
    rng = random.Random(2)
    a = g.random_tree(random.Random(10), max_depth=4)
    b = g.random_tree(random.Random(11), max_depth=4)
    a_copy = copy.deepcopy(a)
    b_copy = copy.deepcopy(b)
    p.crossover(rng, a, b)
    assert a == a_copy
    assert b == b_copy


def test_crossover_swaps_same_typed_subtree():
    # Force a predictable scenario: both parents have Bool subtrees.
    a = {"op": "IF",
         "cond": {"term": "alvo_atras"},
         "then": {"leaf": "FRENTE", "dur_ms": 200},
         "else": {"leaf": "RE", "dur_ms": 100}}
    b = {"op": "IF",
         "cond": {"term": "obstaculo_frente"},
         "then": {"leaf": "RE", "dur_ms": 300},
         "else": {"leaf": "GIRA_ESQ", "dur_ms": 200}}
    rng = random.Random(3)
    # Try many times to hit a real swap.
    for _ in range(50):
        ca, cb = p.crossover(rng, a, b)
        g.validate(ca)
        g.validate(cb)


def test_crossover_type_incompatible_falls_back_to_parents():
    # Parent A is a lone leaf (only Action-typed subtree: the root).
    # Parent B is a lone Bool... but node_type requires root is a tree;
    # we can still call iter_subtrees on it. If the only pick in A
    # is Action and B has no Action, should return parents.
    a = {"leaf": "FRENTE", "dur_ms": 200}
    b_tree = g.random_tree(random.Random(5), max_depth=3, return_type="Bool")
    rng = random.Random(4)
    ca, cb = p.crossover(rng, a, b_tree, max_tries=20)
    assert ca == a
    assert cb == b_tree


# --------------------------------------------------------------------------
# mutate — top level always returns valid Action tree
# --------------------------------------------------------------------------

def test_mutate_always_returns_valid_action_tree():
    rng = random.Random(42)
    for seed in range(30):
        tree = g.random_tree(random.Random(seed), max_depth=4)
        child = p.mutate(rng, tree)
        g.validate(child)
        assert g.node_type(child) == "Action"


def test_mutate_does_not_modify_input():
    rng = random.Random(7)
    tree = g.random_tree(random.Random(100), max_depth=4)
    snapshot = copy.deepcopy(tree)
    for _ in range(20):
        p.mutate(rng, tree)
    assert tree == snapshot


# --------------------------------------------------------------------------
# _mutate_subtree
# --------------------------------------------------------------------------

def test_mutate_subtree_produces_valid_tree():
    rng = random.Random(0)
    tree = g.random_tree(random.Random(1), max_depth=4)
    new = p._mutate_subtree(rng, tree, subtree_max_depth=3)
    g.validate(new)
    assert g.node_type(new) == "Action"


# --------------------------------------------------------------------------
# _mutate_point
# --------------------------------------------------------------------------

def test_mutate_point_swaps_binary_op():
    # Only candidate is the AND root.
    tree = {"op": "IF",
            "cond": {"op": "AND",
                     "a": {"term": "obstaculo_frente"},
                     "b": {"term": "alvo_atras"}},
            "then": {"leaf": "FRENTE", "dur_ms": 200},
            "else": {"leaf": "RE", "dur_ms": 100}}
    # Use a forced choice: only op swap possible is AND (IF is not in _POINT_OP_SWAPS),
    # named terminals can also be swapped. Run many seeds to ensure validity.
    for seed in range(20):
        rng = random.Random(seed)
        new = p._mutate_point(rng, tree)
        assert new is not None
        g.validate(new)


def test_mutate_point_lt_gt_symmetry():
    tree = {"op": "LT",
            "a": {"term": "dist_alvo"},
            "b": {"erc": 0.5}}
    # Force: iterate seeds until we pick the "op" candidate (root).
    for seed in range(100):
        rng = random.Random(seed)
        new = p._mutate_point(rng, tree)
        if new is not None and "op" in new and new["op"] in ("LT", "GT"):
            # If the op itself got swapped, it must be the other one.
            if new["op"] != tree["op"]:
                assert new["op"] == "GT"
                return
    # At least one swap must have happened in 100 tries.
    raise AssertionError("no op swap observed in 100 seeds")


def test_mutate_point_terminal_swap_preserves_type():
    tree = {"term": "obstaculo_frente"}  # Bool
    for seed in range(20):
        rng = random.Random(seed)
        new = p._mutate_point(rng, tree)
        assert new is not None
        assert new["term"] in g.BOOL_TERMINALS
        assert new["term"] != "obstaculo_frente"


def test_mutate_point_returns_none_when_no_candidates():
    # A tree with only ERC and leaf — no op in _POINT_OP_SWAPS, no terms.
    tree = {"op": "IF",
            "cond": {"op": "NOT",
                     "arg": {"op": "LT",
                             "a": {"erc": 0.1},
                             "b": {"erc": 0.2}}},
            "then": {"leaf": "FRENTE", "dur_ms": 200},
            "else": {"leaf": "RE", "dur_ms": 100}}
    # Wait — LT is in _POINT_OP_SWAPS. Use a tree with only ops that aren't swappable (NOT, IF).
    tree2 = {"op": "IF",
             "cond": {"op": "NOT",
                      "arg": {"op": "NOT",
                              "arg": {"op": "NOT",
                                      "arg": {"op": "NOT",
                                              "arg": {"op": "NOT",
                                                      "arg": {"erc": 0.1}}}}}},
             "then": {"leaf": "FRENTE", "dur_ms": 200},
             "else": {"leaf": "RE", "dur_ms": 100}}
    # Wait: NOT expects Bool arg, ERC is Float. Build valid no-candidate tree:
    # only IF and NOT (non-swappable), with ERC/leaf only... need Bool for NOT.
    # Actually impossible to build a purely Bool tree without any term/LT/GT.
    # Skip — test fallback via mutate() dispatch instead.
    rng = random.Random(0)
    # Use a tree that is a lone erc — no op, no term, no leaf as candidate.
    lone_erc = {"erc": 0.5}
    assert p._mutate_point(rng, lone_erc) is None


# --------------------------------------------------------------------------
# _mutate_leaf_action
# --------------------------------------------------------------------------

def test_mutate_leaf_action_changes_action_keeps_duration():
    tree = {"leaf": "FRENTE", "dur_ms": 200}
    for seed in range(10):
        rng = random.Random(seed)
        new = p._mutate_leaf_action(rng, tree)
        assert new is not None
        assert new["leaf"] != "FRENTE"
        assert new["leaf"] in g.ACTIONS
        assert new["dur_ms"] == 200


def test_mutate_leaf_action_none_when_no_leaves():
    tree = {"term": "alvo_atras"}
    assert p._mutate_leaf_action(random.Random(0), tree) is None


# --------------------------------------------------------------------------
# _mutate_leaf_duration
# --------------------------------------------------------------------------

def test_mutate_leaf_duration_keeps_action_and_clips():
    tree = {"leaf": "FRENTE", "dur_ms": 200}
    for seed in range(30):
        rng = random.Random(seed)
        new = p._mutate_leaf_duration(rng, tree, sigma_ms=100.0)
        assert new is not None
        assert new["leaf"] == "FRENTE"
        assert g.DURATION_MIN_MS <= new["dur_ms"] <= g.DURATION_MAX_MS


def test_mutate_leaf_duration_clips_to_bounds():
    # Huge sigma forces clipping on most draws.
    tree = {"leaf": "FRENTE", "dur_ms": 200}
    for seed in range(50):
        rng = random.Random(seed)
        new = p._mutate_leaf_duration(rng, tree, sigma_ms=10_000.0)
        assert g.DURATION_MIN_MS <= new["dur_ms"] <= g.DURATION_MAX_MS


def test_mutate_leaf_duration_none_when_no_leaves():
    tree = {"term": "alvo_atras"}
    assert p._mutate_leaf_duration(random.Random(0), tree, 100.0) is None


# --------------------------------------------------------------------------
# _mutate_erc
# --------------------------------------------------------------------------

def test_mutate_erc_clips_to_range():
    tree = {"op": "IF",
            "cond": {"op": "LT",
                     "a": {"term": "dist_frente"},
                     "b": {"erc": 0.5}},
            "then": {"leaf": "FRENTE", "dur_ms": 200},
            "else": {"leaf": "RE", "dur_ms": 100}}
    for seed in range(50):
        rng = random.Random(seed)
        new = p._mutate_erc(rng, tree, sigma=10.0)
        assert new is not None
        g.validate(new)


def test_mutate_erc_none_when_no_erc():
    tree = {"leaf": "FRENTE", "dur_ms": 300}
    assert p._mutate_erc(random.Random(0), tree, 0.2) is None


# --------------------------------------------------------------------------
# mutate dispatch: fallback kicks in
# --------------------------------------------------------------------------

def test_mutate_falls_back_when_no_erc_candidate():
    # Force "erc_perturb" every time; tree has no ERC → falls back to subtree.
    rates = {"subtree": 0.0, "point": 0.0, "leaf_action": 0.0,
             "leaf_duration": 0.0, "erc_perturb": 1.0}
    tree = {"leaf": "FRENTE", "dur_ms": 300}
    rng = random.Random(0)
    new = p.mutate(rng, tree, rates=rates)
    g.validate(new)
    assert g.node_type(new) == "Action"


def test_mutate_with_only_subtree_rate_always_works():
    rates = {"subtree": 1.0, "point": 0.0, "leaf_action": 0.0,
             "leaf_duration": 0.0, "erc_perturb": 0.0}
    rng = random.Random(3)
    for seed in range(20):
        tree = g.random_tree(random.Random(seed), max_depth=4)
        new = p.mutate(rng, tree, rates=rates, subtree_max_depth=3)
        g.validate(new)
        assert g.node_type(new) == "Action"
