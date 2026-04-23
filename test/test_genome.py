"""Unit tests for evolution.genome — the typed GP tree."""
import random

import pytest

from evolutionary_controller_ros.evolution import genome as g


# --------------------------------------------------------------------------
# node_type
# --------------------------------------------------------------------------

def test_node_type_for_each_kind():
    assert g.node_type({"term": "alvo_proximo"}) == "Bool"
    assert g.node_type({"term": "dist_frente"}) == "Float"
    assert g.node_type({"erc": 0.1}) == "Float"
    assert g.node_type({"leaf": "FRENTE", "dur_ms": 500}) == "Action"
    assert g.node_type({"op": "AND",
                        "a": {"term": "obstaculo_frente"},
                        "b": {"term": "obstaculo_esq"}}) == "Bool"
    assert g.node_type({"op": "IF",
                        "cond": {"term": "alvo_atras"},
                        "then": {"leaf": "FRENTE", "dur_ms": 500},
                        "else": {"leaf": "RE", "dur_ms": 200}}) == "Action"


def test_node_type_rejects_unknown_terminal():
    with pytest.raises(ValueError):
        g.node_type({"term": "sensor_inexistente"})


# --------------------------------------------------------------------------
# size / depth
# --------------------------------------------------------------------------

def test_size_and_depth_of_leaf():
    assert g.size({"leaf": "RE", "dur_ms": 100}) == 1
    assert g.depth({"leaf": "RE", "dur_ms": 100}) == 1


def test_size_and_depth_of_if_tree():
    tree = {"op": "IF",
            "cond": {"term": "alvo_atras"},
            "then": {"leaf": "FRENTE", "dur_ms": 500},
            "else": {"leaf": "RE", "dur_ms": 200}}
    assert g.size(tree) == 4
    assert g.depth(tree) == 2


# --------------------------------------------------------------------------
# iter_subtrees / get_at / set_at
# --------------------------------------------------------------------------

def test_iter_subtrees_yields_all_with_paths():
    tree = {"op": "IF",
            "cond": {"term": "alvo_atras"},
            "then": {"leaf": "FRENTE", "dur_ms": 500},
            "else": {"leaf": "RE", "dur_ms": 200}}
    items = list(g.iter_subtrees(tree))
    paths = [p for p, _, _ in items]
    types = [t for _, _, t in items]
    assert paths == [(), ("cond",), ("then",), ("else",)]
    assert types == ["Action", "Bool", "Action", "Action"]


def test_get_at_round_trips_paths():
    tree = {"op": "IF",
            "cond": {"term": "alvo_atras"},
            "then": {"leaf": "FRENTE", "dur_ms": 500},
            "else": {"leaf": "RE", "dur_ms": 200}}
    assert g.get_at(tree, ()) is tree
    assert g.get_at(tree, ("cond",))["term"] == "alvo_atras"
    assert g.get_at(tree, ("then",))["leaf"] == "FRENTE"


def test_set_at_does_not_mutate_original():
    tree = {"op": "IF",
            "cond": {"term": "alvo_atras"},
            "then": {"leaf": "FRENTE", "dur_ms": 500},
            "else": {"leaf": "RE", "dur_ms": 200}}
    new = g.set_at(tree, ("then",), {"leaf": "GIRA_ESQ", "dur_ms": 300})
    assert new["then"]["leaf"] == "GIRA_ESQ"
    assert tree["then"]["leaf"] == "FRENTE"


def test_set_at_root_returns_new_subtree():
    tree = {"leaf": "FRENTE", "dur_ms": 500}
    new = g.set_at(tree, (), {"leaf": "RE", "dur_ms": 100})
    assert new == {"leaf": "RE", "dur_ms": 100}


# --------------------------------------------------------------------------
# random_tree
# --------------------------------------------------------------------------

def test_random_tree_produces_valid_action_trees():
    rng = random.Random(42)
    for _ in range(50):
        tree = g.random_tree(rng, max_depth=5)
        g.validate(tree)
        assert g.node_type(tree) == "Action"
        assert g.depth(tree) <= 5


def test_random_tree_respects_each_return_type():
    rng = random.Random(42)
    for t in ("Bool", "Float", "Action"):
        for _ in range(10):
            tree = g.random_tree(rng, max_depth=4, return_type=t)
            assert g.node_type(tree) == t
            g.validate(tree)


def test_random_tree_at_depth_1_is_always_terminal():
    rng = random.Random(0)
    for _ in range(20):
        tree = g.random_tree(rng, max_depth=1)
        assert "op" not in tree


def test_random_tree_is_deterministic_with_seed():
    a = g.random_tree(random.Random(7), max_depth=5)
    b = g.random_tree(random.Random(7), max_depth=5)
    assert a == b


def test_random_tree_rejects_zero_depth():
    with pytest.raises(ValueError):
        g.random_tree(random.Random(0), max_depth=0)


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def test_validate_rejects_unknown_terminal():
    with pytest.raises(ValueError):
        g.validate({"term": "sensor_inexistente"})


def test_validate_rejects_type_mismatch_in_op_args():
    # LT expects two Floats — giving a Bool should fail.
    with pytest.raises(ValueError):
        g.validate({"op": "LT",
                    "a": {"term": "alvo_atras"},
                    "b": {"term": "dist_frente"}})


def test_validate_rejects_missing_op_children():
    with pytest.raises(ValueError):
        g.validate({"op": "IF",
                    "cond": {"term": "alvo_atras"},
                    "then": {"leaf": "FRENTE", "dur_ms": 500}})
        # missing "else"


def test_validate_rejects_duration_below_minimum():
    with pytest.raises(ValueError):
        g.validate({"leaf": "FRENTE", "dur_ms": 10})


def test_validate_rejects_duration_above_maximum():
    with pytest.raises(ValueError):
        g.validate({"leaf": "FRENTE", "dur_ms": 5000})


def test_validate_rejects_erc_out_of_range():
    with pytest.raises(ValueError):
        g.validate({"erc": 2.5})


def test_validate_rejects_unknown_action_name():
    with pytest.raises(ValueError):
        g.validate({"leaf": "VOAR", "dur_ms": 300})


def test_validate_accepts_well_formed_tree():
    g.validate({"op": "IF",
                "cond": {"op": "LT",
                         "a": {"term": "dist_frente"},
                         "b": {"erc": 0.3}},
                "then": {"leaf": "GIRA_ESQ", "dur_ms": 200},
                "else": {"leaf": "FRENTE", "dur_ms": 400}})


# --------------------------------------------------------------------------
# JSON round-trip
# --------------------------------------------------------------------------

def test_json_roundtrip_preserves_tree():
    rng = random.Random(7)
    original = g.random_tree(rng, max_depth=5)
    s = g.to_json(original)
    restored = g.from_json(s)
    assert restored == original


def test_from_json_validates():
    bad = '{"term": "sensor_inexistente"}'
    with pytest.raises(ValueError):
        g.from_json(bad)


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------

def test_evaluate_leaf_returns_action_and_duration():
    assert g.evaluate({"leaf": "FRENTE", "dur_ms": 500}, {}) == ("FRENTE", 500)


def test_evaluate_if_picks_then_vs_else_by_condition():
    tree = {"op": "IF",
            "cond": {"term": "alvo_atras"},
            "then": {"leaf": "RE", "dur_ms": 300},
            "else": {"leaf": "FRENTE", "dur_ms": 200}}
    assert g.evaluate(tree, {"alvo_atras": True}) == ("RE", 300)
    assert g.evaluate(tree, {"alvo_atras": False}) == ("FRENTE", 200)


def test_evaluate_lt_against_erc():
    tree = {"op": "IF",
            "cond": {"op": "LT",
                     "a": {"term": "dist_frente"},
                     "b": {"erc": 0.3}},
            "then": {"leaf": "GIRA_ESQ", "dur_ms": 200},
            "else": {"leaf": "FRENTE", "dur_ms": 400}}
    assert g.evaluate(tree, {"dist_frente": 0.1}) == ("GIRA_ESQ", 200)
    assert g.evaluate(tree, {"dist_frente": 0.5}) == ("FRENTE", 400)


def test_evaluate_composed_and_or_not():
    tree = {"op": "IF",
            "cond": {"op": "AND",
                     "a": {"term": "alvo_proximo"},
                     "b": {"op": "NOT",
                           "arg": {"term": "alvo_atras"}}},
            "then": {"leaf": "FRENTE", "dur_ms": 300},
            "else": {"leaf": "RE", "dur_ms": 100}}
    assert g.evaluate(tree,
                      {"alvo_proximo": True,
                       "alvo_atras": False}) == ("FRENTE", 300)
    assert g.evaluate(tree,
                      {"alvo_proximo": True,
                       "alvo_atras": True}) == ("RE", 100)
