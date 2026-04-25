"""Typed-tree genome for Genetic Programming.

A genome is a dict representing a strongly-typed decision tree. Types are:

    "Action" : root of a genome; either an IF node or a leaf (action, duration_ms).
    "Bool"   : logical conditions and comparisons.
    "Float"  : sensor readings (normalized) or ERCs (evolved constants in [-1, 1]).

Function set
    IF(Bool, Action, Action)  -> Action
    AND(Bool, Bool)           -> Bool
    NOT(Bool)                 -> Bool
    LT (Float, Float)         -> Bool    # a < b
    GT (Float, Float)         -> Bool    # a > b

    (OR removed — redundant with AND+NOT, smaller search space.)

Terminal set (go-to-goal reactive controller — target pose is injected)
    Bool  : 8 pre-cooked sensor booleans (thresholds live in utils/sensors.py).
    Float : 8 normalized sensor floats + ERC (Ephemeral Random Constant).
    Action: leaf (action_name, duration_ms in [50, 1000]).

Node dict formats
    {"op": "AND", "a": ..., "b": ...}
    {"op": "NOT", "arg": ...}
    {"op": "LT",  "a": ..., "b": ...}
    {"op": "GT",  "a": ..., "b": ...}
    {"op": "IF",  "cond": ..., "then": ..., "else": ...}
    {"term": "<name>"}
    {"erc": 0.37}
    {"leaf": "FRENTE", "dur_ms": 500}

Inspection, random generation, serialization and evaluation live here.
Genetic operators (crossover, mutation) live in evolution/population.py;
selection in evolution/algorithm.py.

History: the CTF-specific terminals (bandeira_*, na_base_propria, etc.)
and the PEGAR/SOLTAR/PARAR actions were removed in the go-to-goal refactor
— the GP now evolves pure reactive navigation, and the flag-grab / drop
are driven by a hardcoded state machine in gp_controller's "ctf" mode.
"""
import json
import random
from typing import Iterator


ACTIONS = (
    "FRENTE", "RE", "GIRA_ESQ", "GIRA_DIR",
)

BOOL_TERMINALS = (
    # Obstacle cones (thresholded on lidar minimum per direction).
    "obstaculo_frente", "obstaculo_esq", "obstaculo_dir",
    # Target-bearing sectors (cones on the relative angle to the target).
    "alvo_frente", "alvo_esq", "alvo_dir", "alvo_atras",
    # Target proximity (distance < alvo_prox_radius from config).
    "alvo_proximo",
)

FLOAT_TERMINALS = (
    # Lidar cone distances (normalized, clipped).
    "dist_frente", "dist_esq", "dist_dir", "dist_atras",
    # Target (injected via params — generic goal, not flag/base specific).
    "dist_alvo", "angulo_alvo",
    # Odometry-reported velocities (drift-free proprioception).
    "velocidade_linear", "velocidade_angular",
)

OPS_SPEC = {
    "IF":  (("Bool", "Action", "Action"), "Action"),
    "AND": (("Bool", "Bool"), "Bool"),
    "NOT": (("Bool",), "Bool"),
    "LT":  (("Float", "Float"), "Bool"),
    "GT":  (("Float", "Float"), "Bool"),
}

OP_CHILD_KEYS = {
    "IF":  ("cond", "then", "else"),
    "AND": ("a", "b"),
    "NOT": ("arg",),
    "LT":  ("a", "b"),
    "GT":  ("a", "b"),
}

ERC_RANGE = (-1.0, 1.0)
DURATION_MIN_MS = 50
# Upper bound chosen so a single FRENTE action at v=0.4 m/s covers at most
# ~12cm of blind motion — smaller than reach_radius_m (0.5m), so a tree
# that homes in on the target doesn't overshoot past it between decisions.
DURATION_MAX_MS = 300


# ==========================================================================
# Node inspection
# ==========================================================================

def node_type(node: dict) -> str:
    if "op" in node:
        return OPS_SPEC[node["op"]][1]
    if "term" in node:
        name = node["term"]
        if name in BOOL_TERMINALS:
            return "Bool"
        if name in FLOAT_TERMINALS:
            return "Float"
        raise ValueError(f"unknown terminal name: {name!r}")
    if "erc" in node:
        return "Float"
    if "leaf" in node:
        return "Action"
    raise ValueError(f"malformed node (no op/term/erc/leaf): {node!r}")


def size(node: dict) -> int:
    if "op" in node:
        return 1 + sum(size(node[k]) for k in OP_CHILD_KEYS[node["op"]])
    return 1


def depth(node: dict) -> int:
    if "op" in node:
        return 1 + max(depth(node[k]) for k in OP_CHILD_KEYS[node["op"]])
    return 1


def iter_subtrees(node: dict, _path: tuple = ()) -> Iterator[tuple]:
    """Yield (path, subtree, type) for every subtree including the root.

    Path is a tuple of keys; () is the root. Use get_at / set_at to index.
    """
    yield (_path, node, node_type(node))
    if "op" in node:
        for k in OP_CHILD_KEYS[node["op"]]:
            yield from iter_subtrees(node[k], _path + (k,))


def get_at(root: dict, path: tuple) -> dict:
    node = root
    for k in path:
        node = node[k]
    return node


def set_at(root: dict, path: tuple, new_subtree: dict) -> dict:
    """Return a new tree with the subtree at `path` replaced.

    Does not mutate `root`. Copies only the spine along `path`; untouched
    siblings are shared by reference (safe because genomes are not mutated
    in place — operators always build new trees).
    """
    if not path:
        return new_subtree
    out_root = {**root}
    cur = out_root
    for k in path[:-1]:
        cur[k] = {**cur[k]}
        cur = cur[k]
    cur[path[-1]] = new_subtree
    return out_root


# ==========================================================================
# Random generation (grow initialization)
# ==========================================================================

def random_tree(
    rng: random.Random,
    max_depth: int,
    return_type: str = "Action",
    *,
    op_prob: float = 0.5,
    erc_prob: float = 0.3,
) -> dict:
    """Generate a random tree with the given return type and depth bound.

    Grow strategy: at each non-leaf depth, with probability `op_prob` emit
    an operator (if any exists for the type); otherwise emit a terminal.
    At `max_depth` the node is forced to be a terminal. `erc_prob` controls
    how often Float terminals become ERCs instead of named sensors.
    """
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    return _grow(rng, return_type, max_depth - 1, op_prob, erc_prob)


def _grow(rng, t: str, budget: int, op_prob: float, erc_prob: float) -> dict:
    if t == "Float":
        return _random_float_terminal(rng, erc_prob)
    if budget <= 0:
        return _random_terminal(rng, t, erc_prob)

    ops_for_type = [op for op, (_, ret) in OPS_SPEC.items() if ret == t]
    if not ops_for_type or rng.random() >= op_prob:
        return _random_terminal(rng, t, erc_prob)

    op = rng.choice(ops_for_type)
    arg_types, _ = OPS_SPEC[op]
    keys = OP_CHILD_KEYS[op]
    out = {"op": op}
    for k, at in zip(keys, arg_types):
        out[k] = _grow(rng, at, budget - 1, op_prob, erc_prob)
    return out


def _random_terminal(rng, t: str, erc_prob: float) -> dict:
    if t == "Bool":
        return {"term": rng.choice(BOOL_TERMINALS)}
    if t == "Float":
        return _random_float_terminal(rng, erc_prob)
    if t == "Action":
        return {
            "leaf": rng.choice(ACTIONS),
            "dur_ms": rng.randint(DURATION_MIN_MS, DURATION_MAX_MS),
        }
    raise ValueError(f"unknown type: {t!r}")


def _random_float_terminal(rng, erc_prob: float) -> dict:
    if rng.random() < erc_prob:
        lo, hi = ERC_RANGE
        return {"erc": rng.uniform(lo, hi)}
    return {"term": rng.choice(FLOAT_TERMINALS)}


# ==========================================================================
# Validation
# ==========================================================================

def validate(node: dict) -> None:
    """Raise ValueError if `node` is not a well-formed genome tree."""
    _validate(node, expected_type=None)


def _validate(node, expected_type):
    if not isinstance(node, dict):
        raise ValueError(f"node must be a dict, got {type(node).__name__}")

    kinds = [k for k in ("op", "term", "erc", "leaf") if k in node]
    if len(kinds) != 1:
        raise ValueError(f"node must have exactly one of op/term/erc/leaf: {node!r}")

    t = node_type(node)
    if expected_type is not None and t != expected_type:
        raise ValueError(f"type mismatch: expected {expected_type}, got {t} in {node!r}")

    keys = set(node.keys())
    if "op" in node:
        op = node["op"]
        if op not in OPS_SPEC:
            raise ValueError(f"unknown op: {op!r}")
        expected_keys = set(OP_CHILD_KEYS[op]) | {"op"}
        if keys != expected_keys:
            raise ValueError(f"op {op} expects keys {expected_keys}, got {keys} in {node!r}")
        arg_types, _ = OPS_SPEC[op]
        for k, at in zip(OP_CHILD_KEYS[op], arg_types):
            _validate(node[k], at)
    elif "term" in node:
        if keys != {"term"}:
            raise ValueError(f"terminal must have only 'term' key: {node!r}")
        name = node["term"]
        if name not in BOOL_TERMINALS and name not in FLOAT_TERMINALS:
            raise ValueError(f"unknown terminal: {name!r}")
    elif "erc" in node:
        if keys != {"erc"}:
            raise ValueError(f"erc must have only 'erc' key: {node!r}")
        v = node["erc"]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"erc value must be a number: {node!r}")
        lo, hi = ERC_RANGE
        if not (lo <= v <= hi):
            raise ValueError(f"erc out of range {ERC_RANGE}: {v}")
    else:  # leaf
        if keys != {"leaf", "dur_ms"}:
            raise ValueError(f"leaf must have keys {{'leaf','dur_ms'}}: {node!r}")
        if node["leaf"] not in ACTIONS:
            raise ValueError(f"unknown action: {node['leaf']!r}")
        d = node["dur_ms"]
        if isinstance(d, bool) or not isinstance(d, int):
            raise ValueError(f"dur_ms must be int: {node!r}")
        if not (DURATION_MIN_MS <= d <= DURATION_MAX_MS):
            raise ValueError(f"dur_ms out of [{DURATION_MIN_MS},{DURATION_MAX_MS}]: {d}")


# ==========================================================================
# Serialization
# ==========================================================================

def to_json(node: dict) -> str:
    return json.dumps(node, separators=(",", ":"))


def from_json(s: str) -> dict:
    node = json.loads(s)
    validate(node)
    return node


# ==========================================================================
# Evaluation (used by gp_controller at each control tick)
# ==========================================================================

def evaluate(node: dict, sensors: dict) -> tuple:
    """Evaluate an Action-typed tree; return (action_name, duration_ms).

    `sensors` must map every BOOL_TERMINAL and FLOAT_TERMINAL name appearing
    in the tree to its current value (bool or float). Floats are expected
    to be normalized.
    """
    if "leaf" in node:
        return (node["leaf"], node["dur_ms"])
    if node.get("op") == "IF":
        if _eval_bool(node["cond"], sensors):
            return evaluate(node["then"], sensors)
        return evaluate(node["else"], sensors)
    raise ValueError(f"not an Action node: {node!r}")


def _eval_bool(node: dict, sensors: dict) -> bool:
    if "term" in node:
        return bool(sensors[node["term"]])
    op = node.get("op")
    if op == "AND":
        return _eval_bool(node["a"], sensors) and _eval_bool(node["b"], sensors)
    if op == "OR":
        return _eval_bool(node["a"], sensors) or _eval_bool(node["b"], sensors)
    if op == "NOT":
        return not _eval_bool(node["arg"], sensors)
    if op == "LT":
        return _eval_float(node["a"], sensors) < _eval_float(node["b"], sensors)
    if op == "GT":
        return _eval_float(node["a"], sensors) > _eval_float(node["b"], sensors)
    raise ValueError(f"not a Bool node: {node!r}")


def _eval_float(node: dict, sensors: dict) -> float:
    if "erc" in node:
        return float(node["erc"])
    if "term" in node:
        return float(sensors[node["term"]])
    raise ValueError(f"not a Float node: {node!r}")
