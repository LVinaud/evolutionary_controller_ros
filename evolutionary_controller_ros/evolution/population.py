"""Genetic operators over typed GP trees (crossover, mutations, population init).

All operators are pure: they never mutate their input trees, always returning
new trees. This is safe because genome.set_at only copies the spine along the
modified path, leaving untouched siblings shared by reference.

Mutations
    subtree        : replace a random subtree with a fresh random tree of the
                     same return type (depth budget `subtree_max_depth`).
    point          : swap a single operator or a single named terminal for
                     another of the same type. Candidates are binary ops
                     (LT<->GT) and named terminals. No-op if the tree has
                     no candidate node.
    leaf_action    : pick a random Action leaf; change its action name to a
                     different one; keep duration_ms.
    leaf_duration  : pick a random Action leaf; perturb duration_ms by
                     Gaussian(sigma); clip to [DURATION_MIN_MS, DURATION_MAX_MS].
    erc_perturb    : pick a random ERC node; perturb value by Gaussian(sigma);
                     clip to ERC_RANGE.

If the sampled mutation type has no applicable candidates, the operator falls
back to subtree mutation (which always applies).
"""
import random

from . import genome as g


DEFAULT_MUTATION_RATES = {
    "subtree":       0.30,
    "point":         0.20,
    "leaf_action":   0.20,
    "leaf_duration": 0.20,
    "erc_perturb":   0.10,
}

# Binary ops that can be point-swapped with each other. OR was dropped in
# the go-to-goal refactor, so AND no longer has a same-arity same-type
# partner to swap with — only LT<->GT remain.
_POINT_OP_SWAPS = {
    "LT":  "GT", "GT": "LT",
}


# ==========================================================================
# Population initialization
# ==========================================================================

def init_population(
    rng: random.Random,
    size: int,
    max_depth: int,
    *,
    op_prob: float = 0.5,
    erc_prob: float = 0.3,
) -> list:
    """Return `size` random Action-rooted trees using grow initialization."""
    return [g.random_tree(rng, max_depth, "Action",
                          op_prob=op_prob, erc_prob=erc_prob)
            for _ in range(size)]


# ==========================================================================
# Crossover (subtree swap with type matching)
# ==========================================================================

def crossover(
    rng: random.Random,
    parent_a: dict,
    parent_b: dict,
    *,
    max_tries: int = 10,
) -> tuple:
    """Exchange compatible subtrees between two parents; return (child_a, child_b).

    Picks a random subtree in A and a same-type random subtree in B, swaps.
    If no compatible pair is found within `max_tries`, returns the parents
    unchanged (rare, but possible for degenerate trees).
    """
    subs_b_by_type: dict = {}
    for p, s, t in g.iter_subtrees(parent_b):
        subs_b_by_type.setdefault(t, []).append((p, s))

    subs_a = list(g.iter_subtrees(parent_a))
    for _ in range(max_tries):
        path_a, _sub_a, type_a = rng.choice(subs_a)
        candidates_b = subs_b_by_type.get(type_a, [])
        if not candidates_b:
            continue
        path_b, sub_b = rng.choice(candidates_b)
        sub_a = g.get_at(parent_a, path_a)
        child_a = g.set_at(parent_a, path_a, sub_b)
        child_b = g.set_at(parent_b, path_b, sub_a)
        return child_a, child_b
    return parent_a, parent_b


# ==========================================================================
# Mutation — top-level dispatch
# ==========================================================================

def mutate(
    rng: random.Random,
    tree: dict,
    rates: dict | None = None,
    *,
    subtree_max_depth: int = 3,
    duration_sigma_ms: float = 100.0,
    erc_sigma: float = 0.2,
) -> dict:
    """Apply exactly one mutation; return a new tree."""
    rates = rates or DEFAULT_MUTATION_RATES
    kind = _sample_weighted(rng, rates)

    if kind == "point":
        result = _mutate_point(rng, tree)
        if result is not None:
            return result
        kind = "subtree"  # fallback
    if kind == "leaf_action":
        result = _mutate_leaf_action(rng, tree)
        if result is not None:
            return result
        kind = "subtree"
    if kind == "leaf_duration":
        result = _mutate_leaf_duration(rng, tree, duration_sigma_ms)
        if result is not None:
            return result
        kind = "subtree"
    if kind == "erc_perturb":
        result = _mutate_erc(rng, tree, erc_sigma)
        if result is not None:
            return result
        kind = "subtree"
    return _mutate_subtree(rng, tree, subtree_max_depth)


def _sample_weighted(rng: random.Random, weights: dict) -> str:
    total = sum(weights.values())
    r = rng.random() * total
    acc = 0.0
    for name, w in weights.items():
        acc += w
        if r < acc:
            return name
    return name  # floating-point fallback


# ==========================================================================
# Individual mutation operators
# ==========================================================================

def _mutate_subtree(rng, tree, subtree_max_depth):
    path, _sub, t = rng.choice(list(g.iter_subtrees(tree)))
    new_sub = g.random_tree(rng, subtree_max_depth, return_type=t)
    return g.set_at(tree, path, new_sub)


def _mutate_point(rng, tree):
    candidates = []
    for path, sub, _t in g.iter_subtrees(tree):
        if "op" in sub and sub["op"] in _POINT_OP_SWAPS:
            candidates.append((path, sub, "op"))
        elif "term" in sub:
            candidates.append((path, sub, "term"))
    if not candidates:
        return None

    path, sub, kind = rng.choice(candidates)
    if kind == "op":
        new_sub = {**sub, "op": _POINT_OP_SWAPS[sub["op"]]}
    else:
        pool = (g.BOOL_TERMINALS if sub["term"] in g.BOOL_TERMINALS
                else g.FLOAT_TERMINALS)
        choices = [n for n in pool if n != sub["term"]]
        new_sub = {"term": rng.choice(choices)}
    return g.set_at(tree, path, new_sub)


def _mutate_leaf_action(rng, tree):
    leaves = [(p, s) for p, s, _ in g.iter_subtrees(tree) if "leaf" in s]
    if not leaves:
        return None
    path, leaf = rng.choice(leaves)
    choices = [a for a in g.ACTIONS if a != leaf["leaf"]]
    new_leaf = {"leaf": rng.choice(choices), "dur_ms": leaf["dur_ms"]}
    return g.set_at(tree, path, new_leaf)


def _mutate_leaf_duration(rng, tree, sigma_ms):
    leaves = [(p, s) for p, s, _ in g.iter_subtrees(tree) if "leaf" in s]
    if not leaves:
        return None
    path, leaf = rng.choice(leaves)
    perturbed = leaf["dur_ms"] + int(round(rng.gauss(0.0, sigma_ms)))
    clipped = max(g.DURATION_MIN_MS, min(g.DURATION_MAX_MS, perturbed))
    new_leaf = {"leaf": leaf["leaf"], "dur_ms": clipped}
    return g.set_at(tree, path, new_leaf)


def _mutate_erc(rng, tree, sigma):
    ercs = [(p, s) for p, s, _ in g.iter_subtrees(tree) if "erc" in s]
    if not ercs:
        return None
    path, erc = rng.choice(ercs)
    perturbed = erc["erc"] + rng.gauss(0.0, sigma)
    lo, hi = g.ERC_RANGE
    clipped = max(lo, min(hi, perturbed))
    return g.set_at(tree, path, {"erc": clipped})
