"""Smoke test: exercise the entire evolutionary pipeline without ROS.

Runs four stages:
    1) Build a population of 10 trees and report shape stats.
    2) Exercise each of the 5 mutation operators on a fixed seed tree.
    3) Exercise crossover on two fixed trees.
    4) Run a 3-generation GA with a deterministic fake evaluator and
       show that best/avg fitness improves across generations.

No ROS, no Gazebo — this is the "is the pure-Python part coherent?"
check before we ever touch the simulator.

Usage:
    python3 scripts/smoke_pipeline.py
"""
import random
import sys
from pathlib import Path

# Allow running from the package root without colcon install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evolutionary_controller_ros.evolution import (  # noqa: E402
    genome as g,
    population as pop,
    algorithm as alg,
)


# ==========================================================================
# Stage 1 — population init
# ==========================================================================

def stage1_population():
    print("\n=== STAGE 1: init_population(n=10, max_depth=5) ===")
    rng = random.Random(7)
    trees = pop.init_population(rng, size=10, max_depth=5)

    sizes = [g.size(t) for t in trees]
    depths = [g.depth(t) for t in trees]
    print(f"size:  min={min(sizes)}  avg={sum(sizes)/len(sizes):.1f}  max={max(sizes)}")
    print(f"depth: min={min(depths)}  avg={sum(depths)/len(depths):.1f}  max={max(depths)}")

    # Every tree must be a valid Action-rooted tree.
    for i, t in enumerate(trees):
        g.validate(t)                  # raises on error
        assert g.node_type(t) == "Action", f"tree {i} is not Action-rooted"
    print("all 10 trees validate and return Action.")

    return trees


# ==========================================================================
# Stage 2 — each mutation operator in isolation
# ==========================================================================

def _fixed_seed_tree(rng):
    # Keep growing until we get a tree rich enough to show off every
    # mutation (has an op, a leaf, and an ERC).
    for _ in range(500):
        t = g.random_tree(rng, max_depth=4, return_type="Action")
        has_op   = any("op"   in n for _p, n, _t in g.iter_subtrees(t))
        has_erc  = any("erc"  in n for _p, n, _t in g.iter_subtrees(t))
        has_leaf = any("leaf" in n for _p, n, _t in g.iter_subtrees(t))
        if has_op and has_erc and has_leaf:
            return t
    raise RuntimeError("could not grow a tree with op+erc+leaf within 500 tries")


def stage2_mutations():
    print("\n=== STAGE 2: each mutation operator ===")
    rng = random.Random(11)
    parent = _fixed_seed_tree(rng)
    print(f"parent: size={g.size(parent)} depth={g.depth(parent)}")

    ops = [
        ("subtree",       {"subtree": 1.0}),
        ("point",         {"point": 1.0}),
        ("leaf_action",   {"leaf_action": 1.0}),
        ("leaf_duration", {"leaf_duration": 1.0}),
        ("erc_perturb",   {"erc_perturb": 1.0}),
    ]
    for name, rates in ops:
        mrng = random.Random(42)
        child = pop.mutate(mrng, parent, rates=rates)
        g.validate(child)
        ok_sizes = f"{g.size(parent)}→{g.size(child)}"
        different = g.to_json(child) != g.to_json(parent)
        print(f"  mutate[{name:<14}] size {ok_sizes:>6}  "
              f"changed={different}  validated")


# ==========================================================================
# Stage 3 — crossover
# ==========================================================================

def stage3_crossover():
    print("\n=== STAGE 3: crossover ===")
    rng = random.Random(3)
    p1 = g.random_tree(rng, max_depth=4, return_type="Action")
    p2 = g.random_tree(rng, max_depth=4, return_type="Action")
    print(f"p1 size={g.size(p1)}  p2 size={g.size(p2)}")

    swaps = 0
    for trial in range(5):
        crng = random.Random(100 + trial)
        c1, c2 = pop.crossover(crng, p1, p2)
        g.validate(c1)
        g.validate(c2)
        if g.to_json(c1) != g.to_json(p1) or g.to_json(c2) != g.to_json(p2):
            swaps += 1
    print(f"type-respecting swaps in 5 trials: {swaps}/5")

    # Parents must be untouched (immutability guarantee).
    orig_p1 = g.to_json(p1)
    pop.crossover(random.Random(0), p1, p2)
    assert g.to_json(p1) == orig_p1, "crossover mutated parent!"
    print("parents unchanged after crossover.")


# ==========================================================================
# Stage 4 — run_ga with a fake evaluator
# ==========================================================================

def _fake_evaluator_factory():
    """Evaluator that rewards trees emitting FRENTE on a dummy feature set.

    Returns the fitness vector [action_is_frente, -size] so lexicase has
    a real trade-off: 'picks FRENTE' vs 'small tree'. Over generations,
    both should rise (especially after elitism).
    """
    # A tiny set of feature vectors the evaluator queries the tree against.
    feature_samples = [
        # Baseline — nothing obstructing, flag invisible.
        {k: 0.0 for k in _ALL_FLOAT_TERMS},
        # Flag centered right in front — should reward FRENTE.
        {**{k: 0.0 for k in _ALL_FLOAT_TERMS},
         "angulo_bandeira_inimiga": 0.0},
    ]
    # Fill bool defaults to False so evaluate() doesn't blow up.
    for s in feature_samples:
        for b in _ALL_BOOL_TERMS:
            s[b] = False
        s["bandeira_inimiga_visivel"] = True
        s["bandeira_centralizada"] = True

    def evaluator(tree):
        frente_hits = 0
        for feats in feature_samples:
            action, _dur = g.evaluate(tree, feats)
            if action == "FRENTE":
                frente_hits += 1
        return [float(frente_hits), float(-g.size(tree))]

    return evaluator


_ALL_FLOAT_TERMS = [
    "dist_frente", "dist_esq", "dist_dir", "dist_atras",
    "angulo_bandeira_inimiga", "dist_base_propria", "angulo_base_propria",
    "dist_zona_deploy", "angulo_zona_deploy",
    "velocidade_linear", "velocidade_angular",
]
_ALL_BOOL_TERMS = [
    "obstaculo_frente", "obstaculo_esq", "obstaculo_dir",
    "bandeira_inimiga_visivel",
    "bandeira_esquerda", "bandeira_direita", "bandeira_centralizada",
    "segurando_bandeira", "na_base_propria", "na_zona_deploy",
    "base_inimiga_visivel",
]


def stage4_ga():
    print("\n=== STAGE 4: run_ga(pop=10, gens=3, elite_k=1) ===")
    rng = random.Random(19)
    evaluator = _fake_evaluator_factory()

    history = []

    def on_gen(gen, population, case_matrix, best_idx):
        # case_matrix[i] = [frente_hits, -size, ...parsimony?]
        col_frente = [row[0] for row in case_matrix]
        col_neg_size = [row[1] for row in case_matrix]
        best_frente = max(col_frente)
        avg_frente = sum(col_frente) / len(col_frente)
        avg_size = -sum(col_neg_size) / len(col_neg_size)
        history.append((gen, best_frente, avg_frente, avg_size))
        print(f"  gen {gen}: best_frente={best_frente:.1f}  "
              f"avg_frente={avg_frente:.2f}  "
              f"avg_size={avg_size:.2f}  "
              f"best_size={g.size(population[best_idx])}")

    champion = alg.run_ga(
        rng,
        evaluator=evaluator,
        pop_size=10,
        n_generations=3,
        init_max_depth=4,
        crossover_rate=0.9,
        elite_k=1,
        on_generation=on_gen,
    )
    g.validate(champion)
    print(f"champion: size={g.size(champion)} depth={g.depth(champion)}")

    # Monotonicity-ish check: best_frente in last generation ≥ first.
    first_best = history[0][1]
    last_best = history[-1][1]
    assert last_best >= first_best, (
        f"best_frente regressed: gen0={first_best} gen{len(history)-1}={last_best}")
    print(f"best_frente: gen0={first_best} → gen{len(history)-1}={last_best}  (non-regression OK)")


# ==========================================================================
# Main
# ==========================================================================

def main():
    print("smoke_pipeline: exercising evolution/ stack without ROS")
    stage1_population()
    stage2_mutations()
    stage3_crossover()
    stage4_ga()
    print("\nALL STAGES OK")


if __name__ == "__main__":
    main()
