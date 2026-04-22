"""Main GA loop: ε-lexicase selection (with MAD) + Koza-style reproduction.

Selection
    ε-lexicase (La Cava 2016). For each selection, we pick a random ordering
    of the fitness cases and iteratively keep only the individuals whose
    score on the current case is within ε of the best score on that case.
    `ε = median(|x_i − median(x)|)` is recomputed per case from the
    population-level scores (adaptive MAD).

Parsimony
    Tree size is added as one extra "case" (negated so that smaller is
    better under lexicase's "max is best" convention). No α hyperparameter;
    lexicase itself decides when size is a discriminating feature.

Reproduction (Koza-style)
    For each offspring slot not filled by elitism:
        - with prob `crossover_rate`: pick two parents, crossover, keep one
          child (the other is discarded — simpler than bookkeeping both).
        - else: pick one parent, mutate.

Elitism
    The top-K individuals (by mean of their fitness-case vector) are copied
    unchanged into the next generation. K defaults to 1 but is parametric.
"""
import random
import statistics
from typing import Callable

from . import genome as g
from . import population as p


# ==========================================================================
# Selection
# ==========================================================================

def epsilon_lexicase_select(
    rng: random.Random,
    pop: list,
    case_matrix: list,
    *,
    epsilons: list | None = None,
) -> dict:
    """Pick one individual via ε-lexicase.

    `case_matrix[i]` is the fitness-case vector of individual i; all vectors
    must share the same length C. All cases follow "larger is better".
    `epsilons` (length C) can be precomputed; otherwise it is derived from
    `case_matrix` via MAD.
    """
    if not pop:
        raise ValueError("population is empty")
    n_cases = len(case_matrix[0])
    if epsilons is None:
        epsilons = compute_mad_epsilons(case_matrix)

    survivors = list(range(len(pop)))
    case_order = list(range(n_cases))
    rng.shuffle(case_order)

    for c in case_order:
        best = max(case_matrix[i][c] for i in survivors)
        threshold = best - epsilons[c]
        survivors = [i for i in survivors if case_matrix[i][c] >= threshold]
        if len(survivors) == 1:
            break

    return pop[rng.choice(survivors)]


def compute_mad_epsilons(case_matrix: list) -> list:
    """Median absolute deviation, per column, over the full population."""
    n_cases = len(case_matrix[0])
    eps = []
    for c in range(n_cases):
        col = [row[c] for row in case_matrix]
        med = statistics.median(col)
        eps.append(statistics.median(abs(x - med) for x in col))
    return eps


# ==========================================================================
# Elitism
# ==========================================================================

def elite(pop: list, case_matrix: list, k: int = 1) -> list:
    """Return the top-k individuals by mean of their case vector.

    Ties are broken by original index (stable). With k >= len(pop), returns
    the whole population sorted.
    """
    if k <= 0:
        return []
    scores = [(sum(row) / len(row), i) for i, row in enumerate(case_matrix)]
    scores.sort(key=lambda t: (-t[0], t[1]))
    top = scores[:k]
    return [pop[i] for _, i in top]


# ==========================================================================
# Fitness matrix assembly (base cases + parsimony case)
# ==========================================================================

def assemble_case_matrix(
    base_cases: list,
    pop: list,
    *,
    include_parsimony: bool = True,
) -> list:
    """Attach a negated-size parsimony case to each base case vector.

    `base_cases[i]` is the "raw" case vector for individual i from the
    evaluator (all larger-is-better). We append `-size(tree)` so smaller
    trees win ties under lexicase.
    """
    if not include_parsimony:
        return [list(row) for row in base_cases]
    return [list(row) + [-float(g.size(tree))]
            for row, tree in zip(base_cases, pop)]


# ==========================================================================
# Main GA loop
# ==========================================================================

def run_ga(
    rng: random.Random,
    *,
    evaluator: Callable[[dict], list],
    pop_size: int = 20,
    n_generations: int = 30,
    init_max_depth: int = 5,
    init_op_prob: float = 0.5,
    init_erc_prob: float = 0.3,
    crossover_rate: float = 0.9,
    mutation_rates: dict | None = None,
    elite_k: int = 1,
    include_parsimony: bool = True,
    on_generation: Callable | None = None,
) -> dict:
    """Run the GA and return the best individual across all generations.

    `evaluator(tree) -> list[float]` is supplied by the caller (the ROS
    episode runner in production; a pure-Python stub in tests). It returns
    a vector of fitness cases, all "larger is better". Tree size is added
    automatically as an extra case when `include_parsimony=True`.

    `on_generation(gen_idx, pop, case_matrix, best_idx)` is an optional
    callback invoked once per generation after evaluation; useful for
    logging and checkpointing without coupling this function to I/O.
    """
    pop = p.init_population(rng, pop_size, init_max_depth,
                            op_prob=init_op_prob, erc_prob=init_erc_prob)
    best_tree = None
    best_mean = -float("inf")

    for gen in range(n_generations):
        base_cases = [evaluator(tree) for tree in pop]
        case_matrix = assemble_case_matrix(base_cases, pop,
                                           include_parsimony=include_parsimony)

        means = [sum(row) / len(row) for row in case_matrix]
        gen_best_idx = max(range(pop_size), key=lambda i: means[i])
        if means[gen_best_idx] > best_mean:
            best_mean = means[gen_best_idx]
            best_tree = pop[gen_best_idx]

        if on_generation is not None:
            on_generation(gen, pop, case_matrix, gen_best_idx)

        if gen == n_generations - 1:
            break

        pop = _breed_next_generation(
            rng, pop, case_matrix,
            pop_size=pop_size,
            crossover_rate=crossover_rate,
            mutation_rates=mutation_rates,
            elite_k=elite_k,
        )

    return best_tree


def _breed_next_generation(
    rng: random.Random,
    pop: list,
    case_matrix: list,
    *,
    pop_size: int,
    crossover_rate: float,
    mutation_rates: dict | None,
    elite_k: int,
) -> list:
    next_pop = list(elite(pop, case_matrix, k=elite_k))
    eps = compute_mad_epsilons(case_matrix)

    while len(next_pop) < pop_size:
        if rng.random() < crossover_rate:
            a = epsilon_lexicase_select(rng, pop, case_matrix, epsilons=eps)
            b = epsilon_lexicase_select(rng, pop, case_matrix, epsilons=eps)
            child, _ = p.crossover(rng, a, b)
            next_pop.append(child)
        else:
            parent = epsilon_lexicase_select(rng, pop, case_matrix,
                                             epsilons=eps)
            next_pop.append(p.mutate(rng, parent, rates=mutation_rates))

    return next_pop[:pop_size]
