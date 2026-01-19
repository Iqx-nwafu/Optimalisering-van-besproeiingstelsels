# Modified NSGA-II solution to output top K candidate solutions by composite score.
#
"""
NSGA‑II algorithm for multi‑objective irrigation grouping optimisation.

This patched variant adds a post-processing step to report the same number of
top candidate solutions as produced by the PPO baseline.  After running the
NSGA‑II algorithm, the final population is sorted by a composite score
(variance + penalty on negative minimum margin), and the top ``TOPK_RESULTS``
solutions are printed.  The original printouts of the Pareto front and best
solution are preserved.
"""

from __future__ import annotations

import random
from typing import List, Tuple, Dict, Optional

import numpy as np

try:
    from ppo_env import IrrigationGroupingEnv
    from comparison_utils import (
        ComparisonConfig,
        build_environment as build_shared_env,
        composite_score,
        evaluate_order,
    )
except ImportError:
    raise ImportError(
        "Required modules not found. Ensure `ppo_env.py` and `comparison_utils.py` "
        "are accessible in the PYTHONPATH."
    )


def evaluate_permutation(env: IrrigationGroupingEnv, order: List[int]) -> Tuple[float, float]:
    """Run an episode using the specified ordering and return objective values.

    Parameters
    ----------
    env : IrrigationGroupingEnv
        Environment instance.
    order : List[int]
        List of lateral indices specifying the order of activation.

    Returns
    -------
    tuple(float, float)
        (final_variance, negative_min_margin) objective values.  A smaller
        final variance and a smaller negative minimum margin (i.e. a larger
        minimum margin) are better.
    """
    metrics = evaluate_order(env, order)
    return metrics["final_var"], -metrics["min_margin"]


def dominates(f1: Tuple[float, float], f2: Tuple[float, float]) -> bool:
    """Check if fitness tuple f1 dominates f2 (both objectives to be minimized)."""
    return (f1[0] <= f2[0] and f1[1] <= f2[1]) and (f1[0] < f2[0] or f1[1] < f2[1])


def non_dominated_sort(fitnesses: List[Tuple[float, float]]) -> List[List[int]]:
    """Perform non‑dominated sorting on a list of fitnesses.

    Returns a list of fronts, each front being a list of indices.
    """
    population_size = len(fitnesses)
    S = [set() for _ in range(population_size)]  # solutions dominated by i
    n = [0] * population_size  # number of solutions that dominate i
    fronts: List[List[int]] = []
    # Identify domination relations
    for i in range(population_size):
        for j in range(population_size):
            if i == j:
                continue
            if dominates(fitnesses[i], fitnesses[j]):
                S[i].add(j)
            elif dominates(fitnesses[j], fitnesses[i]):
                n[i] += 1
        if n[i] == 0:
            # i belongs to first front
            if len(fronts) == 0:
                fronts.append([])
            fronts[0].append(i)
    # Generate subsequent fronts
    current_front = 0
    while current_front < len(fronts):
        next_front: List[int] = []
        for i in fronts[current_front]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def crowding_distance(front: List[int], fitnesses: List[Tuple[float, float]]) -> Dict[int, float]:
    """Compute crowding distances for a single front.

    Parameters
    ----------
    front : List[int]
        List of individual indices in the same non‑dominated front.
    fitnesses : List[Tuple[float, float]]
        List of objective tuples for the entire population.

    Returns
    -------
    Dict[int, float]
        Mapping from individual index to its crowding distance.
    """
    distance = {i: 0.0 for i in front}
    num_objectives = 2
    # For each objective, sort the front by that objective
    for m in range(num_objectives):
        # Extract the m‑th objective values
        obj_values = [(i, fitnesses[i][m]) for i in front]
        obj_values.sort(key=lambda x: x[1])
        # Assign infinite distance to boundary solutions
        distance[obj_values[0][0]] = float("inf")
        distance[obj_values[-1][0]] = float("inf")
        # Normalisation factor to avoid division by zero
        min_val = obj_values[0][1]
        max_val = obj_values[-1][1]
        span = max_val - min_val if max_val != min_val else 1.0
        # Compute distances for internal points
        for k in range(1, len(obj_values) - 1):
            prev_val = obj_values[k - 1][1]
            next_val = obj_values[k + 1][1]
            # Normalised absolute difference
            distance[obj_values[k][0]] += (next_val - prev_val) / span
    return distance


def tournament_selection(
    pop: List[List[int]],
    fronts: List[List[int]],
    distances: List[Dict[int, float]],
    pop_size: int,
    rng: random.Random,
) -> List[List[int]]:
    """Binary tournament selection based on rank and crowding distance.

    Parameters
    ----------
    pop : List[List[int]]
        Current population (list of permutations).
    fronts : List[List[int]]
        Non‑dominated fronts.
    distances : List[Dict[int, float]]
        Crowding distances for each front.
    pop_size : int
        Desired number of selected individuals.
    rng : random.Random
        Random number generator.

    Returns
    -------
    List[List[int]]
        Selected parent individuals.
    """
    # Create a mapping from individual index to its front rank
    rank = {}
    for i, f in enumerate(fronts):
        for idx in f:
            rank[idx] = i
    selected: List[List[int]] = []
    while len(selected) < pop_size:
        i1, i2 = rng.randrange(len(pop)), rng.randrange(len(pop))
        r1, r2 = rank[i1], rank[i2]
        if r1 < r2:
            winner = i1
        elif r2 < r1:
            winner = i2
        else:
            # same rank – compare crowding distances
            d1 = distances[r1].get(i1, 0.0)
            d2 = distances[r2].get(i2, 0.0)
            if d1 > d2:
                winner = i1
            else:
                winner = i2
        selected.append(pop[winner])
    return selected


def order_crossover(parent1: List[int], parent2: List[int], rng: random.Random) -> List[int]:
    """Perform order crossover (OX) on two parent permutations.

    A segment from the first parent is copied to the child, and the remaining
    positions are filled with the order of genes from the second parent.
    """
    size = len(parent1)
    a, b = sorted(rng.sample(range(size), 2))
    child = [-1] * size
    # Copy segment from parent1
    child[a:b] = parent1[a:b]
    # Fill remaining positions with genes from parent2
    fill_candidates = [g for g in parent2 if g not in child]
    fill_idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = fill_candidates[fill_idx]
            fill_idx += 1
    return child


def swap_mutation(permutation: List[int], rng: random.Random) -> List[int]:
    """Swap two positions in the permutation."""
    size = len(permutation)
    i, j = rng.sample(range(size), 2)
    perm = permutation.copy()
    perm[i], perm[j] = perm[j], perm[i]
    return perm


def build_environment(seed: int = 0, config: Optional[ComparisonConfig] = None) -> IrrigationGroupingEnv:
    """Instantiate the irrigation grouping environment with shared settings."""
    return build_shared_env(seed=seed, config=config)


def nsga2(
    env: IrrigationGroupingEnv,
    pop_size: int = 20,
    generations: int = 30,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    seed: Optional[int] = None,
) -> Tuple[List[List[int]], List[Tuple[float, float]]]:
    """Run the NSGA‑II algorithm.

    Parameters
    ----------
    env : IrrigationGroupingEnv
        Environment used to evaluate permutations.
    pop_size : int
        Number of individuals in the population.
    generations : int
        Number of generations to evolve.
    crossover_rate : float
        Probability of performing crossover on a pair of parents.
    mutation_rate : float
        Probability of performing mutation on an offspring.
    seed : Optional[int]
        Random seed.

    Returns
    -------
    Tuple[List[List[int]], List[Tuple[float, float]]]
        Final population and their corresponding fitness values.
    """
    rng = random.Random(seed)
    N = env.N
    # Initialise population with random permutations
    population: List[List[int]] = [rng.sample(range(N), N) for _ in range(pop_size)]
    # Evaluate initial population
    fitnesses: List[Tuple[float, float]] = [evaluate_permutation(env, ind) for ind in population]
    for gen in range(generations):
        # Perform non‑dominated sorting and compute crowding distances
        fronts = non_dominated_sort(fitnesses)
        distances: List[Dict[int, float]] = []
        for f in fronts:
            distances.append(crowding_distance(f, fitnesses))
        # Select parents via tournament selection
        parents = tournament_selection(population, fronts, distances, pop_size, rng)
        # Generate offspring
        offspring: List[List[int]] = []
        while len(offspring) < pop_size:
            p1 = parents[rng.randrange(pop_size)]
            p2 = parents[rng.randrange(pop_size)]
            if rng.random() < crossover_rate:
                child = order_crossover(p1, p2, rng)
            else:
                child = p1.copy()
            if rng.random() < mutation_rate:
                child = swap_mutation(child, rng)
            offspring.append(child)
        # Evaluate offspring
        offspring_fitnesses: List[Tuple[float, float]] = [evaluate_permutation(env, ind) for ind in offspring]
        # Combine populations
        combined_population = population + offspring
        combined_fitnesses = fitnesses + offspring_fitnesses
        # Sort combined population using NSGA‑II selection
        new_population: List[List[int]] = []
        new_fitnesses: List[Tuple[float, float]] = []
        fronts = non_dominated_sort(combined_fitnesses)
        for f in fronts:
            cd = crowding_distance(f, combined_fitnesses)
            # Sort individuals in this front by crowding distance descending
            sorted_front = sorted(f, key=lambda i: cd[i], reverse=True)
            for idx in sorted_front:
                if len(new_population) < pop_size:
                    new_population.append(combined_population[idx])
                    new_fitnesses.append(combined_fitnesses[idx])
                else:
                    break
            if len(new_population) >= pop_size:
                break
        population, fitnesses = new_population, new_fitnesses
        print(f"Generation {gen + 1}/{generations} completed.")
    return population, fitnesses


def main() -> None:
    """Run NSGA‑II and print the resulting Pareto front and top candidate solutions.

    In addition to listing the Pareto‑optimal solutions, this entry point now
    prints the top ``TOPK_RESULTS`` solutions based on the composite score,
    ensuring the same number of outputs as the PPO baseline.
    """
    # Define how many solutions to output (e.g. match PPO's TOPK)
    TOPK_RESULTS = 30
    config = ComparisonConfig()
    env = build_environment(seed=0, config=config)
    print(f"Environment loaded with {env.N} laterals.")
    pop, fits = nsga2(env, pop_size=20, generations=30, crossover_rate=0.9, mutation_rate=0.2, seed=0)
    # After evolution, identify non‑dominated solutions in the final population
    fronts = non_dominated_sort(fits)
    pareto_front = fronts[0]
    print(f"Found {len(pareto_front)} Pareto‑optimal solutions in the final generation:")
    for idx in pareto_front:
        solution = pop[idx]
        var, neg_min = fits[idx]
        print(f"  Solution index {idx}: variance={var:.6f}, min_margin={-neg_min:.6f}")
        # Optionally print the schedule as lateral IDs
        lids = [env.lateral_ids[i] for i in solution]
        print(f"    Ordering: {lids}")
    # Composite-score ranking for top solutions
    scored = [(composite_score(fits[i][0], -fits[i][1]), i) for i in range(len(pop))]
    scored.sort(key=lambda x: x[0])
    best_score, best_idx = scored[0]
    print("\nBest solution by composite score (variance + margin penalty):")
    print(f"Composite score: {best_score:.6f}")
    print(f"Fitness: final_var={fits[best_idx][0]:.6f}, neg_min_margin={fits[best_idx][1]:.6f}")
    print(f"Ordering: {pop[best_idx]}")
    # New: print top K solutions based on composite score
    print(f"\nTop {TOPK_RESULTS} candidate solutions by composite score:")
    for rank, (score, idx) in enumerate(scored[:TOPK_RESULTS], start=1):
        var, neg_min = fits[idx]
        print(
            f"{rank:02d}: composite_score={score:.6f}, final_var={var:.6f}, min_margin={-neg_min:.6f}"
        )
        lids = [env.lateral_ids[i] for i in pop[idx]]
        print(f"    Ordering: {lids}")


if __name__ == "__main__":
    main()
