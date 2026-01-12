"""
NSGA‑II algorithm for multi‑objective irrigation grouping optimisation.

This script provides a pure Python/NumPy implementation of the Non‑Dominated
Sorting Genetic Algorithm II (NSGA‑II) to solve the irrigation grouping
problem.  NSGA‑II is a popular evolutionary algorithm for multi‑objective
optimisation that combines fast non‑dominated sorting with a crowding
distance measure to maintain a diverse set of Pareto‑optimal solutions【246753596889157†L171-L189】.

The algorithm is described as follows【246753596889157†L171-L189】:

1. **Initialisation** – generate an initial population of candidate solutions (here,
   random permutations of all laterals) and evaluate their objective values.
2. **Non‑dominated sorting** – rank individuals into fronts based on Pareto
   dominance.  Individuals in the first front are non‑dominated, those in the
   second front are dominated only by individuals in the first front, and so
   on.  A crowding distance is computed within each front to encourage
   diversity【246753596889157†L171-L189】.
3. **Selection** – use binary tournament selection based on rank and crowding
   distance to choose parents.
4. **Crossover and mutation** – produce offspring via order crossover and
   swap mutation.  Crossover exchanges segments between two parent
   permutations while preserving the sequence property; mutation swaps two
   positions in a permutation.
5. **Environmental selection** – combine parent and offspring populations,
   perform non‑dominated sorting and crowding distance computation again,
   and select the best individuals to form the next generation.

The objectives for the irrigation grouping problem are:

* Minimise the variance of the minimum hydraulic margins (s_g) across groups.
* Maximise the minimum margin across the entire episode (implemented by
  minimising its negative).

Each candidate solution is represented as a permutation of the lateral
indices.  Evaluation is performed using the ``evaluate_permutation``
function defined below.  You must provide an environment instance
(``IrrigationGroupingEnv``) configured with your hydraulic network.

Usage
-----
1. Make sure ``tree_evaluator.py`` and ``ppo_env.py`` are available and that
   ``Nodes.xlsx`` and ``Pipes.xlsx`` reside in the working directory.
2. Adjust the population size, number of generations, crossover and mutation
   rates in the ``main`` function to suit your problem size and time budget.
3. Run this script.  The final Pareto front is printed along with objective
   values.

Note
----
This implementation is intended for experimentation and educational purposes.
For large populations or many generations it may be slow because it is
implemented purely in Python.  Libraries such as ``pymoo`` provide
optimised implementations of NSGA‑II, but may not be available in all
environments.  Nevertheless, the algorithm here follows the same steps as
described in the literature【246753596889157†L171-L189】.
"""

from __future__ import annotations

import random
from typing import List, Tuple, Dict, Optional

import numpy as np

try:
    from ppo_env import IrrigationGroupingEnv
    from tree_evaluator import (
        TreeHydraulicEvaluator,
        load_nodes_xlsx,
        load_pipes_xlsx,
        build_lateral_ids_for_field_nodes,
        is_field_node_id,
    )
except ImportError:
    raise ImportError(
        "Required modules not found. Ensure `ppo_env.py` and `tree_evaluator.py` "
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
    obs, _ = env.reset()
    final_var = None
    neg_min_margin = None
    for act in order:
        obs, reward, done, truncated, info = env.step(int(act))
        if done:
            final_var = info.get("final_var", info.get("running_var"))
            neg_min_margin = -info.get("min_s_over_episode", 0.0)
            break
    if final_var is None:
        final_var = info.get("final_var", info.get("running_var"))
        neg_min_margin = -info.get("min_s_over_episode", 0.0)
    return float(final_var), float(neg_min_margin)


def dominates(f1: Tuple[float, float], f2: Tuple[float, float]) -> bool:
    """Check if fitness tuple f1 dominates f2 (both objectives to be minimized)."""
    return (f1[0] <= f2[0] and f1[1] <= f2[1]) and (f1[0] < f2[0] or f1[1] < f2[1])


def non_dominated_sort(fitnesses: List[Tuple[float, float]]) -> List[List[int]]:
    """Perform non‑dominated sorting on a list of fitnesses.

    Returns a list of fronts, each front being a list of indices.
    """
    population_size = len(fitnesses)
    S = [set() for _ in range(population_size)]  # solutions dominated by i
    n = [0] * population_size                   # number of solutions that dominate i
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
        distance[obj_values[0][0]] = float('inf')
        distance[obj_values[-1][0]] = float('inf')
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
    rng: random.Random
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


def build_environment(seed: int = 0) -> IrrigationGroupingEnv:
    """Instantiate the irrigation grouping environment using network data.

    This function loads the network from Excel files, constructs the lateral
    mapping and single margin dictionary, and returns an ``IrrigationGroupingEnv``
    instance.  Adjust hyperparameters here to match your PPO setup if needed.
    """
    nodes = load_nodes_xlsx("Nodes.xlsx")
    edges = load_pipes_xlsx("Pipes.xlsx")
    evaluator = TreeHydraulicEvaluator(nodes=nodes, edges=edges, root="J0", H0=25.0, Hmin=11.59)
    field_nodes = [nid for nid in nodes.keys() if is_field_node_id(nid)]
    lateral_ids, lateral_to_node = build_lateral_ids_for_field_nodes(field_nodes)
    single_margin_map: Dict[str, float] = {}
    for lid in lateral_ids:
        r = evaluator.evaluate_group([lid], lateral_to_node=lateral_to_node, q_lateral=0.012)
        single_margin_map[lid] = float(r.min_margin)
    env = IrrigationGroupingEnv(
        evaluator=evaluator,
        lateral_ids=lateral_ids,
        lateral_to_node=lateral_to_node,
        single_margin_map=single_margin_map,
        beta_infeasible=1e4,
        alpha_var_final=0.0,
        lambda_branch_soft=0.1,
        seed=seed,
    )
    return env


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
    """Run NSGA‑II and print the resulting Pareto front."""
    env = build_environment(seed=0)
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


if __name__ == "__main__":
    main()