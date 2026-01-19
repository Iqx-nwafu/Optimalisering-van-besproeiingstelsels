"""
Shared comparison utilities for NSGA-II, DNN, and PPO baselines.

This module centralizes environment construction, scoring, and evaluation so
that all algorithms are compared under the same hydraulic conditions and
metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ppo_env import IrrigationGroupingEnv, RewardWeights, RewardScales, SafetyShaping
from tree_evaluator import (
    TreeHydraulicEvaluator,
    load_nodes_xlsx,
    load_pipes_xlsx,
    build_lateral_ids_for_field_nodes,
    is_field_node_id,
)


@dataclass(frozen=True)
class ComparisonConfig:
    """Configuration shared by all algorithms for a fair comparison."""

    H0: float = 25.0
    Hmin: float = 11.59
    q_lateral: float = 0.012
    include_single_margin: bool = True
    reward_weights: RewardWeights = RewardWeights(
        w_var_step=0.5,
        w_mean_step=0.1,
        w_safe_step=0.2,
        w_var_final=0.67,
        w_mean_final=0.33,
        w_min_final=0.0,
    )
    reward_scales: RewardScales = RewardScales(
        var_step=1.0,
        mean_step=1.0,
        var_final=1.0,
        mean_scale=10.0,
    )
    safety_shaping: SafetyShaping = SafetyShaping(
        s_safe=1.0,
        mode="linear",
        tau=0.5,
    )
    infeasible_reward: float = -1.0
    invalid_action_reward: float = -1.0
    lambda_branch_soft: float = 0.05
    reward_clip: float = 1.0


def _load_network() -> Tuple[Dict, List, List[str], Dict[str, str], Dict[str, float]]:
    """Load network data and precompute lateral metadata."""
    nodes = load_nodes_xlsx("Nodes.xlsx")
    edges = load_pipes_xlsx("Pipes.xlsx")

    evaluator = TreeHydraulicEvaluator(nodes=nodes, edges=edges, root="J0", H0=25.0, Hmin=11.59)

    field_nodes = [nid for nid in nodes.keys() if is_field_node_id(nid)]
    lateral_ids, lateral_to_node = build_lateral_ids_for_field_nodes(field_nodes)

    single_margin_map = {}
    for lid in lateral_ids:
        r = evaluator.evaluate_group([lid], lateral_to_node=lateral_to_node, q_lateral=0.012)
        single_margin_map[lid] = float(r.min_margin)

    return nodes, edges, lateral_ids, lateral_to_node, single_margin_map


def build_environment(seed: int, config: Optional[ComparisonConfig] = None) -> IrrigationGroupingEnv:
    """Build an IrrigationGroupingEnv with shared comparison settings."""
    cfg = config or ComparisonConfig()
    nodes, edges, lateral_ids, lateral_to_node, single_margin_map = _load_network()

    evaluator = TreeHydraulicEvaluator(nodes=nodes, edges=edges, root="J0", H0=cfg.H0, Hmin=cfg.Hmin)

    env = IrrigationGroupingEnv(
        evaluator=evaluator,
        lateral_ids=lateral_ids,
        lateral_to_node=lateral_to_node,
        Hmin=cfg.Hmin,
        q_lateral=cfg.q_lateral,
        include_single_margin=cfg.include_single_margin,
        single_margin_map=(single_margin_map if cfg.include_single_margin else None),
        infeasible_reward=cfg.infeasible_reward,
        invalid_action_reward=cfg.invalid_action_reward,
        lambda_branch_soft=cfg.lambda_branch_soft,
        reward_weights=cfg.reward_weights,
        reward_scales=cfg.reward_scales,
        safety_shaping=cfg.safety_shaping,
        reward_clip=cfg.reward_clip,
        seed=seed,
    )
    return env


def evaluate_order(env: IrrigationGroupingEnv, order: List[int]) -> Dict[str, float]:
    """Run one episode for a fixed action order and return key metrics."""
    _, _ = env.reset()
    info_last: Dict[str, float] = {}
    done = False
    for act in order:
        _, _, done, _, info = env.step(int(act))
        info_last = info
        if done:
            break

    final_var = float(info_last.get("final_var", info_last.get("running_var", 0.0)))
    min_margin = float(info_last.get("min_s_over_episode", 0.0))
    final_mean_s = float(info_last.get("final_mean_s", 0.0))
    ok = bool(info_last.get("ok", True))

    return {
        "final_var": final_var,
        "min_margin": min_margin,
        "final_mean_s": final_mean_s,
        "ok": ok,
    }


def composite_score(final_var: float, min_margin: float, margin_weight: float = 1.0) -> float:
    """Combine variance and minimum margin into a single minimization score."""
    return float(final_var + margin_weight * (-min_margin))

