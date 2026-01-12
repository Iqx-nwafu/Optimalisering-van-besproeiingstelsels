# train_maskable_ppo.py
from __future__ import annotations

import os
import multiprocessing as mp
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from tree_evaluator import (
    TreeHydraulicEvaluator,
    load_nodes_xlsx,
    load_pipes_xlsx,
    build_lateral_ids_for_field_nodes,
    is_field_node_id,
)
from ppo_env import IrrigationGroupingEnv, RewardWeights, RewardScales, SafetyShaping


@lru_cache(maxsize=1)
def _load_base() -> Tuple[Dict, List, List[str], Dict[str, str], Dict[str, float]]:
    """
    Load Nodes.xlsx/Pipes.xlsx once, and precompute:
    - nodes, edges
    - lateral_ids, lateral_to_node
    - single_margin_map (optional feature)
    """
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


def _build_env(
    seed: int,
    reward_weights: RewardWeights,
    reward_scales: RewardScales,
    safety_shaping: SafetyShaping,
    include_single_margin: bool = True,
) -> IrrigationGroupingEnv:
    nodes, edges, lateral_ids, lateral_to_node, single_margin_map = _load_base()

    # Each env needs its own evaluator instance (safe for subprocess)
    evaluator = TreeHydraulicEvaluator(nodes=nodes, edges=edges, root="J0", H0=25.0, Hmin=11.59)

    env = IrrigationGroupingEnv(
        evaluator=evaluator,
        lateral_ids=lateral_ids,
        lateral_to_node=lateral_to_node,

        include_single_margin=include_single_margin,
        single_margin_map=(single_margin_map if include_single_margin else None),

        # normalized hard constraints
        infeasible_reward=-1.0,
        invalid_action_reward=-1.0,

        # soft constraint
        lambda_branch_soft=0.05,

        # reward engineering
        reward_weights=reward_weights,
        reward_scales=reward_scales,
        safety_shaping=safety_shaping,
        reward_clip=1.0,

        seed=seed,
    )
    return env


def make_env(
    seed: int,
    reward_weights: RewardWeights = RewardWeights(),
    reward_scales: RewardScales = RewardScales(),
    safety_shaping: SafetyShaping = SafetyShaping(),
    include_single_margin: bool = True,
):
    """Single env for evaluation/debug; keep compatible with eval_one.py."""
    env = _build_env(seed, reward_weights, reward_scales, safety_shaping, include_single_margin)
    return Monitor(env)


def make_vec_env(
    num_envs: int,
    start_seed: int = 0,
    reward_weights: RewardWeights = RewardWeights(),
    reward_scales: RewardScales = RewardScales(),
    safety_shaping: SafetyShaping = SafetyShaping(),
    include_single_margin: bool = True,
):
    """Parallel VecEnv for training."""
    def thunk(rank: int):
        def _init():
            return _build_env(
                seed=start_seed + rank,
                reward_weights=reward_weights,
                reward_scales=reward_scales,
                safety_shaping=safety_shaping,
                include_single_margin=include_single_margin,
            )
        return _init

    env_fns = [thunk(i) for i in range(num_envs)]
    vec = SubprocVecEnv(env_fns)
    vec = VecMonitor(vec)
    return vec


def linear_schedule(initial_value: float):
    def f(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return f


if __name__ == "__main__":
    mp.freeze_support()

    # i5-12400F (6C/12T) recommended defaults:
    num_envs = int(os.environ.get("N_ENVS", "6"))          # try 6 first; then 8 if stable
    torch_threads = int(os.environ.get("TORCH_THREADS", "4"))
    torch.set_num_threads(torch_threads)

    # ---- Reward configuration (tune these)
    reward_weights = RewardWeights(
        w_var_step=0.5,
        w_mean_step=0.1,
        w_safe_step=0.2,
        w_var_final=0.67,
        w_mean_final=0.33,
        w_min_final=0.0,
    )
    reward_scales = RewardScales(
        var_step=1.0,
        mean_step=1.0,
        var_final=1.0,
        mean_scale=10.0,
    )
    safety_shaping = SafetyShaping(
        s_safe=1.0,      # start penalizing when margin < 1m
        mode="linear",   # or "exp"
        tau=0.5,
    )

    train_env = make_vec_env(
        num_envs=num_envs,
        start_seed=0,
        reward_weights=reward_weights,
        reward_scales=reward_scales,
        safety_shaping=safety_shaping,
        include_single_margin=True,
    )
    eval_env = make_env(
        seed=10_000,
        reward_weights=reward_weights,
        reward_scales=reward_scales,
        safety_shaping=safety_shaping,
        include_single_margin=True,
    )

    eval_cb = MaskableEvalCallback(
        eval_env,
        eval_freq=max(1, 50_000 // num_envs),  # vector env counts timesteps faster
        deterministic=True,
        best_model_save_path="./best_model",
        log_path="./eval_logs",
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./tb",
        device="cpu",

        # CPU parallel-friendly hyperparams
        n_steps=256,
        batch_size=512,        # ensure divides n_steps*num_envs
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=linear_schedule(3e-4),
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    model.learn(total_timesteps=2_000_000, callback=eval_cb)
    model.save("ppo_irrigation_masked")
