# train_maskable_ppo.py
import numpy as np
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from tree_evaluator import (
    TreeHydraulicEvaluator,
    load_nodes_xlsx,
    load_pipes_xlsx,
    build_lateral_ids_for_field_nodes,
    is_field_node_id,
)
from ppo_env import IrrigationGroupingEnv


def make_env(seed: int):
    nodes = load_nodes_xlsx("Nodes.xlsx")
    edges = load_pipes_xlsx("Pipes.xlsx")
    evaluator = TreeHydraulicEvaluator(nodes=nodes, edges=edges, root="J0", H0=25.0, Hmin=11.59)

    field_nodes = [nid for nid in nodes.keys() if is_field_node_id(nid)]
    lateral_ids, lateral_to_node = build_lateral_ids_for_field_nodes(field_nodes)

    # single_margin_map 可选：给 policy 一个启发式特征
    single_margin_map = {}
    for lid in lateral_ids:
        r = evaluator.evaluate_group([lid], lateral_to_node=lateral_to_node, q_lateral=0.012)
        single_margin_map[lid] = float(r.min_margin)

    env = IrrigationGroupingEnv(
        evaluator=evaluator,
        lateral_ids=lateral_ids,
        lateral_to_node=lateral_to_node,
        single_margin_map=single_margin_map,
        beta_infeasible=1e4,
        alpha_var_final=0.0,       # 第1条修改：先关掉回合末额外罚
        lambda_branch_soft=0.1,
        seed=seed,
    )
    return Monitor(env)


if __name__ == "__main__":
    train_env = make_env(seed=0)
    eval_env = make_env(seed=1)

    eval_cb = MaskableEvalCallback(
        eval_env,
        eval_freq=10_000,
        deterministic=True,
        best_model_save_path="./best_model",
        log_path="./eval_logs",
    )

    model = MaskablePPO(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./tb",
    )

    model.learn(total_timesteps=1_000_000, callback=eval_cb)  # 先跑通 1e5，再逐步加大
    model.save("ppo_irrigation_masked")
