# ppo_env.py
"""
Gymnasium environment for MaskablePPO on irrigation grouping.

Reward engineering (normalized):
- Hard constraint: if any group violates Hmin => terminate with reward = -1.
- Step shaping: reward = -w_var_step * Δvar + w_mean_step * Δmean - w_safe_step * safety_penalty + soft_branch_penalty
- Terminal reward: -w_var_final * final_var + w_mean_final * tanh(mean_s/mean_scale) (+ optional min margin term)
- All rewards clipped to [-1, 1] (configurable reward_clip).

This keeps action masking (action_masks) for sb3-contrib MaskablePPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces


# -------------------------
# Utilities for your node naming
# -------------------------

def _node_num(nid: str) -> int:
    nid = nid.strip().upper()
    if not nid.startswith("J"):
        return -1
    try:
        return int(nid[1:])
    except ValueError:
        return -1


def branch_id_from_field_node(nid: str) -> int:
    """J11..J16 -> branch 1; ...; J91..J96 -> 9; J101..J106 -> 10."""
    n = _node_num(nid)
    if n >= 101:
        return 10
    return n // 10


def pos_in_branch_from_field_node(nid: str) -> int:
    """J11->1 ... J16->6; J101->1 ... J106->6."""
    n = _node_num(nid)
    return n % 10


# -------------------------
# Online mean/variance (Welford)
# -------------------------

@dataclass
class WelfordState:
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def variance(self) -> float:
        if self.n <= 1:
            return 0.0
        return self.M2 / (self.n - 1)

    def update(self, x: float) -> Tuple[float, float, float, float]:
        """Return old_mean,new_mean, old_var,new_var."""
        old_mean = self.mean
        old_var = self.variance()

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

        new_mean = self.mean
        new_var = self.variance()
        return old_mean, new_mean, old_var, new_var


# -------------------------
# Reward configs
# -------------------------

@dataclass(frozen=True)
class RewardWeights:
    # step-level shaping
    w_var_step: float = 0.6
    w_mean_step: float = 0.1
    w_safe_step: float = 0.3

    # terminal objectives
    w_var_final: float = 0.7
    w_mean_final: float = 0.3
    w_min_final: float = 0.0  # optional: encourage larger worst-group margin


@dataclass(frozen=True)
class RewardScales:
    # scale raw terms to O(1)
    var_step: float = 1.0
    mean_step: float = 1.0
    var_final: float = 1.0
    mean_scale: float = 10.0  # tanh(mean_s/mean_scale)


@dataclass(frozen=True)
class SafetyShaping:
    """Penalty starts before violation."""
    s_safe: float = 1.0       # (m) if 0<=s_g<s_safe => penalize
    mode: str = "linear"      # "linear" or "exp"
    tau: float = 0.5          # for "exp"


def safety_penalty(s_g: float, cfg: SafetyShaping) -> float:
    """
    Return penalty in [0,1].
    - s_g >= s_safe => 0
    - 0 <= s_g < s_safe => increases to 1 as s_g -> 0+
    - s_g < 0 => caller should terminate (hard constraint)
    """
    if s_g >= cfg.s_safe:
        return 0.0
    if s_g <= 0.0:
        return 1.0
    x = (cfg.s_safe - s_g) / cfg.s_safe  # (0,1]
    if cfg.mode == "linear":
        return float(np.clip(x, 0.0, 1.0))
    if cfg.mode == "exp":
        return float(1.0 - math.exp(-(x / max(1e-6, cfg.tau))))
    raise ValueError(f"Unknown SafetyShaping.mode={cfg.mode}")


# -------------------------
# Environment
# -------------------------

class IrrigationGroupingEnv(gym.Env):
    """120-step episode. Every 4 steps evaluate one group via fast hydraulics."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        evaluator,  # TreeHydraulicEvaluator
        lateral_ids: List[str],
        lateral_to_node: Dict[str, str],
        Hmin: float = 11.59,
        q_lateral: float = 0.012,

        # normalized hard constraint rewards
        infeasible_reward: float = -1.0,
        invalid_action_reward: float = -1.0,

        # soft constraint (small magnitude)
        lambda_branch_soft: float = 0.05,

        # reward engineering
        reward_weights: RewardWeights = RewardWeights(),
        reward_scales: RewardScales = RewardScales(),
        safety_shaping: SafetyShaping = SafetyShaping(),
        reward_clip: float = 1.0,

        # features
        include_single_margin: bool = True,
        single_margin_map: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.evaluator = evaluator
        self.lateral_ids = list(lateral_ids)
        self.N = len(self.lateral_ids)
        self.lateral_to_node = dict(lateral_to_node)

        self.Hmin = float(Hmin)
        self.q_lateral = float(q_lateral)

        self.infeasible_reward = float(infeasible_reward)
        self.invalid_action_reward = float(invalid_action_reward)
        self.lambda_branch_soft = float(lambda_branch_soft)

        self.w = reward_weights
        self.sc = reward_scales
        self.safe_cfg = safety_shaping
        self.reward_clip = float(reward_clip)

        self.include_single_margin = bool(include_single_margin)
        self.single_margin_map = single_margin_map or {}

        self.rng = np.random.default_rng(seed)

        # Precompute static features
        self._feat_static, self._branch_ids = self._build_static_features()
        self.F_static = self._feat_static.shape[1]

        # Observation: per-lateral features + selected flag; and global stats
        self.F = self.F_static + 1
        self.observation_space = spaces.Dict(
            {
                "feat": spaces.Box(low=-np.inf, high=np.inf, shape=(self.N, self.F), dtype=np.float32),
                "global": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(self.N)

        # Episode state
        self.selected = np.zeros(self.N, dtype=bool)
        self.current_group: List[int] = []
        self.group_scores: List[float] = []
        self.stats = WelfordState()
        self.step_count = 0

        # logs
        self.soft_sum = 0.0
        self.min_s_over_episode = float("inf")

    def _build_static_features(self) -> Tuple[np.ndarray, np.ndarray]:
        feats = []
        branch_ids = []
        for lid in self.lateral_ids:
            node = self.lateral_to_node[lid]
            z = float(self.evaluator.nodes[node].z)
            b = float(branch_id_from_field_node(node))
            p = float(pos_in_branch_from_field_node(node))
            side = 0.0 if lid.endswith("_L") else 1.0

            row = [z, b, p, side]
            if self.include_single_margin:
                row.append(float(self.single_margin_map.get(lid, 0.0)))

            feats.append(row)
            branch_ids.append(int(b))

        return np.asarray(feats, dtype=np.float32), np.asarray(branch_ids, dtype=np.int32)

    # MaskablePPO hook
    def action_masks(self) -> np.ndarray:
        return (~self.selected).astype(bool)

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.selected[:] = False
        self.current_group = []
        self.group_scores = []
        self.stats = WelfordState()
        self.step_count = 0
        self.soft_sum = 0.0
        self.min_s_over_episode = float("inf")
        return self._get_obs(), {}

    def _get_obs(self) -> Dict[str, np.ndarray]:
        sel = self.selected.astype(np.float32).reshape(self.N, 1)
        feat = np.concatenate([self._feat_static, sel], axis=1).astype(np.float32)

        g_index = len(self.group_scores)
        within = len(self.current_group)

        glob = np.array(
            [
                float(self.step_count) / 120.0,
                float(g_index) / 30.0,
                float(within) / 4.0,
                float(self.stats.mean),
                float(self.stats.variance()),
                float(self.stats.n),
            ],
            dtype=np.float32,
        )
        return {"feat": feat, "global": glob}

    def _clip(self, r: float) -> float:
        return float(np.clip(r, -self.reward_clip, self.reward_clip))

    def step(self, action: int):
        # invalid action
        if action < 0 or action >= self.N or self.selected[action]:
            info = {
                "ok": False,
                "reason": "invalid_action",
                "groups_done": len(self.group_scores),
                "within_group": len(self.current_group),
                "running_mean_s": float(self.stats.mean),
                "running_var_s": float(self.stats.variance()),
                "soft_sum": float(self.soft_sum),
            }
            return self._get_obs(), self._clip(self.invalid_action_reward), True, False, info

        # apply action
        self.selected[action] = True
        self.current_group.append(action)
        self.step_count += 1

        reward = 0.0
        terminated = False

        # complete one group
        if len(self.current_group) == 4:
            group_lids = [self.lateral_ids[i] for i in self.current_group]

            # soft penalty: branch repetition in group
            branch_counts: Dict[int, int] = {}
            for i in self.current_group:
                b = int(self._branch_ids[i])
                branch_counts[b] = branch_counts.get(b, 0) + 1
            soft = -self.lambda_branch_soft * sum(max(0, c - 1) for c in branch_counts.values())
            self.soft_sum += float(soft)

            res = self.evaluator.evaluate_group(
                group_lids, lateral_to_node=self.lateral_to_node, q_lateral=self.q_lateral
            )

            # hard constraint violated
            if not res.ok:
                terminated = True
                info = {
                    "ok": False,
                    "min_margin": float(res.min_margin),
                    "min_pressure_head": float(res.min_pressure_head),
                    "soft_sum": float(self.soft_sum),
                    "groups_done": len(self.group_scores),
                    "within_group": 4,
                    "running_mean_s": float(self.stats.mean),
                    "running_var_s": float(self.stats.variance()),
                    "final_var": float(self.stats.variance()),
                    "final_mean_s": float(self.stats.mean),
                    "min_s_over_episode": (
                        float(self.min_s_over_episode) if math.isfinite(self.min_s_over_episode) else float("nan")
                    ),
                }
                # normalized terminal reward
                reward = self.infeasible_reward + soft
                return self._get_obs(), self._clip(reward), terminated, False, info

            # feasible group
            s_g = float(res.min_margin)

            old_mean, new_mean, old_var, new_var = self.stats.update(s_g)

            # step reward components (all O(1))
            r_var = -self.w.w_var_step * ((new_var - old_var) / max(1e-8, self.sc.var_step))
            r_mean = self.w.w_mean_step * ((new_mean - old_mean) / max(1e-8, self.sc.mean_step))
            r_safe = -self.w.w_safe_step * safety_penalty(s_g, self.safe_cfg)

            reward = r_var + r_mean + r_safe + soft

            self.group_scores.append(s_g)
            self.min_s_over_episode = min(self.min_s_over_episode, s_g)
            self.current_group = []

        # end of episode
        if self.step_count == 120 and not terminated:
            final_var = float(self.stats.variance())
            mean_s = float(self.stats.mean)
            min_s = float(self.min_s_over_episode) if math.isfinite(self.min_s_over_episode) else float("nan")

            r_term = 0.0
            r_term += -self.w.w_var_final * (final_var / max(1e-8, self.sc.var_final))
            r_term += self.w.w_mean_final * math.tanh(mean_s / max(1e-8, self.sc.mean_scale))
            if self.w.w_min_final != 0.0 and math.isfinite(min_s):
                r_term += self.w.w_min_final * math.tanh(min_s / max(1e-8, self.sc.mean_scale))

            reward += r_term
            terminated = True

        info = {
            "ok": True,  # if it ends here, it's feasible
            "groups_done": len(self.group_scores),
            "within_group": len(self.current_group),
            "running_mean_s": float(self.stats.mean),
            "running_var_s": float(self.stats.variance()),
            "soft_sum": float(self.soft_sum),
        }
        if terminated:
            info.update(
                {
                    "final_var": float(self.stats.variance()),
                    "final_mean_s": float(self.stats.mean),
                    "min_s_over_episode": (
                        float(self.min_s_over_episode) if math.isfinite(self.min_s_over_episode) else float("nan")
                    ),
                    "soft_penalty_sum": float(-self.soft_sum),
                    "group_scores": list(self.group_scores),
                }
            )

        return self._get_obs(), self._clip(float(reward)), terminated, False, info



# =========================
# Minimal wiring example (you will adapt paths/modules)
# =========================

if __name__ == "__main__":
    # Example shows how you'd wire the env once your evaluator is ready.
    # Replace imports with your module names.
    from tree_evaluator import (
        TreeHydraulicEvaluator,
        load_nodes_xlsx,
        load_pipes_xlsx,
        build_lateral_ids_for_field_nodes,
        is_field_node_id,
    )

    # Load network
    nodes = load_nodes_xlsx("Nodes.xlsx")
    edges = load_pipes_xlsx("Pipes.xlsx")

    evaluator = TreeHydraulicEvaluator(nodes=nodes, edges=edges, root="J0", H0=25.0, Hmin=11.59)

    # Build laterals
    field_nodes = [nid for nid in nodes.keys() if is_field_node_id(nid)]
    lateral_ids, lateral_to_node = build_lateral_ids_for_field_nodes(field_nodes)

    # (Optional) compute single_margin_map by evaluating each lateral alone
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
        alpha_var_final=0.0,
        lambda_branch_soft=0.1,
        seed=0,
    )

    # Quick sanity check with random actions
    obs, _ = env.reset()
    done = False
    ep_r = 0.0
    while not done:
        mask = env.action_masks()
        valid = np.where(mask)[0]
        a = int(env.rng.choice(valid))
        obs, r, done, trunc, info = env.step(a)
        ep_r += r
    print("episode reward:", ep_r)
    print("groups done:", info.get("groups_done"))

    # For training with MaskablePPO (sb3-contrib), you'll do something like:
    # from sb3_contrib import MaskablePPO
    # model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    # model.learn(1_000_000)

