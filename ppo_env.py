"""Gymnasium environment for PPO-style training on the irrigation grouping problem.

Goal
----
One episode constructs 30 irrigation groups by selecting 120 laterals without replacement:
- Action at each step: choose one lateral index in [0, 119].
- Every 4 selections forms one group; we run fast tree hydraulics to compute feasibility.
- Hard constraint: if any group violates Hmin at opened nodes, the episode terminates immediately
  with a large negative reward.
- Main objective (your choice): minimize variance of group minimum margins s_g across 30 groups.

Implementation notes
--------------------
- Uses action masking (recommended): sb3-contrib MaskablePPO.
  The env exposes action_masks() returning a bool mask of valid actions.
- Observation is a Dict:
    obs['feat']   : (N, F) per-lateral features (static + dynamic mask as a feature)
    obs['global'] : (G,) global features (progress, within-group position, running mean/var)

You must have the evaluator code (TreeHydraulicEvaluator) available.
This env expects:
- nodes, edges from your Excel
- lateral_to_node mapping

You can keep this file separate and import from your evaluator module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import math
import numpy as np

import gymnasium as gym
from gymnasium import spaces

# Import your fast evaluator from the previous canvas file.
# If you saved that file as `tree_evaluator.py`, do:
# from tree_evaluator import (
#     TreeHydraulicEvaluator, load_nodes_xlsx, load_pipes_xlsx,
#     build_lateral_ids_for_field_nodes, is_field_node_id
# )


# =========================
# Utilities
# =========================

def _node_num(nid: str) -> int:
    nid = nid.strip().upper()
    if not nid.startswith('J'):
        return -1
    try:
        return int(nid[1:])
    except ValueError:
        return -1


def branch_id_from_field_node(nid: str) -> int:
    """Your numbering pattern:
    - J11..J16 -> branch 1
    - J21..J26 -> branch 2
    ...
    - J91..J96 -> branch 9
    - J101..J106 -> branch 10
    """
    n = _node_num(nid)
    if n >= 101:
        return 10
    return n // 10


def pos_in_branch_from_field_node(nid: str) -> int:
    """J11->1 ... J16->6; J101->1 ... J106->6."""
    n = _node_num(nid)
    return n % 10


@dataclass
class WelfordState:
    n: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def variance(self) -> float:
        if self.n <= 1:
            return 0.0
        return self.M2 / (self.n - 1)

    def update(self, x: float) -> Tuple[float, float]:
        """Update with new sample x.

        Returns
        -------
        (old_var, new_var)
        """
        old_var = self.variance()
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        new_var = self.variance()
        return old_var, new_var


# =========================
# Environment
# =========================

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
        # Reward weights
        beta_infeasible: float = 1e4,     # strong penalty for any deficit
        alpha_var_final: float = 1.0,     # final variance penalty
        lambda_branch_soft: float = 0.1,  # soft penalty for branch repetition within a group
        # Feature toggles
        include_single_margin: bool = True,
        single_margin_map: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        assert len(lateral_ids) > 0
        self.evaluator = evaluator
        self.lateral_ids = list(lateral_ids)
        self.N = len(self.lateral_ids)
        self.lateral_to_node = dict(lateral_to_node)
        self.Hmin = float(Hmin)
        self.q_lateral = float(q_lateral)

        self.beta_infeasible = float(beta_infeasible)
        self.alpha_var_final = float(alpha_var_final)
        self.lambda_branch_soft = float(lambda_branch_soft)

        self.include_single_margin = bool(include_single_margin)
        self.single_margin_map = single_margin_map or {}

        self.rng = np.random.default_rng(seed)

        # Precompute static per-lateral features
        self._feat_static, self._branch_ids = self._build_static_features()
        self.F_static = self._feat_static.shape[1]

        # Observation spaces
        # Add 1 dynamic feature: selected_mask (0/1)
        self.F = self.F_static + 1
        self.observation_space = spaces.Dict(
            {
                "feat": spaces.Box(low=-np.inf, high=np.inf, shape=(self.N, self.F), dtype=np.float32),
                "global": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            }
        )

        # Action: pick a lateral index
        self.action_space = spaces.Discrete(self.N)

        # Internal episode state
        self.selected = np.zeros(self.N, dtype=bool)
        self.current_group: List[int] = []  # indices
        self.group_scores: List[float] = []
        self.stats = WelfordState()
        self.step_count = 0

        # ---- NEW: episode-level logging accumulators
        self.soft_sum = 0.0                      # sum of soft rewards (negative if penalties)
        self.min_s_over_episode = float("inf")   # min of s_g across feasible groups

    def _build_static_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build (N, F_static) features.

        Features (recommended minimal set):
        - z (elevation of the field node)
        - branch_id (1..10)
        - pos_in_branch (1..6)
        - side (L=0, R=1)
        - (optional) single_margin (when opening just this lateral)

        Note: normalize/standardize in your policy pipeline if needed.
        """
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
                sm = float(self.single_margin_map.get(lid, 0.0))
                row.append(sm)

            feats.append(row)
            branch_ids.append(int(b))

        feat_arr = np.asarray(feats, dtype=np.float32)
        branch_arr = np.asarray(branch_ids, dtype=np.int32)
        return feat_arr, branch_arr

    # ---- MaskablePPO compatibility (sb3-contrib)
    def action_masks(self) -> np.ndarray:
        """Valid actions are those not selected yet."""
        return (~self.selected).astype(bool)

    # ---- Gym API
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
        # per-lateral dynamic feature: selected flag
        sel = self.selected.astype(np.float32).reshape(self.N, 1)
        feat = np.concatenate([self._feat_static, sel], axis=1).astype(np.float32)

        g_index = len(self.group_scores)  # completed groups
        within = len(self.current_group)  # 0..3

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

    def step(self, action: int):
        # Enforce validity
        if action < 0 or action >= self.N or self.selected[action]:
            # invalid action -> strong negative reward and terminate
            obs = self._get_obs()
            return obs, -self.beta_infeasible, True, False, {
                "reason": "invalid_action",
                "groups_done": len(self.group_scores),
                "within_group": len(self.current_group),
                "running_mean_s": float(self.stats.mean),
                "running_var_s": float(self.stats.variance()),
                "soft_sum": float(getattr(self, "soft_sum", 0.0)),
            }

        # Apply action
        self.selected[action] = True
        self.current_group.append(action)
        self.step_count += 1

        reward = 0.0
        terminated = False

        # If group completed (4 picks), evaluate hydraulics
        if len(self.current_group) == 4:
            group_lids = [self.lateral_ids[i] for i in self.current_group]

            # Soft penalty: branch repetition within the group
            branch_counts: Dict[int, int] = {}
            for i in self.current_group:
                b = int(self._branch_ids[i])
                branch_counts[b] = branch_counts.get(b, 0) + 1
            soft = -self.lambda_branch_soft * sum(max(0, c - 1) for c in branch_counts.values())

            # ---- NEW: accumulate soft penalties/rewards
            self.soft_sum += float(soft)

            res = self.evaluator.evaluate_group(group_lids, lateral_to_node=self.lateral_to_node, q_lateral=self.q_lateral)

            if not res.ok:
                # Hard constraint violated: terminate immediately
                # res.min_margin is negative
                deficit = max(0.0, -float(res.min_margin))
                reward = -(self.beta_infeasible * deficit)
                terminated = True
                info = {
                    "ok": False,
                    "min_margin": float(res.min_margin),
                    "min_pressure_head": float(res.min_pressure_head),
                    "soft_penalty": float(soft),
                    "groups_done": len(self.group_scores),
                    "within_group": len(self.current_group),
                    "running_mean_s": float(self.stats.mean),
                    "running_var_s": float(self.stats.variance()),
                    "soft_sum": float(self.soft_sum),
                    "final_var": float(self.stats.variance()),
                    "final_mean_s": float(self.stats.mean),
                    "min_s_over_episode": (
                        float(self.min_s_over_episode) if math.isfinite(self.min_s_over_episode) else float("nan")
                    ),
                }
                return self._get_obs(), reward + soft, terminated, False, info

            # Feasible: update variance stats with s_g (min_margin)
            s_g = float(res.min_margin)
            old_v, new_v = self.stats.update(s_g)

            # Dense reward: reduce variance increment
            reward = -(new_v - old_v) + soft

            self.group_scores.append(s_g)

            # ---- NEW: track worst (minimum) feasible group margin over the episode
            self.min_s_over_episode = min(self.min_s_over_episode, float(s_g))
            self.current_group = []

        # Episode end
        if self.step_count == 120 and not terminated:
            # Final variance penalty (no deficits because we terminate on violation)
            final_var = float(self.stats.variance())
            reward += -self.alpha_var_final * final_var
            terminated = True

        obs = self._get_obs()
        info = {
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
                        float(self.min_s_over_episode)
                        if math.isfinite(self.min_s_over_episode)
                        else float("nan")
                    ),
                    "soft_penalty_sum": float(-self.soft_sum),
                    "group_scores": list(self.group_scores),
                }
            )
        return obs, float(reward), terminated, False, info


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

