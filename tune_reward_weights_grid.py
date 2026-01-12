# tune_reward_weights_grid.py
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from train_maskable_ppo import make_vec_env, make_env
from ppo_env import RewardWeights, RewardScales, SafetyShaping


def evaluate(model: MaskablePPO, env, n_episodes: int = 10):
    vals, means, mins = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        info_last = {}
        while not done:
            mask = env.unwrapped.action_masks()
            a, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, done, _, info_last = env.step(int(a))
        vals.append(float(info_last.get("final_var", np.nan)))
        means.append(float(info_last.get("final_mean_s", np.nan)))
        mins.append(float(info_last.get("min_s_over_episode", np.nan)))
    return float(np.nanmean(vals)), float(np.nanmean(means)), float(np.nanmean(mins))


if __name__ == "__main__":
    out = Path("./weight_search")
    out.mkdir(parents=True, exist_ok=True)

    num_envs = 6
    timesteps_per_cfg = 250_000
    n_eval_eps = 10

    reward_scales = RewardScales(var_step=1.0, mean_step=1.0, var_final=1.0, mean_scale=10.0)
    safety_shaping = SafetyShaping(s_safe=1.0, mode="linear", tau=0.5)

    # Small grid (expand carefully)
    w_var_f_list = [0.6, 0.7, 0.8]
    w_mean_f_list = [0.2, 0.3, 0.4]
    w_safe_step_list = [0.2, 0.3, 0.4]
    w_var_step_list = [0.5, 0.6, 0.7]
    w_mean_step = 0.1

    # Composite score: minimize variance + safety shortfall
    s_target = 14.0
    k_safe = 1.0

    best = None
    recs = []
    cfg_id = 0

    for w_var_f, w_mean_f, w_safe_s, w_var_s in itertools.product(
        w_var_f_list, w_mean_f_list, w_safe_step_list, w_var_step_list
    ):
        # normalize final weights to sum=1
        s = w_var_f + w_mean_f
        w_var_f_n = w_var_f / s
        w_mean_f_n = w_mean_f / s

        reward_weights = RewardWeights(
            w_var_step=w_var_s,
            w_mean_step=w_mean_step,
            w_safe_step=w_safe_s,
            w_var_final=w_var_f_n,
            w_mean_final=w_mean_f_n,
            w_min_final=0.0,
        )

        train_env = make_vec_env(
            num_envs=num_envs,
            start_seed=cfg_id * 10_000,
            reward_weights=reward_weights,
            reward_scales=reward_scales,
            safety_shaping=safety_shaping,
            include_single_margin=True,
        )
        eval_env = make_env(
            seed=cfg_id * 10_000 + 999,
            reward_weights=reward_weights,
            reward_scales=reward_scales,
            safety_shaping=safety_shaping,
            include_single_margin=True,
        )

        model = MaskablePPO(
            "MultiInputPolicy",
            train_env,
            verbose=0,
            device="cpu",
            n_steps=256,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            ent_coef=0.01,
            clip_range=0.2,
        )
        model.learn(total_timesteps=timesteps_per_cfg)

        final_var, mean_s, min_s = evaluate(model, eval_env, n_episodes=n_eval_eps)
        score = final_var + k_safe * max(0.0, s_target - mean_s)

        rec = {
            "cfg_id": cfg_id,
            "weights": reward_weights.__dict__,
            "final_var": final_var,
            "mean_s": mean_s,
            "min_s": min_s,
            "score": score,
        }
        recs.append(rec)

        if best is None or score < best["score"]:
            best = rec

        (out / "grid_results.json").write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
        cfg_id += 1

    print("BEST:", best)
