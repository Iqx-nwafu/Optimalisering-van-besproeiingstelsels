# tune_reward_weights_optuna.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from train_maskable_ppo import make_vec_env, make_env
from ppo_env import RewardWeights, RewardScales, SafetyShaping

try:
    import optuna
except Exception as e:
    raise SystemExit("Optuna not installed. Run: pip install optuna") from e


def evaluate(model: MaskablePPO, env, n_episodes: int = 5):
    vals, means = [], []
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
    return float(np.nanmean(vals)), float(np.nanmean(means))


def objective(trial: optuna.Trial) -> float:
    w_var_final = trial.suggest_float("w_var_final", 0.4, 0.9)
    w_mean_final = 1.0 - w_var_final

    w_var_step = trial.suggest_float("w_var_step", 0.3, 0.8)
    w_safe_step = trial.suggest_float("w_safe_step", 0.1, 0.6)
    w_mean_step = trial.suggest_float("w_mean_step", 0.0, 0.2)

    s_safe = trial.suggest_float("s_safe", 0.5, 3.0)

    reward_weights = RewardWeights(
        w_var_step=w_var_step,
        w_mean_step=w_mean_step,
        w_safe_step=w_safe_step,
        w_var_final=w_var_final,
        w_mean_final=w_mean_final,
        w_min_final=0.0,
    )
    reward_scales = RewardScales(var_step=1.0, mean_step=1.0, var_final=1.0, mean_scale=10.0)
    safety_shaping = SafetyShaping(s_safe=s_safe, mode="linear", tau=0.5)

    num_envs = 6
    timesteps = 200_000

    train_env = make_vec_env(
        num_envs=num_envs,
        start_seed=trial.number * 10_000,
        reward_weights=reward_weights,
        reward_scales=reward_scales,
        safety_shaping=safety_shaping,
        include_single_margin=True,
    )
    eval_env = make_env(
        seed=trial.number * 10_000 + 999,
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
        learning_rate=3e-4,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
    )
    model.learn(total_timesteps=timesteps)

    final_var, mean_s = evaluate(model, eval_env, n_episodes=5)

    s_target = 14.0
    k_safe = 1.0
    score = final_var + k_safe * max(0.0, s_target - mean_s)

    trial.set_user_attr("final_var", final_var)
    trial.set_user_attr("mean_s", mean_s)
    return score


if __name__ == "__main__":
    out = Path("./weight_search")
    out.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "attrs": study.best_trial.user_attrs,
    }
    (out / "optuna_best.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    print(best)
