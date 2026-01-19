# eval_one.py
import json
import heapq
from pathlib import Path
from datetime import datetime

import numpy as np
from sb3_contrib import MaskablePPO

# 你训练时的 env 构造函数
from comparison_utils import ComparisonConfig, composite_score
from train_maskable_ppo import make_env  # make_env 在这里定义并返回 Monitor(env) :contentReference[oaicite:4]{index=4}


def run_episode(model, env, deterministic: bool):
    obs, _ = env.reset()
    done = False
    ep_r = 0.0
    actions = []
    info_last = {}

    while not done:
        # 单环境 + Monitor 包装时，稳妥写法：
        mask = env.unwrapped.action_masks()
        a, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
        a = int(a)
        actions.append(a)

        obs, r, done, _, info = env.step(a)
        ep_r += float(r)
        info_last = info

    return ep_r, actions, info_last


def actions_to_groups(actions, lateral_ids, group_size=4):
    """把 120 个 action index 映射为 30 组×4条支管的 lateral_id 列表。"""
    assert len(actions) % group_size == 0
    G = len(actions) // group_size
    groups = []
    for g in range(G):
        idxs = actions[group_size * g: group_size * g + group_size]
        groups.append([lateral_ids[i] for i in idxs])
    return groups


def push_topk_minvar(heap, rec, k, counter):
    """
    维护 top-k (final_var 最小)。
    用最小堆实现“弹出最差”，所以 key 取 (-final_var)，这样 heap[0] 是最差(最大 final_var)。
    """
    item = (-rec["final_var"], counter, rec)  # counter 防止 final_var 相同导致比较 rec
    heapq.heappush(heap, item)
    if len(heap) > k:
        heapq.heappop(heap)


def heap_to_sorted_records(heap):
    """把 heap 转成按 final_var 升序排列的 record 列表。"""
    recs = [it[2] for it in heap]
    recs.sort(key=lambda r: r["final_var"])
    return recs


if __name__ == "__main__":
    # -------- 你关心的两个参数：采样次数 & 保存的 top-k
    N_SAMPLES = 5000
    TOPK = 30

    # 输出目录
    out_dir = Path("./sample_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 构造环境 + 载入模型
    config = ComparisonConfig()
    env = make_env(seed=1, config=config)
    model = MaskablePPO.load("ppo_irrigation_masked")

    # 先跑一条 deterministic 基线（便于对照）
    r_det, actions_det, info_det = run_episode(model, env, deterministic=True)
    print("Deterministic reward:", r_det)
    final_var = float(info_det.get("final_var", 0.0))
    min_margin = float(info_det.get("min_s_over_episode", 0.0))
    print("final_var:", final_var)
    print("final_mean_s:", info_det.get("final_mean_s"))
    print("min_s_over_episode:", min_margin)
    print("soft_sum:", info_det.get("soft_sum"))
    print("composite_score:", composite_score(final_var, min_margin))

    lateral_ids = env.unwrapped.lateral_ids

    # top-k 容器 + 统计
    top_heap = []
    counter = 0
    n_feasible = 0
    n_infeasible = 0
    best = None  # final_var 最小者

    for k in range(N_SAMPLES):
        r, actions, info = run_episode(model, env, deterministic=False)

        fv = info.get("final_var", None)
        if fv is None:
            # 理论上 episode 正常终止都会带 final_var；防御性处理
            continue

        # 跳过硬约束失败的样本（info 里会标 ok=False）:contentReference[oaicite:5]{index=5}
        if info.get("ok", True) is False:
            n_infeasible += 1
            continue

        n_feasible += 1

        rec = {
            "sample_id": int(k),
            "final_var": float(fv),
            "reward": float(r),
            "final_mean_s": float(info.get("final_mean_s", float("nan"))),
            "min_s_over_episode": float(info.get("min_s_over_episode", float("nan"))),
            # ppo_env 末尾会给 soft_penalty_sum；如果没有就用 -soft_sum 兜底 :contentReference[oaicite:6]{index=6}
            "soft_penalty_sum": float(info.get("soft_penalty_sum", -float(info.get("soft_sum", 0.0)))),
            "actions": [int(a) for a in actions],
            "groups": actions_to_groups(actions, lateral_ids, group_size=4),
        }

        # 可选：保存每组 s_g 序列（环境里叫 group_scores）:contentReference[oaicite:7]{index=7}
        if "group_scores" in info:
            rec["group_scores"] = [float(x) for x in info["group_scores"]]

        # 更新 best（最终解）
        if (best is None) or (rec["final_var"] < best["final_var"]):
            best = rec

        # 更新 top-k
        push_topk_minvar(top_heap, rec, TOPK, counter)
        counter += 1

        # 进度打印（可按需调频）
        if (k + 1) % 200 == 0:
            cur_best = best["final_var"] if best is not None else None
            print(f"[{k+1}/{N_SAMPLES}] feasible={n_feasible} infeasible={n_infeasible} best_final_var={cur_best}")

    # 排序输出 top-k
    top_records = heap_to_sorted_records(top_heap)

    # -------- 落盘
    # 1) 最优解
    best_path = out_dir / f"best_min_final_var_{tag}.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    # 2) top-20
    top_path = out_dir / f"top{TOPK}_min_final_var_{tag}.json"
    with top_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "N_SAMPLES": int(N_SAMPLES),
                "TOPK": int(TOPK),
                "feasible": int(n_feasible),
                "infeasible": int(n_infeasible),
                "top": top_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== DONE =====")
    print("Feasible:", n_feasible, "Infeasible:", n_infeasible)
    print("BEST(final_var):", best["final_var"], "reward:", best["reward"])
    print("Saved:", str(best_path))
    print("Saved:", str(top_path))

    # 控制台展示 top-20 概览
    print("\nTop candidates (by final_var):")
    for i, rec in enumerate(top_records, 1):
        print(
            f"{i:02d}  final_var={rec['final_var']:.6f}  "
            f"min_s={rec['min_s_over_episode']:.6f}  mean_s={rec['final_mean_s']:.6f}  "
            f"soft_penalty_sum={rec['soft_penalty_sum']:.6f}  sample_id={rec['sample_id']}"
        )
