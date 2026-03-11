"""
AlphaRouter 20 种子测试脚本。
用 20 个不同种子测试 best_model，汇总 Travel time / Delay / Waiting time (CAV+HV 与 仅 CAV)，
输出总表及 max/min/mean 统计。
"""
import sys
import os

# 注入 alpha_router 参数，确保加载正确模型
sys.argv = [sys.argv[0], "--algo", "alpha_router"]

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from start import exe_flow

if __name__ == "__main__":
    model_name = "0.5_alpha_router_0315/best_model"
    path = "./res/hangzhou/exe_2983.sumocfg"
    ratio = 0.5
    cav_seed = 8
    n_seeds = 20
    seeds = list(range(n_seeds))

    print("=" * 70)
    print(f"  AlphaRouter 测试 | 模型={model_name} | {n_seeds} 种子 | ratio={ratio}")
    print("=" * 70)

    results = []
    for i, seed in enumerate(seeds):
        print(f"\n--- 种子 {i+1}/{n_seeds}: seed={seed} ---")
        try:
            metrics = exe_flow(exe=True, name=model_name, path=path, ratio=ratio,
                              cav_seed=cav_seed, seed=str(seed))
            metrics["seed"] = seed
            results.append(metrics)
            print(f"  Travel(CAV+HV)={metrics['travel_time_cav_hv']:.2f}s  "
                  f"Delay(CAV+HV)={metrics['delay_cav_hv']:.2f}s  "
                  f"Waiting(CAV+HV)={metrics['waiting_time_cav_hv']:.2f}s")
            print(f"  Travel(CAV)={metrics['travel_time_cav']:.2f}s  "
                  f"Delay(CAV)={metrics['delay_cav']:.2f}s  "
                  f"Waiting(CAV)={metrics['waiting_time_cav']:.2f}s")
        except Exception as e:
            print(f"  种子 {seed} 失败: {e}")
            results.append({
                "seed": seed,
                "travel_time_cav_hv": np.nan, "delay_cav_hv": np.nan, "waiting_time_cav_hv": np.nan,
                "travel_time_cav": np.nan, "delay_cav": np.nan, "waiting_time_cav": np.nan,
            })

    df = pd.DataFrame(results)
    valid = df.dropna(how="all", subset=["travel_time_cav_hv"])

    # 汇总统计
    cols = ["travel_time_cav_hv", "delay_cav_hv", "waiting_time_cav_hv",
            "travel_time_cav", "delay_cav", "waiting_time_cav"]
    stats = {}
    for c in cols:
        stats[c] = {
            "mean": float(valid[c].mean()),
            "min": float(valid[c].min()),
            "max": float(valid[c].max()),
        }

    # 保存 20 次明细
    out_dir = "./simudata/0.5_alpha_router_0315"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/test_20seeds_detail.csv", index=False)
    print(f"\n20 次测试明细已保存: {out_dir}/test_20seeds_detail.csv")

    # 打印总表
    print("\n" + "=" * 70)
    print("  TABLE: Performance (20 seeds) - AlphaRouter")
    print("=" * 70)
    print("\n【20 次测试明细】")
    print(df[["seed"] + cols].to_string(index=False))

    print("\n【汇总统计 (mean / min / max)】")
    print("-" * 70)
    print(f"{'Metric':<25} {'CAV+HV (s)':>18} {'CAV only (s)':>18}")
    print("-" * 70)
    print(f"{'Travel time':<25} "
          f"mean={stats['travel_time_cav_hv']['mean']:>6.2f} min={stats['travel_time_cav_hv']['min']:>6.2f} max={stats['travel_time_cav_hv']['max']:>6.2f}   "
          f"mean={stats['travel_time_cav']['mean']:>6.2f} min={stats['travel_time_cav']['min']:>6.2f} max={stats['travel_time_cav']['max']:>6.2f}")
    print(f"{'Delay':<25} "
          f"mean={stats['delay_cav_hv']['mean']:>6.2f} min={stats['delay_cav_hv']['min']:>6.2f} max={stats['delay_cav_hv']['max']:>6.2f}   "
          f"mean={stats['delay_cav']['mean']:>6.2f} min={stats['delay_cav']['min']:>6.2f} max={stats['delay_cav']['max']:>6.2f}")
    print(f"{'Waiting time':<25} "
          f"mean={stats['waiting_time_cav_hv']['mean']:>6.2f} min={stats['waiting_time_cav_hv']['min']:>6.2f} max={stats['waiting_time_cav_hv']['max']:>6.2f}   "
          f"mean={stats['waiting_time_cav']['mean']:>6.2f} min={stats['waiting_time_cav']['min']:>6.2f} max={stats['waiting_time_cav']['max']:>6.2f}")
    print("-" * 70)

    # 保存汇总表
    summary = pd.DataFrame([
        {"Metric": "Travel time (s)", "CAV+HV_mean": stats["travel_time_cav_hv"]["mean"],
         "CAV+HV_min": stats["travel_time_cav_hv"]["min"], "CAV+HV_max": stats["travel_time_cav_hv"]["max"],
         "CAV_mean": stats["travel_time_cav"]["mean"], "CAV_min": stats["travel_time_cav"]["min"],
         "CAV_max": stats["travel_time_cav"]["max"]},
        {"Metric": "Delay (s)", "CAV+HV_mean": stats["delay_cav_hv"]["mean"],
         "CAV+HV_min": stats["delay_cav_hv"]["min"], "CAV+HV_max": stats["delay_cav_hv"]["max"],
         "CAV_mean": stats["delay_cav"]["mean"], "CAV_min": stats["delay_cav"]["min"],
         "CAV_max": stats["delay_cav"]["max"]},
        {"Metric": "Waiting time (s)", "CAV+HV_mean": stats["waiting_time_cav_hv"]["mean"],
         "CAV+HV_min": stats["waiting_time_cav_hv"]["min"], "CAV+HV_max": stats["waiting_time_cav_hv"]["max"],
         "CAV_mean": stats["waiting_time_cav"]["mean"], "CAV_min": stats["waiting_time_cav"]["min"],
         "CAV_max": stats["waiting_time_cav"]["max"]},
    ])
    summary.to_csv(f"{out_dir}/test_20seeds_summary.csv", index=False)
    print(f"\n汇总表已保存: {out_dir}/test_20seeds_summary.csv")
    print("\n测试完成。")
