"""
DSO (Dynamic System Optimal) 独立训练入口。
复用 start.py 中的 main() 和 exe_flow()，避免与其他正在运行的训练冲突。

Usage:
    python start_dso.py                        # 训练 + 测试
    python start_dso.py --train_only           # 仅训练
    python start_dso.py --test_only            # 仅测试
    python start_dso.py --ratio 0.3            # 指定 CAV 渗透率
"""
import sys
import os
import argparse

_real_argv = sys.argv[:]

_launcher = argparse.ArgumentParser(add_help=False)
_launcher.add_argument("--train_only", action="store_true")
_launcher.add_argument("--test_only", action="store_true")
_launcher.add_argument("--ratio", type=float, default=0.5)
_launcher.add_argument("--cav_seed", type=int, default=8)
_launch_args, _ = _launcher.parse_known_args(_real_argv[1:])

sys.argv = ["start_dso.py", "--algo", "dso"]

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from start import main, exe_flow


if __name__ == "__main__":
    torch.manual_seed(config.SIMULATION["SEED"])
    np.random.seed(config.SIMULATION["SEED"])

    rt = _launch_args.ratio
    cav_seed = _launch_args.cav_seed
    name = None

    if not _launch_args.test_only:
        print("=" * 60)
        print(f"  DSO Training | ratio={rt}")
        print("=" * 60)
        name = main(exe=False, comment="", light_path=None, ratio=rt)
        print(f"训练完成，模型名: {name}")

    if not _launch_args.train_only:
        if name is None:
            name = f"{rt}_dso_0315"
        fname = name + "/best_model"
        print("=" * 60)
        print(f"  DSO Testing | model={fname} | ratio={rt} | cav_seed={cav_seed}")
        print("=" * 60)
        exe_flow(exe=True, name=fname, path="./res/hangzhou/exe_2983.sumocfg",
                 ratio=rt, cav_seed=cav_seed)
        print("测试完成")
