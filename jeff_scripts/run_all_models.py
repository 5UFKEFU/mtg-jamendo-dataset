#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历 models/<dataset>/<model>/best_model.pth 批量预测

规则：
1. 若今天已存在 results_<dataset>_<model>_<YYYYMMDD>.csv → 自动跳过该模型
2. 默认只依赖当前目录：models/、batch_predict_csv.py
"""

import os, subprocess, sys, glob, argparse
from datetime import datetime

# ────────────────────────── CLI ──────────────────────────
def arg():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="jamendo | mtat | msd")
    p.add_argument("--audio_folder", default="my_music")
    p.add_argument("--script", default="batch_predict_csv.py",
                   help="单模型预测脚本路径")
    p.add_argument("--results_dir", default=".",
                   help="结果文件输出目录（默认当前目录）")
    p.add_argument("--model_root", default="models",
                   help="模型权重根目录（默认 ./models）")
    return p.parse_args()

# ────────────────────────── 主流程 ──────────────────────────
def main():
    a      = arg()
    here   = os.path.dirname(os.path.abspath(__file__))
    root   = os.path.join(here, a.model_root, a.dataset)
    models = sorted(os.path.basename(p)
                    for p in glob.glob(f"{root}/*")
                    if os.path.isdir(p))
    today  = datetime.now().strftime("%Y%m%d")

    if not models:
        print(f"⚠️  在 {root} 未找到任何模型目录，检查 --dataset / --model_root")
        sys.exit(1)

    print("将依次运行:", ", ".join(models))
    for m in models:
        out_csv = os.path.join(a.results_dir,
                               f"results_{a.dataset}_{m}_{today}.csv")
        if os.path.exists(out_csv):
            print(f"⏩ {m} 已有今日结果，跳过")
            continue

        cmd = [sys.executable, a.script,
               "--dataset", a.dataset,
               "--model",   m,
               "--audio_folder", a.audio_folder,
               "--model_root", root]          # 传递权重目录
        print("▶", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()