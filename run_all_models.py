#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历 models/<dataset>/<model>/best_model.pth 批量预测
如果今天已经生成过 results_<dataset>_<model>_<YYYYMMDD>.csv
则自动跳过该模型。
"""

import os, subprocess, sys, glob, argparse
from datetime import datetime

def arg():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="jamendo | mtat | msd")
    p.add_argument("--audio_folder", default="my_music")
    p.add_argument("--script", default="batch_predict_csv.py")
    p.add_argument("--results_dir", default=".", help="结果文件所在目录（默认当前目录）")
    return p.parse_args()

def main():
    a = arg()
    root = f"./models/{a.dataset}"
    models = [os.path.basename(p) for p in glob.glob(f"{root}/*") if os.path.isdir(p)]
    today  = datetime.now().strftime("%Y%m%d")

    print("将依次运行:", ", ".join(models))
    for m in models:
        out_csv = os.path.join(a.results_dir, f"results_{a.dataset}_{m}_{today}.csv")
        if os.path.exists(out_csv):
            print(f"⏩ {m} 已有今日结果，跳过")
            continue

        cmd = [sys.executable, a.script,
               "--dataset", a.dataset,
               "--model",   m,
               "--audio_folder", a.audio_folder]
        print("▶", " ".join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()