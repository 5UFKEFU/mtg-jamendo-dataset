#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量音乐自动标签脚本（多模型可选版）

示例：
    python batch_predict.py --dataset jamendo --model musicnn --audio_folder my_music
"""

# ───────────────────────────── 依赖 ──────────────────────────────
import os, sys, json, argparse, subprocess
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tabulate import tabulate

# ───────────────────────────── 动态导入官方模型 ──────────────────────────────
REPO_PATH = os.path.abspath("../sota-music-tagging-models")     # ← 按实际路径调整
TRAIN_PATH = os.path.join(REPO_PATH, "training")
if TRAIN_PATH not in sys.path:
    sys.path.insert(0, TRAIN_PATH)

try:
    import model as md        # 官方所有模型类都在 training/model.py
except Exception as e:
    raise RuntimeError(
        f"无法导入官方模型：{e}\n"
        "请确认 REPO_PATH 指向正确，并已执行\n"
        "  pip install -e ../sota-music-tagging-models"
    )

# ───────────────────────────── CLI 参数 ──────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser("batch music tagging")
    ap.add_argument("--dataset", required=True, help="jamendo | mtat | msd")
    ap.add_argument("--model",   required=True,
                    help="attention | crnn | fcn | hcnn | musicnn | musicnn600 | sample | se | short | short_res")
    ap.add_argument("--audio_folder", default="my_music")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--exts", nargs="+", default=[".wav"], help=".wav .mp3 …")
    ap.add_argument("--model_root", default="../sota-music-tagging-models/models")
    return ap.parse_args()

# ───────────────────────────── 设备 ──────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ───────────────────────────── 标签（Jamendo‑56） ──────────────────────────────
id_to_label = [
    ("classical", "古典"), ("electronic", "电子"), ("experimental", "实验"), ("folk", "民谣"),
    ("hip-hop", "嘻哈"), ("instrumental", "器乐"), ("international", "世界音乐"), ("pop", "流行"),
    ("rock", "摇滚"), ("jazz", "爵士"), ("happy", "快乐"), ("sad", "悲伤"), ("angry", "愤怒"),
    ("relaxed", "放松"), ("romantic", "浪漫"), ("epic", "史诗"), ("dark", "黑暗"), ("fun", "有趣"),
    ("calm", "平静"), ("acoustic guitar", "木吉他"), ("electric guitar", "电吉他"), ("piano", "钢琴"),
    ("synthesizer", "合成器"), ("violin", "小提琴"), ("drums", "鼓"), ("bass", "贝斯"),
    ("vocals", "人声"), ("male vocal", "男声"), ("female vocal", "女声"), ("choral", "合唱"),
    ("spoken word", "口语"), ("a cappella", "无伴奏合唱"), ("beatboxing", "口技"),
    ("autotune", "电音修音"), ("shouting", "喊叫"), ("whistling", "口哨"), ("party", "派对"),
    ("travel", "旅行"), ("sleep", "睡眠"), ("study", "学习"), ("workout", "健身"),
    ("meditation", "冥想"), ("commercial", "商业"), ("children", "儿童"), ("holiday", "假日"),
    ("wedding", "婚礼"), ("vintage", "复古"), ("modern", "现代"), ("retro", "怀旧"),
    ("ambient", "氛围"), ("cinematic", "电影感"), ("lo-fi", "低保真")
]

tag_groups = {
    "genre": ["classical", "electronic", "experimental", "folk", "hip-hop",
              "instrumental", "international", "pop", "rock", "jazz"],
    "mood": ["happy", "sad", "angry", "relaxed", "romantic", "epic", "dark", "fun", "calm"],
    "instrument": ["acoustic guitar", "electric guitar", "piano", "synthesizer",
                   "violin", "drums", "bass", "vocals", "male vocal", "female vocal"],
    "vocal style": ["choral", "spoken word", "a cappella", "beatboxing",
                    "autotune", "shouting", "whistling"],
    "scene": ["party", "travel", "sleep", "study", "workout", "meditation",
              "commercial", "children", "holiday", "wedding"],
    "style": ["vintage", "modern", "retro", "ambient", "cinematic", "lo-fi"]
}

# ───────────────────────────── 音频预处理 ──────────────────────────────
def preprocess_audio(path, sr_out=16000, dur=30):
    wav, sr = torchaudio.load(path)
    if sr != sr_out:
        wav = torchaudio.transforms.Resample(sr, sr_out)(wav)
    need = sr_out * dur
    if wav.shape[1] < need:
        wav = F.pad(wav, (0, need - wav.shape[1]))
    else:
        wav = wav[:, :need]
    return wav

# ───────────────────────────── 实例化模型 ──────────────────────────────
MODEL_NAMES = {
    "crnn":       "CRNN",
    "fcn":        "FCN",
    "sample":     "SampleCNN",
    "se":         "SampleCNN_SE",      # 源码使用下划线
    "short":      "ShortChunkCNN",
    "short_res":  "ShortChunkCNN_Res",
    "attention":  "CNNSA",
    "hcnn":       "HarmonicCNN",
    "musicnn":    "Musicnn",
    "musicnn600": "Musicnn",
}

def build_model(model_key, n_classes, dataset):
    if model_key not in MODEL_NAMES:
        raise ValueError(f"未知模型: {model_key}")

    cls_name = MODEL_NAMES[model_key]
    mdl_cls = getattr(md, cls_name)

    # 各模型构造函数略有不同，做一个 try‑chain
    for kwargs in (
        dict(n_classes=n_classes, dataset=dataset),
        dict(num_classes=n_classes, dataset=dataset),
        dict(n_classes=n_classes),
        dict(num_classes=n_classes),
        dict(dataset=dataset),
        {}
    ):
        try:
            return mdl_cls(**kwargs)
        except TypeError:
            continue
    raise RuntimeError(f"无法实例化 {cls_name}")

def load_model(weight_path, model_key, dataset):
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(weight_path)
    state = torch.load(weight_path, map_location=device)

    # 用最后一个全连接的 bias 推断类别数
    n_classes = max(v.shape[0] for k, v in state.items() if k.endswith(".bias") and v.ndim == 1)
    net = build_model(model_key, n_classes, dataset)

    # 某些模型保存了额外的 buffers（如 Mel 滤波bank）
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print("⚠️  State missing:", missing)
    if unexpected:
        print("⚠️  State unexpected:", unexpected)

    if torch.cuda.device_count() > 1:
        print(f"🚀  使用 {torch.cuda.device_count()} 张 GPU 并行")
        net = nn.DataParallel(net)
    return net.to(device).eval()

# ───────────────────────────── 预测 ──────────────────────────────
def predict_one(path, net, top_k, th):
    wav = preprocess_audio(path).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(net(wav)).squeeze().cpu()

    preds = []
    for i, p in enumerate(prob.tolist()):
        if p < th:
            continue
        if i < len(id_to_label):
            en, zh = id_to_label[i]
        else:               # 600 类时无中文
            en, zh = f"tag_{i}", f"tag_{i}"
        preds.append({"tag": en, "zh": zh, "prob": round(p, 3)})
    preds.sort(key=lambda x: x["prob"], reverse=True)
    return preds[:top_k]

def group_preds(preds):
    grp = {g: [] for g in tag_groups}
    grp["other"] = []
    for p in preds:
        for g, lst in tag_groups.items():
            if p["tag"].lower() in [t.lower() for t in lst]:
                grp[g].append(p); break
        else:
            grp["other"].append(p)
    return grp

# ───────────────────────────── 辅助 ──────────────────────────────
def find_audio(root, exts):
    exts = tuple(e.lower() for e in exts)
    return [os.path.join(r, f)
            for r, _, fs in os.walk(root)
            for f in fs if f.lower().endswith(exts)]

# ───────────────────────────── 主程序 ──────────────────────────────
def main():
    args = parse_args()
    ckpt = os.path.join(args.model_root, args.dataset, args.model, "best_model.pth")
    net  = load_model(ckpt, args.model, args.dataset)

    files = find_audio(args.audio_folder, args.exts)
    if not files:
        print("⚠️  未找到音频文件"); return
    print(f"🎧  共 {len(files)} 首音频，开始分析 …")

    results = {}
    for f in files:
        try:
            ps = predict_one(f, net, args.top_k, args.threshold)
            gp = group_preds(ps); results[f] = gp
            rows = [[g, p["tag"], p["zh"], p["prob"]] for g, ts in gp.items() for p in ts]
            print(f"\n🎵 {os.path.basename(f)}")
            print(tabulate(rows, headers=["类别", "EN", "中文", "概率"], tablefmt="grid") if rows else "  <无标签>")
        except Exception as e:
            print(f"❌  {f}: {e}")

    out = f"results_{args.dataset}_{args.model}_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"\n✅  完成，结果保存在 {out}")

if __name__ == "__main__":
    main()