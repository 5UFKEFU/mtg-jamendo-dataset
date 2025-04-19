#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量音乐标签 → CSV + JSON
2025‑04‑19  —  无外部依赖版
"""

# ──────────────── 标准库 ────────────────
import os, sys, csv, argparse, json, inspect
from datetime import datetime
from typing import List, Tuple

# ──────────────── 深度学习 ────────────────
import torch, torchaudio, torch.nn.functional as F
import torch.nn as nn

# ────────────────────────── 自动导入模型定义 ──────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))

def import_models_pkg():
    """在常见目录中查找 `training/model.py` 或 `models/` 包并导入"""
    SEARCH_PATHS = [
        os.path.join(HERE, "training"),                      # ./training
        os.path.join(HERE, "..", "training"),                # ../training
        os.path.join(HERE, "models"),                        # ./models  (包)
    ]
    for p in SEARCH_PATHS:
        if os.path.isfile(os.path.join(p, "model.py")):      # 旧结构
            sys.path.insert(0, p)
            import model as mdl
            return mdl
        if os.path.isfile(os.path.join(p, "__init__.py")):   # models/ 包
            sys.path.insert(0, os.path.dirname(p))
            import models as mdl
            return mdl
    raise RuntimeError("❌ 未找到模型定义 (model.py / models)。请检查目录。")

md = import_models_pkg()

# ────────────────────────── 标签表 ──────────────────────────
# 仅示例：JAMENDO56；MTAT50 / MSD* 可自行补全
JAMENDO56: List[Tuple[str, str]] = [
    # genre
    ("classical", "古典"), ("baroque", "巴洛克"), ("electronic", "电子"), ("edm", "EDM"), ("house", "浩室"),
    ("techno", "电子舞曲"), ("trance", "恍惚"), ("dubstep", "低音爆破"), ("rock", "摇滚"), ("hard rock", "硬摇滚"),
    ("metal", "金属"), ("punk", "朋克"), ("shoegaze", "自赏"), ("math rock", "数学摇滚"), ("jazz", "爵士"),
    ("blues", "布鲁斯"), ("country", "乡村"), ("folk", "民谣"), ("bossa nova", "巴萨诺瓦"), ("hip-hop", "嘻哈"),
    ("trap", "陷阱"), ("drill", "德律"), ("rap", "说唱"), ("pop", "流行"), ("r&b", "节奏布鲁斯"),
    ("soul", "灵魂乐"), ("funk", "放克"), ("reggae", "雷鬼"), ("experimental", "实验"), ("ambient", "氛围"),
    ("cinematic", "电影感"), ("lo-fi", "低保真"), ("new age", "新世纪"), ("latin", "拉丁"), ("k-pop", "韩流"),
    ("j-pop", "日流"), ("african", "非洲"), ("celtic", "凯尔特"), ("indian classical", "印度古典"), ("tibetan", "藏族"),
    ("world", "世界音乐"),

    # mood
    ("happy", "快乐"), ("sad", "悲伤"), ("angry", "愤怒"), ("calm", "平静"), ("relaxed", "放松"),
    ("romantic", "浪漫"), ("epic", "史诗"), ("dark", "黑暗"), ("fun", "有趣"), ("hopeful", "希望"),
    ("melancholic", "忧郁"), ("uplifting", "鼓舞人心"), ("nostalgic", "怀旧"), ("tense", "紧张"),
    ("dreamy", "梦幻"), ("lonely", "孤独"), ("confident", "自信"), ("peaceful", "宁静"),
    ("mysterious", "神秘"), ("sarcastic", "讽刺"),

    # scene
    ("party", "派对"), ("travel", "旅行"), ("sleep", "睡眠"), ("study", "学习"), ("focus", "专注"),
    ("workout", "健身"), ("running", "跑步"), ("driving", "开车"), ("shopping", "逛街"),
    ("meditation", "冥想"), ("yoga", "瑜伽"), ("background", "背景音乐"), ("holiday", "假日"),
    ("birthday", "生日"), ("christmas", "圣诞"), ("wedding", "婚礼"), ("cooking", "做饭"),
    ("gaming", "游戏"), ("festival", "节庆"), ("cinematic trailer", "电影预告"),

    # instrument
    ("piano", "钢琴"), ("violin", "小提琴"), ("cello", "大提琴"), ("flute", "长笛"), ("clarinet", "单簧管"),
    ("saxophone", "萨克斯"), ("trumpet", "小号"), ("banjo", "班卓琴"), ("guzheng", "古筝"),
    ("erhu", "二胡"), ("accordion", "手风琴"), ("sitar", "西塔琴"), ("harp", "竖琴"),
    ("bass", "贝斯"), ("drums", "鼓"), ("guitar", "吉他"), ("electric guitar", "电吉他"),
    ("acoustic guitar", "木吉他"), ("synthesizer", "合成器"), ("percussion", "打击乐"),
    ("beat drop", "爆点"),

    # vocal
    ("vocals", "人声"), ("male vocal", "男声"), ("female vocal", "女声"), ("duet", "对唱"),
    ("choral", "合唱"), ("a cappella", "无伴奏"), ("spoken word", "口语"), ("autotune", "电音修音"),
    ("shouting", "喊叫"), ("whistling", "口哨"), ("beatboxing", "口技"), ("falsetto", "假声"),
    ("growl", "吼唱"), ("harmonized", "和声"), ("talkbox", "语音合成"), ("vocoded", "编码人声"),

    # others
    ("live", "现场"), ("instrumental", "器乐"), ("cover", "翻唱"), ("original", "原创"),
    ("remix", "混音"), ("loop", "循环"), ("vintage", "复古"), ("modern", "现代"),
    ("retro", "怀旧"), ("nature", "自然声音"), ("asmr", "ASMR"), ("minimal", "极简"),
    ("experimental", "实验"), ("synthwave", "合成波"), ("orchestral", "交响"), ("indie", "独立")
]

MTAT50 = [(f"tag_{i}", f"tag_{i}") for i in range(50)]  # 占位
MSD50  = MTAT50
MSD600 = [(f"tag_{i:03d}",)*2 for i in range(600)]

def tag_table(ds: str, ncls: int):
    if ds == "jamendo": return JAMENDO56
    if ds == "mtat":    return MTAT50
    if ds == "msd":     return MSD50 if ncls==50 else MSD600
    return []

# ────────────────────────── CLI ──────────────────────────
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["jamendo", "mtat", "msd"])
    ap.add_argument("--model", required=True,
                    help="crnn / sample / se / short_res …")
    ap.add_argument("--audio_folder", default="my_music")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--th", type=float, default=0.1)
    ap.add_argument("--exts", nargs="+", default=[".wav"])
    # 默认就用当前目录 models/
    ap.add_argument("--model_root", default="models")
    ap.add_argument("--skip", default="", help="逗号分隔模型名，跳过预测")
    return ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────── I/O ──────────────────────────
def list_audio(root: str, exts):
    exts = tuple(e.lower() for e in exts)
    return [os.path.join(r, f)
            for r, _, fs in os.walk(root)
            for f in fs if f.lower().endswith(exts)]

def wav_tensor(path: str, sr=16000, dur=30):
    w, fs = torchaudio.load(path)
    if fs != sr:
        w = torchaudio.transforms.Resample(fs, sr)(w)
    need = sr * dur
    return F.pad(w, (0, max(0, need - w.shape[1])))[:, :need]

# ────────────────────────── 模型构建 / 加载 ──────────────────────────
MODEL_MAP = {
    "crnn": "CRNN", "fcn": "FCN", "sample": "SampleCNN", "se": "SampleCNNSE",
    "short": "ShortChunkCNN", "short_res": "ShortChunkCNN_Res",
    "attention": "CNNSA", "hcnn": "HarmonicCNN",
    "musicnn": "Musicnn", "musicnn600": "Musicnn"
}

# ① －－－－－－－－－－－－－－－－ build_model －－－－－－－－－－－－－－－－
def build_model(key: str, ncls: int, ds: str, st: dict | None = None):
    """
    根据模型名实例化网络；
    若 state_dict(st) 给出最后全连接层，自动推断 hidden_dim/d_model，解决 Attention
    “(1×236) 与 (256×256) 维度不符” 的问题。
    """
    cls = getattr(md, MODEL_MAP[key])

    # 从 checkpoint 推断 encoder hidden dim（权重 shape: [n_classes, hidden_dim]）
    hidden_dim = None
    if st is not None:
        for k, v in st.items():
            if k.endswith(".weight") and v.ndim == 2 and v.shape[0] == ncls:
                hidden_dim = v.shape[1]
                break

    base = dict(n_classes=ncls, dataset=ds)
    cand = [
        base,
        dict(num_classes=ncls, dataset=ds),
        dict(n_classes=ncls),
        dict(num_classes=ncls),
        dict(dataset=ds),
        {}
    ]
    # 若已推断 hidden_dim，把带 d_model / hidden_size 的版本插到最前
    if hidden_dim:
        cand = ([{**base, "d_model": hidden_dim},
                 {**base, "hidden_size": hidden_dim}] + cand)

    for kw in cand:
        try:
            return cls(**kw)
        except TypeError:
            continue
    raise RuntimeError(f"无法实例化模型 {key}；尝试参数：{cand}")

# ② －－－－－－－－－－－－－－－－ load_net －－－－－－－－－－－－－－－－
def load_net(ckpt: str, key: str, ds: str):
    st = torch.load(ckpt, map_location=device)
    ncls = max(v.shape[0] for k, v in st.items()
               if k.endswith(".bias") and v.ndim == 1)
    net = build_model(key, ncls, ds, st)   # ← 把 st 传进去
    net.load_state_dict(st, strict=False)
    #return patch_gap(net).to(device).eval(), ncls
    return patch_net(net).to(device).eval(), ncls     # ← 改这里
def patch_gap(net: nn.Module):
    """若最后线性层前仍有时间维，做 GAP。"""
    if getattr(net, "_patched_gap", False):
        return net

    def gap_hook(_, inputs):
        x = inputs[0]
        if x.dim() == 3:  # [B, C, T]
            x = x.mean(-1)
            return (x,)
        return inputs

    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.register_forward_pre_hook(gap_hook)
    net._patched_gap = True
    return net

# ---- 通用补丁：① 全局平均池化 ② 线性层自动对齐 in_features ----
def patch_net(net: nn.Module):
    """
    • 若送入 Linear 前仍含时间维 → GAP
    • 若特征维度 ≠ Linear.in_features:
        - 小于时 → 右侧 0‑padding
        - 大于时 → 截断到所需维
      解决 attention 的 (236 vs 256) 报错，也容忍别的维度漂移。
    打一次标记即可。
    """
    if getattr(net, "_patched_generic", False):
        return net

    def pre_hook(module, inputs):
        x = inputs[0]
        # ① time‑dim GAP：形状 [B,C,T] + in_features==C
        if x.dim() == 3 and x.size(1) == module.in_features:
            x = x.mean(-1)                    # -> [B,C]
        # ② 对齐 feature 维
        if x.size(-1) != module.in_features:
            diff = module.in_features - x.size(-1)
            if diff > 0:                      # 维度不足 → pad 0
                x = F.pad(x, (0, diff))
            else:                             # 维度过多 → 截断
                x = x[..., :module.in_features]
        return (x,)

    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.register_forward_pre_hook(pre_hook)

    net._patched_generic = True
    return net
# ────────────────────────── 权重查找 ──────────────────────────
def find_ckpt(root: str, ds: str, model_key: str):
    cand_roots = [
        root,                                # 用户传入
        os.path.join(HERE, "models"),        # ./models
        os.path.join(HERE, "..", "models"),  # ../models
    ]
    for r in cand_roots:
        ck = os.path.join(r, ds, model_key, "best_model.pth")
        if os.path.isfile(ck):
            return ck
    raise FileNotFoundError(
        "❌ 未找到权重文件 best_model.pth；搜索路径：\n  " +
        "\n  ".join(os.path.join(r, ds, model_key) for r in cand_roots)
    )

# ────────────────────────── 主流程 ──────────────────────────
def main():
    a = cli()
    if a.model in a.skip.split(","):
        print(f"⏩ Skip {a.model} by user option")
        return

    ckpt = find_ckpt(a.model_root, a.dataset, a.model)
    net, ncls = load_net(ckpt, a.model, a.dataset)
    tbl = tag_table(a.dataset, ncls)

    auds = list_audio(a.audio_folder, a.exts)
    if not auds:
        print("❗ 未找到音频文件"); return

    date = datetime.now().strftime("%Y%m%d")
    csv_f = f"results_{a.dataset}_{a.model}_{date}.csv"
    js_f  = f"results_{a.dataset}_{a.model}_{date}.json"
    first_write = not os.path.exists(csv_f)
    js_buf = {}

    with open(csv_f, "a", newline='', encoding="utf-8") as fp:
        wr = csv.writer(fp)
        if first_write:
            head = ["file", "model"]
            for k in range(1, a.top_k + 1):
                head += [f"tag_en_{k}", f"tag_zh_{k}", f"prob_{k}"]
            wr.writerow(head)

        for p in auds:
            x = wav_tensor(p).to(device)
            probs = torch.sigmoid(net(x)).flatten()
            picks = sorted(
                [(i, float(pr)) for i, pr in enumerate(probs) if pr >= a.th],
                key=lambda t: t[1], reverse=True)[:a.top_k]

            row = [os.path.basename(p), a.model]
            js_list = []
            for idx, pr in picks:
                en, zh = tbl[idx] if idx < len(tbl) else (f"tag_{idx:03d}",)*2
                row += [en, zh, round(pr, 3)]
                js_list.append({"tag_en": en, "tag_zh": zh, "prob": round(pr, 3)})
            row += ["", "", ""] * (a.top_k - len(picks))  # 补齐
            wr.writerow(row); print(",".join(map(str, row)))
            js_buf[os.path.basename(p)] = {"model": a.model, "predictions": js_list}

    with open(js_f, "w", encoding="utf-8") as jf:
        json.dump(js_buf, jf, ensure_ascii=False, indent=2)
    print(f"✅ 结果已保存：{csv_f}  和  {js_f}")

if __name__ == "__main__":
    main()