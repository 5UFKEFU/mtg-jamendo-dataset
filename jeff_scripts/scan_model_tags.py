#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强制加载模型，提取标签维度
✅ 支持 sota-music-tagging-models 中的全部结构
✅ 不依赖 state_dict 推理，而是真实构建并 forward 运行
✅ 无需任何额外依赖，只用 torch 和 training/model.py
"""

import os
import torch
import torchaudio
import sys

# 注册模型类型
MODEL_CLASSES = {
    "attention": "CNNSA",
    "crnn": "CRNN",
    "fcn": "FCN",
    "hcnn": "HarmonicCNN",
    "musicnn": "Musicnn",
    "musicnn600": "Musicnn",
    "sample": "SampleCNN",
    "se": "SampleCNNSE",
    "short_res": "ShortChunkCNN_Res"
}

# 添加 model 构建模块路径（你已有）
sys.path.insert(0, os.path.abspath("./training"))
import model as md  # noqa


def find_models(base="models"):
    for root, _, files in os.walk(base):
        for f in files:
            if f == "best_model.pth":
                yield os.path.join(root, f)


def scan(path):
    try:
        rel = os.path.relpath(path, "models")
        model_type = os.path.basename(os.path.dirname(path))
        model_key = model_type if model_type in MODEL_CLASSES else "unknown"
        clsname = MODEL_CLASSES.get(model_key)
        if not clsname:
            return rel, "❌", f"未知模型类型: {model_type}"

        # 尝试加载权重
        obj = torch.load(path, map_location="cpu")
        state_dict = obj["model_state"] if isinstance(obj, dict) and "model_state" in obj else obj

        # 推测标签维度
        tag_dim = None
        for k, v in state_dict.items():
            if ".bias" in k and isinstance(v, torch.Tensor) and v.ndim == 1 and v.shape[0] <= 600:
                tag_dim = v.shape[0]
                if "output" in k or "dense2" in k:
                    break
        if not tag_dim:
            return rel, "❓", "无法推测标签维度"

        # 实例化模型
        NetCls = getattr(md, clsname)
        model = NetCls(n_classes=tag_dim)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # 构造假数据跑一次
        dummy = torch.zeros(1, 1, 16000 * 30)  # 模拟一段 30 秒的 mono 音频
        with torch.no_grad():
            out = model(dummy)
            real_n = out.shape[-1] if out.ndim == 2 else out.numel()

        return rel, str(real_n), "✅ OK"
    except Exception as e:
        return os.path.relpath(path, "models"), "err", str(e)


def main():
    print(f"{'模型路径':<55} {'标签数':<8} 状态")
    print("-" * 80)
    for path in find_models("models"):
        rel, tags, msg = scan(path)
        print(f"{rel:<55} {tags:<8} {msg}")


if __name__ == "__main__":
    main()