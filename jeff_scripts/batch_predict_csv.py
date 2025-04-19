#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量音乐标签 → CSV + JSON   (一次性解决已知所有报错)
"""

# -------- 标准库 --------
import os, sys, csv, argparse, json, inspect
from datetime import datetime

# -------- 深度学习 ----------
import torch, torchaudio, torch.nn.functional as F
import torch.nn as nn

# ────────────────────────── 动态导入官方模型 ──────────────────────────
REPO = os.path.abspath("../sota-music-tagging-models")
sys.path.insert(0, os.path.join(REPO, "training"))
import model as md

# ────────────────────────── 内置标签表 ──────────────────────────
JAMENDO56 = [  # (英文, 中文)
 ("classical","古典"),("electronic","电子"),("experimental","实验"),("folk","民谣"),
 ("hip-hop","嘻哈"),("instrumental","器乐"),("international","世界音乐"),("pop","流行"),
 ("rock","摇滚"),("jazz","爵士"),("happy","快乐"),("sad","悲伤"),("angry","愤怒"),
 ("relaxed","放松"),("romantic","浪漫"),("epic","史诗"),("dark","黑暗"),("fun","有趣"),
 ("calm","平静"),("acoustic guitar","木吉他"),("electric guitar","电吉他"),("piano","钢琴"),
 ("synthesizer","合成器"),("violin","小提琴"),("drums","鼓"),("bass","贝斯"),
 ("vocals","人声"),("male vocal","男声"),("female vocal","女声"),("choral","合唱"),
 ("spoken word","口语"),("a cappella","无伴奏合唱"),("beatboxing","口技"),
 ("autotune","电音修音"),("shouting","喊叫"),("whistling","口哨"),("party","派对"),
 ("travel","旅行"),("sleep","睡眠"),("study","学习"),("workout","健身"),
 ("meditation","冥想"),("commercial","商业"),("children","儿童"),("holiday","假日"),
 ("wedding","婚礼"),("vintage","复古"),("modern","现代"),("retro","怀旧"),
 ("ambient","氛围"),("cinematic","电影感"),("lo-fi","低保真")
]
MTAT50 = [...  ]   # 同前，省略
MSD50 = MTAT50
MSD600 = [f"tag_{i:03d}" for i in range(600)]

# ────────────────────────── CLI ──────────────────────────
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["jamendo","mtat","msd"])
    ap.add_argument("--model", required=True)                 # crnn, sample, se …
    ap.add_argument("--audio_folder", default="my_music")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--th", type=float, default=0.1)
    ap.add_argument("--exts", nargs="+", default=[".wav"])
    ap.add_argument("--model_root", default="../sota-music-tagging-models/models")
    ap.add_argument("--skip", default="", help="逗号分隔模型名列表，预测时跳过")
    return ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────── 标签表选择 ──────────────────────────
def tag_table(ds, ncls):
    if ds == "jamendo": return JAMENDO56
    if ds == "mtat":    return [(t,t) for t in MTAT50]
    if ds == "msd":
        return [(t,t) for t in (MSD50 if ncls==50 else MSD600)]
    return []

# ────────────────────────── I/O 工具 ──────────────────────────
def list_audio(root, exts):
    exts = tuple(e.lower() for e in exts)
    return [os.path.join(r,f) for r,_,fs in os.walk(root)
            for f in fs if f.lower().endswith(exts)]

def wav_tensor(path, sr=16000, dur=30):
    w,fs = torchaudio.load(path)
    if fs!=sr: w = torchaudio.transforms.Resample(fs, sr)(w)
    need = sr*dur
    return F.pad(w,(0,max(0,need-w.shape[1])))[:, :need]

# ────────────────────────── 构造 + 加载模型 ──────────────────────────
MODEL_MAP = {
 "crnn":"CRNN","fcn":"FCN","sample":"SampleCNN","se":"SampleCNNSE",
 "short":"ShortChunkCNN","short_res":"ShortChunkCNN_Res",
 "attention":"CNNSA","hcnn":"HarmonicCNN",
 "musicnn":"Musicnn","musicnn600":"Musicnn"
}
def build_model(key, ncls, ds):
    cls = getattr(md, MODEL_MAP[key])
    for kw in (dict(n_classes=ncls,dataset=ds),dict(num_classes=ncls,dataset=ds),
               dict(n_classes=ncls),dict(num_classes=ncls),
               dict(dataset=ds),{}):
        try: return cls(**kw)
        except TypeError: pass
    raise RuntimeError(f"无法实例化 {key}")

# ---- 通用 GAP 补丁：在线性层入口做 1D 全局平均池化 ----
def patch_gap(net):
    if getattr(net, "_patched_gap", False):
        return net

    def gap_hook(module, inputs):
        x = inputs[0]
        if x.dim() == 3:                  # [B, C, T]
            x = x.mean(-1)                # [B, C]
            return (x,)                   # 必须返回 tuple
        return inputs                     # 其它形状照旧

    # 给所有 nn.Linear 注册 hook
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.register_forward_pre_hook(gap_hook)

    net._patched_gap = True
    return net
    # 收集卷积/子块：遍历 forward graph 不易，这里笨法——
    # 把原 forward 存起来，手动加 GAP 前钩子
    orig_forward = net.forward

    def new_forward(x, *args, **kwargs):
        y = orig_forward(x, *args, **kwargs)
        if y.dim() == 3 and y.size(1) == dense_layer.in_features:
            y = y.mean(-1)                          # GAP
            return dense_layer(y)
        return y

    # 用 inspect 判断是否已包裹
    if not inspect.isfunction(getattr(net,'_orig_forward',None)):
        net._orig_forward = orig_forward
        net.forward = new_forward
        net._patched_gap = True
    return net

def load_net(ckpt, key, ds):
    st = torch.load(ckpt, map_location=device)
    ncls = max(v.shape[0] for k,v in st.items()
               if k.endswith(".bias") and v.ndim==1)
    net = build_model(key, ncls, ds)
    net.load_state_dict(st, strict=False)
    net = patch_gap(net)
    return net.to(device).eval(), ncls

# ────────────────────────── 主流程 ──────────────────────────
def main():
    a = cli()
    if a.model in a.skip.split(","):
        print(f"⏩ Skip {a.model} by user option"); return

    ckpt = os.path.join(a.model_root, a.dataset, a.model, "best_model.pth")
    net, ncls = load_net(ckpt, a.model, a.dataset)
    tbl = tag_table(a.dataset, ncls)

    auds = list_audio(a.audio_folder, a.exts)
    if not auds: print("❗ No audio found"); return

    date = datetime.now().strftime("%Y%m%d")
    csv_f = f"results_{a.dataset}_{a.model}_{date}.csv"
    js_f  = f"results_{a.dataset}_{a.model}_{date}.json"
    first = not os.path.exists(csv_f)
    js_buf = {}

    with open(csv_f, "a", newline='', encoding="utf-8") as fp:
        wr = csv.writer(fp)
        if first:
            head = ["file","model"]
            for k in range(1,a.top_k+1):
                head += [f"tag_en_{k}",f"tag_zh_{k}",f"prob_{k}"]
            wr.writerow(head)

        for p in auds:
            probs = torch.sigmoid(net(wav_tensor(p).to(device))).flatten()
            picks = sorted([(i,float(pr)) for i,pr in enumerate(probs) if pr>=a.th],
                           key=lambda t:t[1], reverse=True)[:a.top_k]

            row=[os.path.basename(p), a.model]; js_list=[]
            for idx, pr in picks:
                en, zh = tbl[idx] if idx < len(tbl) else (f"tag_{idx:03d}",)*2
                row += [en, zh, round(pr,3)]
                js_list.append({"tag_en":en,"tag_zh":zh,"prob":round(pr,3)})
            row += ["","",""]*(a.top_k-len(picks))
            wr.writerow(row); print(",".join(map(str,row)))
            js_buf[os.path.basename(p)] = {"model":a.model,"predictions":js_list}

    with open(js_f,"w",encoding="utf-8") as jf:
        json.dump(js_buf,jf,ensure_ascii=False,indent=2)
    print(f"✅ CSV: {csv_f}  JSON: {js_f}")

if __name__=="__main__":
    main()