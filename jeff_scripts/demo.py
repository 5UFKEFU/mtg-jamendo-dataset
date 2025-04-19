import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import os
import sys
id_to_label = [
    ("classical", "古典"), ("electronic", "电子"), ("experimental", "实验"), ("folk", "民谣"), ("hip-hop", "嘻哈"),
    ("instrumental", "器乐"), ("international", "世界音乐"), ("pop", "流行"), ("rock", "摇滚"), ("jazz", "爵士"),

    ("happy", "快乐"), ("sad", "悲伤"), ("angry", "愤怒"), ("relaxed", "放松"), ("romantic", "浪漫"),
    ("epic", "史诗"), ("dark", "黑暗"), ("fun", "有趣"), ("calm", "平静"),

    ("acoustic guitar", "木吉他"), ("electric guitar", "电吉他"), ("piano", "钢琴"), ("synthesizer", "合成器"),
    ("violin", "小提琴"), ("drums", "鼓"), ("bass", "贝斯"), ("vocals", "人声"),
    ("male vocal", "男声"), ("female vocal", "女声"),

    ("choral", "合唱"), ("spoken word", "口语"), ("a cappella", "无伴奏合唱"), ("beatboxing", "口技"),
    ("autotune", "电音修音"), ("shouting", "喊叫"), ("whistling", "口哨"),

    ("party", "派对"), ("travel", "旅行"), ("sleep", "睡眠"), ("study", "学习"), ("workout", "健身"),
    ("meditation", "冥想"), ("commercial", "商业"), ("children", "儿童"), ("holiday", "假日"), ("wedding", "婚礼"),

    ("vintage", "复古"), ("modern", "现代"), ("retro", "怀旧"), ("ambient", "氛围"), ("cinematic", "电影感"), ("lo-fi", "低保真")
]

def inspect_model_weights(pth_file):
    state_dict = torch.load(pth_file, map_location='cpu')
    print(f"✅ 模型包含 {len(state_dict)} 个参数层：\n")
    for name, param in state_dict.items():
        print(f"{name:35} {tuple(param.shape)}")

def auto_convert_wav(input_path, output_path="fixed.wav"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} not found.")
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        output_path
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg 转换音频失败，请确保已正确安装。")
    return output_path

def preprocess_audio(audio_path, target_sr=16000, duration=10):
    try:
        waveform, sr = torchaudio.load(audio_path)
    except RuntimeError as e:
        print("❌ torchaudio 无法加载音频。请尝试安装依赖：")
        print("   👉 yum install libsndfile")
        print("   👉 pip install soundfile")
        raise e

    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

    expected_len = target_sr * duration
    if waveform.shape[1] < expected_len:
        pad_len = expected_len - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :expected_len]
    return waveform

class CNNMusicTagger(nn.Module):
    def __init__(self, n_classes=56):
        super().__init__()
        self.bn_init = nn.BatchNorm2d(1)

        self.conv_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(64)

        self.conv_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(128)

        self.conv_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_3 = nn.BatchNorm2d(128)

        self.conv_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)

        self.conv_5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn_5 = nn.BatchNorm2d(64)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self._wav_to_spectrogram(x)
        x = self.bn_init(x)

        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))

        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        x = self.dense(x)
        return x

    def _wav_to_spectrogram(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]
        spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )(x.squeeze(1))  # [B, mel, time]
        spec = torchaudio.transforms.AmplitudeToDB()(spec)
        return spec.unsqueeze(1)  # [B, 1, mel, time]

def load_model(model_path):
    model = CNNMusicTagger(n_classes=56)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_tags(audio_path, model_path, top_k=5):
    print(f"🎧 正在处理音频: {audio_path}")
    fixed_path = auto_convert_wav(audio_path)
    waveform = preprocess_audio(fixed_path)
    model = load_model(model_path)
    with torch.no_grad():
        logits = model(waveform)
        probs = torch.sigmoid(logits).squeeze()
        top_indices = torch.topk(probs, k=top_k).indices.tolist()
        return [{
    "tag": id_to_label[i][0],
    "zh": id_to_label[i][1],
    "probability": round(probs[i].item(), 3)
} for i in top_indices]

# 命令行或默认运行
if __name__ == '__main__':
    try:
        result = predict_tags("audio.mp3", "scripts/baseline/models/best_model.pth")
        print("✅ 标签预测结果：")
        for tag in result:
            print(f" - {tag['tag']}（{tag['zh']}）: {tag['probability']}")
    except Exception as e:
        print("❌ 执行失败：", e)
        sys.exit(1)