import torch
import json

# 第一次使用 torch.hub.load 会从 GitHub 下载模型文件和配置，需联网
model = torch.hub.load(
    repo_or_dir='microsoft/CNNMusicTagger',     # GitHub 仓库
    model='cnn_music_tagger',                   # 模型名
    trust_repo=True                             # 明确信任源（必要）
)

# 提取模型中包含的标签（共 183 个）
labels = model.tags

# 打印标签数量和部分内容确认
print(f"共提取标签 {len(labels)} 个，例如前5个：\n", labels[:183])

# 保存为本地 JSON 文件，方便后续查看或处理
with open('index_to_label.json', 'w') as f:
    json.dump(labels, f, indent=2, ensure_ascii=False)

print("✅ 标签保存成功 index_to_label.json")