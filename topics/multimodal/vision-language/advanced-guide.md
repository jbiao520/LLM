# 视觉语言模型深入版

> 面向有 ML 基础读者的视觉语言模型深度指南

## 1. 架构设计

### 1.1 视觉-语言对齐

**目标：** 将视觉特征 $\mathbf{V}$ 映射到语言空间 $\mathbf{L}$

$$\mathbf{H}_v = \text{Projector}(\text{VisionEncoder}(\mathbf{I}))$$

其中 $\mathbf{H}_v \in \mathbb{R}^{N_v \times d}$，$N_v$ 是视觉 token 数量。

### 1.2 投影层设计

**线性投影：**
$$\mathbf{H}_v = \mathbf{W} \cdot \mathbf{V} + \mathbf{b}$$

**MLP 投影 (LLaVA)：**
$$\mathbf{H}_v = \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{V}) \cdot \mathbf{W}_2$$

**Q-Former (BLIP-2)：**
```python
class QFormer(nn.Module):
    def __init__(self, num_queries=32, hidden_dim=768):
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.cross_attention = CrossAttention(hidden_dim)
        self.self_attention = nn.TransformerEncoder(...)

    def forward(self, image_features):
        # Query 从图像特征中提取信息
        q = self.queries.expand(image_features.size(0), -1, -1)
        q = self.cross_attention(q, image_features)
        q = self.self_attention(q)
        return q
```

### 1.3 视觉 Token 压缩

| 方法 | Token 数量 | 信息保留 |
|------|------------|----------|
| 直接投影 | 196-256 | 最高 |
| Q-Former | 32-64 | 高 |
| Perceiver Resampler | 64 | 高 |
| Pixel Shuffle | 49-64 | 中 |

## 2. LLaVA 详解

### 2.1 模型架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLaVA-1.5 架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   视觉编码器: CLIP ViT-L/336                                    │
│   - 输入: 336×336 图像                                          │
│   - 输出: 576 (24×24) 个 Patch，每个 1024 维                   │
│   - 使用 penultimate layer 特征                                │
│                                                                 │
│   投影层: 2 层 MLP                                              │
│   - 输入: 1024 维                                               │
│   - 输出: 4096 维 (Vicuna hidden size)                         │
│                                                                 │
│   LLM: Vicuna-7B/13B                                           │
│   - 标准 Transformer decoder                                   │
│   - 处理视觉 Token + 文本 Token                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 输入格式

```
输入序列:
┌─────────────────────────────────────────────────────────────────┐
│ <image_features>                                                │
│ [0.1, 0.2, ...] × 576 个视觉 Token                              │
│ </image_features>                                               │
│ USER: 描述这张图片                                              │
│ ASSISTANT:                                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     LLM 生成回答
```

### 2.3 训练策略

**Stage 1: 预训练（冻结 LLM）**
- 只训练投影层
- 数据: 558K 图像描述对
- 目标: 对齐视觉-语言表示

**Stage 2: 指令微调（全参数）**
- 训练投影层 + LLM
- 数据: 150K 指令数据
- 目标: 提升指令遵循能力

## 3. BLIP-2 详解

### 3.1 Q-Former 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Q-Former                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   可学习 Query: $\mathbf{Q} \in \mathbb{R}^{N_q \times d}$     │
│                                                                 │
│   层结构:                                                       │
│   1. Self-Attention (Query 之间)                               │
│   2. Cross-Attention (Query ← 图像特征)                        │
│   3. FFN                                                        │
│                                                                 │
│   训练目标:                                                     │
│   - Image-Text Matching (ITM)                                  │
│   - Image-Text Contrastive (ITC)                               │
│   - Language Modeling (LM)                                     │
│                                                                 │
│   输出: 32 个压缩的视觉 Token                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 两阶段训练

**Stage 1: Bootstrap Vision-Language**
- 冻结视觉编码器
- 训练 Q-Former
- 学习视觉-语言对齐

**Stage 2: Bootstrap LLM**
- 冻结视觉编码器 + Q-Former
- 训练 LLM（通过 Q-Former 输出）
- 学习生成能力

## 4. 评估基准

| 模型 | VQAv2 | GQA | TextVQA | COCO Caption |
|------|-------|-----|---------|--------------|
| LLaVA-1.5 (7B) | 78.5 | 62.0 | 58.2 | - |
| LLaVA-1.5 (13B) | 80.0 | 63.3 | 61.3 | - |
| BLIP-2 (7B) | - | - | - | 143.8 CIDEr |
| GPT-4V | 77.4* | 64.5* | 78.0* | - |

*为报告分数，非官方评测

## 5. 高级技术

### 5.1 动态分辨率 (LLaVA-Next)

处理不同分辨率图像：

```python
def process_dynamic_resolution(image, max_patches=6):
    patches = []
    for scale in [1, 1.5, 2]:
        resized = resize_image(image, scale)
        patch = vision_encoder(resized)
        patches.append(patch)

    # 选择最优 patch 组合
    selected = select_best_patches(patches, max_patches)
    return selected
```

### 5.2 多图理解

```
输入序列:
[图1 Token] [图2 Token] ... 用户: 比较这两张图...
```

### 5.3 视频理解

```python
def encode_video(video, fps=1):
    frames = sample_frames(video, fps)
    frame_features = [vision_encoder(f) for f in frames]

    # 时序建模
    video_features = temporal_modeling(frame_features)
    return video_features
```

## 6. 开源模型对比

| 模型 | 视觉编码器 | LLM | 特点 |
|------|-----------|-----|------|
| LLaVA-1.5 | CLIP ViT-L | Vicuna | 简单有效 |
| InternVL | InternViT | Qwen | 国产领先 |
| Qwen-VL | ViT-bigG | Qwen | 多语言 |
| CogVLM | EVA-CLIP | Vicuna | 深度融合 |
| Yi-VL | ViT-H | Yi | 中文优化 |

## 参考文献

- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485)
- [BLIP-2](https://arxiv.org/abs/2301.12597)
- [Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)](https://arxiv.org/abs/2310.03744)
- [InternVL](https://arxiv.org/abs/2312.14238)
