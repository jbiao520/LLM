# 视觉编码器（Vision Encoder）深入版

> 面向有 ML 基础读者的视觉编码器深度指南

## 1. Vision Transformer (ViT)

### 1.1 Patch Embedding

将图像 $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ 切成 $N$ 个 Patch：

$$\mathbf{X}_p = [\mathbf{x}_p^1, \mathbf{x}_p^2, ..., \mathbf{x}_p^N]$$

其中 $N = HW / P^2$，$P$ 是 Patch 大小。

线性投影：

$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_p^1 \mathbf{E}; ...; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}$$

其中 $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 是投影矩阵。

### 1.2 ViT vs CNN

| 特性 | ViT | CNN |
|------|-----|-----|
| 归纳偏置 | 弱（需大量数据） | 强（平移不变性） |
| 全局感受野 | 第一层就有 | 需要堆叠 |
| 计算复杂度 | O(N²) | O(N) |
| 数据需求 | 大 | 相对较小 |

### 1.3 位置编码

**可学习位置编码：**
$$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$$

**2D 正弦位置编码：**
$$PE_{(x,y,2i)} = \sin(x / 10000^{2i/D})$$
$$PE_{(x,y,2i+1)} = \cos(y / 10000^{2i/D})$$

## 2. CLIP

### 2.1 对比学习目标

给定 Batch 中 $N$ 个 (图像, 文本) 对，优化：

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\mathbf{v}_i \cdot \mathbf{t}_i / \tau)}{\sum_{j=1}^{N} \exp(\mathbf{v}_i \cdot \mathbf{t}_j / \tau)}$$

其中：
- $\mathbf{v}_i$：图像 embedding
- $\mathbf{t}_i$：文本 embedding
- $\tau$：温度参数

### 2.2 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLIP 架构                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   图像编码器                        文本编码器                   │
│   ┌─────────────────┐              ┌─────────────────┐         │
│   │ ViT-L/14        │              │ Transformer     │         │
│   │ 或 ResNet-50    │              │ (12层, 512宽)   │         │
│   └────────┬────────┘              └────────┬────────┘         │
│            │                                │                   │
│            ▼                                ▼                   │
│   ┌─────────────────┐              ┌─────────────────┐         │
│   │ Image Encoder   │              │ Text Encoder    │         │
│   │ (projection)    │              │ (projection)    │         │
│   └────────┬────────┘              └────────┬────────┘         │
│            │                                │                   │
│            ▼                                ▼                   │
│   ┌─────────────────────────────────────────────────┐          │
│   │              Shared Embedding Space              │          │
│   │                   (512-d)                        │          │
│   └─────────────────────────────────────────────────┘          │
│                              │                                  │
│                              ▼                                  │
│                    Contrastive Loss                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 零样本推理

```python
def zero_shot_classify(image, class_names):
    # 编码图像
    image_features = image_encoder(image)

    # 编码所有类别名称
    text_features = text_encoder([f"a photo of {c}" for c in class_names])

    # 计算相似度
    logits = image_features @ text_features.T

    # 返回最相似的类别
    return class_names[logits.argmax()]
```

## 3. 高级视觉编码器

### 3.1 Swin Transformer

层次化设计，滑动窗口注意力：

```
Stage 1: Patch 4×4, 96-d, 窗口 7×7
Stage 2: 下采样 2×, 192-d
Stage 3: 下采样 2×, 384-d
Stage 4: 下采样 2×, 768-d
```

窗口内注意力：复杂度 O(M²) vs 全局 O(N²)

### 3.2 EVA-CLIP

更大更强的 CLIP：
- ViT-bigE (1.0B 参数)
- 训练数据: LAION-2B
- 更好的零样本性能

### 3.3 SigLIP

Sigmoid Loss 替代 Softmax：

$$\mathcal{L} = -\frac{1}{|B|} \sum_{i,j} \log \sigma(y_{ij} \cdot \mathbf{v}_i \cdot \mathbf{t}_j)$$

其中 $y_{ij} \in \{-1, +1\}$ 表示是否配对。

**优势：** 不需要全局归一化，更好的批量扩展性。

## 4. 特征提取技巧

### 4.1 多尺度特征

```python
def extract_multiscale_features(image, model):
    features = []
    for scale in [224, 336, 448]:
        resized = resize(image, scale)
        feat = model.encode_image(resized)
        features.append(feat)

    return concat(features)
```

### 4.2 注意力图提取

```python
def get_attention_map(model, image, layer=-1):
    # 获取注意力权重
    attentions = model.get_attention_weights(image)[layer]

    # 取 [CLS] token 对其他 token 的注意力
    cls_attention = attentions[0, 0, 1:]  # [num_patches]

    # reshape 成 2D
    size = int(math.sqrt(cls_attention.shape[0]))
    attention_map = cls_attention.reshape(size, size)

    return attention_map
```

## 5. 评估基准

| 模型 | ImageNet Top-1 | COCO Caption | VQA |
|------|----------------|--------------|-----|
| CLIP ViT-B/32 | 63.2% | - | - |
| CLIP ViT-L/14 | 75.3% | - | - |
| EVA-CLIP | 78.5% | - | - |
| BLIP-2 | - | 143.8 CIDEr | 82.3% |

## 参考文献

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) - ViT
- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) - CLIP
- [Swin Transformer: Hierarchical Vision Transformer](https://arxiv.org/abs/2103.14030)
- [BLIP-2: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597)
