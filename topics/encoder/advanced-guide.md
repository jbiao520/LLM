# 编码器（Encoder）深入版

> 面向有机器学习基础读者的技术详解

## 概述

Transformer 编码器是一种双向序列建模架构，通过自注意力机制捕获序列中所有位置之间的依赖关系。本文深入分析编码器的结构、数学原理及其在 BERT 等模型中的应用。

## Transformer Encoder 结构

### 单层结构

每层编码器包含两个子层：

1. **Multi-Head Self-Attention**
2. **Position-wise Feed-Forward Network**

每个子层后都有残差连接和层归一化：

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

### 完整公式

对于输入序列 $X \in \mathbb{R}^{n \times d}$：

**子层 1：Multi-Head Self-Attention**
$$Z = \text{LayerNorm}(X + \text{MultiHead}(X, X, X))$$

**子层 2：Feed-Forward Network**
$$Y = \text{LayerNorm}(Z + \text{FFN}(Z))$$

其中 FFN 定义为：
$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

## Pre-LN vs Post-LN

### Post-LN（原始 Transformer）

```
x = LayerNorm(x + Sublayer(x))
```

- 梯度在深层网络中可能消失
- 需要 warmup
- 训练不够稳定

### Pre-LN（现代实现）

```
x = x + Sublayer(LayerNorm(x))
```

- 梯度流动更平滑
- 不需要 warmup
- 训练更稳定

### 梯度分析

Pre-LN 的梯度可以直接通过残差路径流向任何层：

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} + \sum_{i=l}^{L-1} \frac{\partial L}{\partial f_i}$$

第一项保证梯度不会消失。

## 位置编码

由于自注意力没有顺序概念，需要位置编码：

### Sinusoidal（原始）

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

### Learnable（BERT）

直接学习位置嵌入：$PE \in \mathbb{R}^{max\_len \times d}$

### RoPE（现代 LLM）

旋转位置嵌入，结合了相对位置信息。

## BERT 架构

### 模型结构

BERT 是一个多层的 Transformer 编码器：

- **BERT-Base**：12 层，768 维，12 头，110M 参数
- **BERT-Large**：24 层，1024 维，16 头，340M 参数

### 输入表示

$$\text{Input} = \text{Token Embedding} + \text{Segment Embedding} + \text{Position Embedding}$$

- **Token Embedding**：词/子词嵌入
- **Segment Embedding**：句子标识（用于句子对任务）
- **Position Embedding**：位置信息

### 特殊 Token

- `[CLS]`：序列开头，用于分类任务
- `[SEP]`：句子分隔符
- `[MASK]`：预训练时遮盖的 token

### 预训练任务

#### Masked Language Model (MLM)

随机遮盖 15% 的 token，预测被遮盖的词：

$$P(mask\_token | context) = \text{softmax}(h \cdot E^T)$$

其中 $h$ 是 `[MASK]` 位置的隐藏状态，$E$ 是词嵌入矩阵。

#### Next Sentence Prediction (NSP)

判断两个句子是否连续：

$$P(isNext) = \text{sigmoid}(h_{[CLS]} \cdot w)$$

### 微调范式

在不同任务上的微调方式：

| 任务 | 输出 | 损失 |
|------|------|------|
| 分类 | $h_{[CLS]}$ → 分类头 | Cross-Entropy |
| NER | 每个 token → 分类头 | Cross-Entropy |
| QA | $h_{start}$, $h_{end}$ → span 预测 | Cross-Entropy |

## 编码器的优势

### 1. 双向上下文

每个位置都能看到整个序列：

$$h_i = f(x_1, x_2, ..., x_n)$$

对比 RNN/LSTM 只能看到左侧上下文。

### 2. 并行计算

所有位置可以同时计算，不像 RNN 需要顺序处理。

### 3. 长距离依赖

任意两个位置之间的路径长度为 O(1)，不像 RNN 是 O(n)。

## 实现细节

### 参数共享

在 BERT 中，词嵌入矩阵和输出分类器共享：

$$\text{MLM\_logits} = h \cdot E^T + b$$

这减少了参数量，并改善了训练稳定性。

### Dropout

通常应用于：
- Attention weights
- FFN 输出
- Embedding sum

Dropout 率通常为 0.1。

## 变体与改进

### RoBERTa

- 移除 NSP 任务
- 更大的 batch size
- 更多数据
- 动态 masking

### ALBERT

- 跨层参数共享
- 因式分解的嵌入
- 减少参数量

### DeBERTa

- 解耦注意力
- 增强的掩码解码器
- 更好的性能

## 参考文献

1. Vaswani et al. (2017). *Attention Is All You Need*
2. Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers*
3. Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
4. Lan et al. (2019). *ALBERT: A Lite BERT for Self-supervised Learning*
5. He et al. (2020). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*
