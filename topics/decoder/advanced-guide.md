# 解码器（Decoder）深入版

> 面向有机器学习基础读者的技术详解

## 概述

Transformer 解码器是自回归生成模型的核心，通过因果注意力机制实现逐 token 生成。本文深入分析解码器结构、自回归生成、KV Cache 等关键技术。

## Transformer Decoder 结构

### 单层结构

每层解码器包含三个子层：

1. **Masked Multi-Head Self-Attention**
2. **Cross-Attention (Encoder-Decoder Attention)**
3. **Position-wise Feed-Forward Network**

### 数学公式

对于第 $l$ 层：

$$h_l' = \text{LayerNorm}(h_{l-1} + \text{MaskedMHSA}(h_{l-1}))$$

$$h_l'' = \text{LayerNorm}(h_l' + \text{CrossAttention}(h_l', c))$$

$$h_l = \text{LayerNorm}(h_l'' + \text{FFN}(h_l''))$$

其中 $c$ 是编码器的输出。

## Masked Self-Attention

### 因果掩码

确保位置 $i$ 只能关注位置 $1$ 到 $i$：

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

注意力计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

### 注意力矩阵形状

对于一个长度为 4 的序列：

$$M = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

经过 softmax 后，$-\infty$ 位置变为 0。

## 自回归生成

### 生成过程

给定输入序列 $x_{1:t}$，预测下一个 token：

$$P(x_{t+1} | x_{1:t}) = \text{softmax}(Wh_t)$$

其中 $h_t$ 是位置 $t$ 的隐藏状态。

### 完整序列概率

$$P(x_1, ..., x_T) = \prod_{t=1}^{T} P(x_t | x_{1:t-1})$$

### 解码策略

#### Greedy Decoding

$$x_{t+1} = \arg\max_{x} P(x | x_{1:t})$$

#### Beam Search

维护 $k$ 个候选序列，每步扩展并保留 top-k。

#### Sampling

$$x_{t+1} \sim P(x | x_{1:t})$$

#### Temperature Sampling

$$P_{temp}(x | x_{1:t}) = \frac{\exp(\log P(x) / T)}{\sum_{x'} \exp(\log P(x') / T)}$$

- $T > 1$：更随机
- $T < 1$：更确定
- $T \to 0$：趋近 greedy

## KV Cache

### 问题

生成第 $t$ 个 token 时，需要重新计算所有之前 token 的 K 和 V。

### 解决方案

缓存之前计算过的 K 和 V：

$$K_{1:t} = [K_1, K_2, ..., K_t]$$
$$V_{1:t} = [V_1, V_2, ..., V_t]$$

每次只需计算新 token 的 Q, K, V，然后更新缓存。

### 实现细节

```python
# 初始化缓存
past_key_values = None

for step in range(max_length):
    # 只处理最后一个 token
    if past_key_values is not None:
        input_ids = input_ids[:, -1:]

    # 前向传播，使用和更新缓存
    outputs = model(input_ids, past_key_values=past_key_values)
    past_key_values = outputs.past_key_values

    # 采样下一个 token
    next_token = sample(outputs.logits)
    input_ids = torch.cat([input_ids, next_token], dim=-1)
```

### 内存分析

KV Cache 的内存占用：

$$M_{cache} = 2 \times L \times B \times n_{heads} \times d_{head} \times (n_{ctx} + n_{gen})$$

其中 $L$ 是层数，$B$ 是 batch size。

## GPT 架构

### 模型配置

| 模型 | 层数 | 隐藏维度 | 注意力头 | 参数量 |
|------|------|---------|---------|--------|
| GPT-2 Small | 12 | 768 | 12 | 124M |
| GPT-2 Medium | 24 | 1024 | 16 | 355M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 175B |

### 训练目标

语言建模损失：

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{1:t-1})$$

## Cross-Attention（用于 Encoder-Decoder）

当解码器与编码器配合时（如机器翻译）：

- **Query** 来自解码器
- **Key, Value** 来自编码器

$$\text{CrossAttention}(Q_d, K_e, V_e) = \text{softmax}\left(\frac{Q_d K_e^T}{\sqrt{d_k}}\right)V_e$$

这允许解码器"查看"编码器的表示。

## 推理优化

### 1. 批处理

同时处理多个请求，提高 GPU 利用率。

### 2. 连续批处理

动态调整批大小，不等待所有序列完成。

### 3. 投机解码

用小模型快速生成候选，大模型验证。

### 4. 量化

INT8/INT4 量化减少内存和加速推理。

## 参考文献

1. Vaswani et al. (2017). *Attention Is All You Need*
2. Radford et al. (2018). *Improving Language Understanding by Generative Pre-Training*
3. Radford et al. (2019). *Language Models are Unsupervised Multitask Learners*
4. Brown et al. (2020). *Language Models are Few-Shot Learners*
5. Leviathan et al. (2023). *Fast Inference from Transformers via Speculative Decoding*
