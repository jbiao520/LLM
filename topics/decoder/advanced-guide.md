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

**公式解释**
- **公式含义**：每层依次做“因果自注意力 → 编码器交叉注意力 → 前馈网络”，每步都加残差并做层归一化。
- **变量说明**：$h_{l-1}$ 为上一层输出；$h_l', h_l''$ 为中间结果；$c$ 为编码器输出。
- **直觉/作用**：残差连接保证信息不丢失，LayerNorm 稳定训练。

## Masked Self-Attention

### 因果掩码

确保位置 $i$ 只能关注位置 $1$ 到 $i$：

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

注意力计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

**公式解释**
- **公式含义**：用掩码 $M$ 把未来位置的注意力分数设为 $-\infty$，softmax 后权重为 0。
- **变量说明**：$M_{ij}$ 表示位置 $i$ 是否能看位置 $j$；$d_k$ 为键维度。
- **直觉/作用**：确保解码时只使用历史信息，符合自回归生成。

### 注意力矩阵形状

对于一个长度为 4 的序列：

$$M = \begin{bmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

经过 softmax 后，$-\infty$ 位置变为 0。

**公式解释**
- **公式含义**：矩阵上三角为 $-\infty$，表示禁止关注未来位置。
- **变量说明**：以长度为 4 的序列为例，行表示当前步，列表示可关注的历史位置。
- **直觉/作用**：让注意力权重只分配给过去和当前。

## 自回归生成

### 生成过程

给定输入序列 $x_{1:t}$，预测下一个 token：

$$P(x_{t+1} | x_{1:t}) = \text{softmax}(Wh_t)$$

其中 $h_t$ 是位置 $t$ 的隐藏状态。

**公式解释**
- **公式含义**：用线性层 $W$ 把隐藏状态映射到词表分数，再用 softmax 得到下一个 token 的概率分布。
- **变量说明**：$h_t$ 为当前步表示；$W$ 为输出投影矩阵。
- **直觉/作用**：把模型内部表示转成可采样的词概率。

### 完整序列概率

$$P(x_1, ..., x_T) = \prod_{t=1}^{T} P(x_t | x_{1:t-1})$$

**公式解释**
- **公式含义**：整句概率等于每一步条件概率的连乘。
- **变量说明**：$T$ 为序列长度；$x_{1:t-1}$ 为历史上下文。
- **直觉/作用**：体现自回归分解假设，逐步生成整句。

### 解码策略

#### Greedy Decoding

$$x_{t+1} = \arg\max_{x} P(x | x_{1:t})$$

**公式解释**
- **公式含义**：每一步都选取概率最高的 token。
- **变量说明**：$\arg\max$ 表示取得使概率最大的 $x$。
- **直觉/作用**：确定性输出，但可能缺乏多样性。

#### Beam Search

维护 $k$ 个候选序列，每步扩展并保留 top-k。

#### Sampling

$$x_{t+1} \sim P(x | x_{1:t})$$

**公式解释**
- **公式含义**：按照概率分布随机采样下一个 token。
- **变量说明**：$\sim$ 表示“服从该分布采样”。
- **直觉/作用**：提高多样性，但可能引入噪声。

#### Temperature Sampling

$$P_{temp}(x | x_{1:t}) = \frac{\exp(\log P(x) / T)}{\sum_{x'} \exp(\log P(x') / T)}$$

- $T > 1$：更随机
- $T < 1$：更确定
- $T \to 0$：趋近 greedy

**公式解释**
- **公式含义**：用温度 $T$ 调整分布的尖锐程度。
- **变量说明**：$T$ 越大，分布越平滑；$T$ 越小，分布越尖锐。
- **直觉/作用**：控制生成的随机性与确定性之间的平衡。

## KV Cache

### 问题

生成第 $t$ 个 token 时，需要重新计算所有之前 token 的 K 和 V。

### 解决方案

缓存之前计算过的 K 和 V：

$$K_{1:t} = [K_1, K_2, ..., K_t]$$
$$V_{1:t} = [V_1, V_2, ..., V_t]$$

每次只需计算新 token 的 Q, K, V，然后更新缓存。

**公式解释**
- **公式含义**：把历史所有步的键和值拼接成缓存，避免重复计算。
- **变量说明**：$K_i, V_i$ 为第 $i$ 步计算得到的键和值。
- **直觉/作用**：时间换空间，将重复计算变成缓存读取。

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

**公式解释**
- **公式含义**：KV Cache 的内存大小与层数、批大小、头数、每头维度及序列长度成正比。
- **变量说明**：$n_{ctx}$ 为上下文长度，$n_{gen}$ 为生成长度；前面的 2 表示 K 和 V 两份缓存。
- **直觉/作用**：序列越长或模型越大，缓存占用越高。

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

**公式解释**
- **公式含义**：对每一步的正确 token 概率取对数并求和，取负号得到损失。
- **变量说明**：$P(x_t | x_{1:t-1})$ 为模型在第 $t$ 步对真实 token 的概率。
- **直觉/作用**：概率越大，损失越小；训练目标是最大化真实序列概率。

## Cross-Attention（用于 Encoder-Decoder）

当解码器与编码器配合时（如机器翻译）：

- **Query** 来自解码器
- **Key, Value** 来自编码器

$$\text{CrossAttention}(Q_d, K_e, V_e) = \text{softmax}\left(\frac{Q_d K_e^T}{\sqrt{d_k}}\right)V_e$$

这允许解码器"查看"编码器的表示。

**公式解释**
- **公式含义**：解码器查询 $Q_d$ 与编码器键 $K_e$ 计算相似度，并对编码器值 $V_e$ 加权求和。
- **变量说明**：$Q_d$ 来自解码器；$K_e, V_e$ 来自编码器；$d_k$ 为键维度。
- **直觉/作用**：在生成时动态对齐源序列信息。

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
