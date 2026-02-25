# 前馈网络层（FFN）深入版

> 面向有机器学习基础读者的技术详解

## 概述

Position-wise Feed-Forward Network 是 Transformer 中的关键组件，负责对每个位置的特征进行非线性变换。本文深入分析 FFN 的数学原理、变体设计及其在 LLM 中的应用。

## 标准 FFN

### 数学定义

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

其中：
- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$
- 通常 $d_{ff} = 4 \times d_{model}$

### Position-wise 含义

对序列中的每个位置独立应用相同的变换：

$$\text{FFN}(X)_{i,:} = \text{FFN}(X_{i,:})$$

不同位置共享参数，但计算是独立的。

### 参数量分析

单个 FFN 层的参数量：

$$P_{FFN} = 2 \times d_{model} \times d_{ff} + d_{model} + d_{ff}$$

对于 $d_{model} = 768$，$d_{ff} = 3072$：

$$P_{FFN} = 2 \times 768 \times 3072 + 768 + 3072 = 4,722,432$$

## 激活函数选择

### ReLU

$$\text{FFN}_{ReLU}(x) = \max(0, xW_1)W_2$$

- 简单高效
- 原始 Transformer 使用

### GELU

$$\text{FFN}_{GELU}(x) = \text{GELU}(xW_1)W_2$$

- 平滑的 ReLU 替代
- BERT、GPT 使用
- 性能略好

### Swish/SiLU

$$\text{FFN}_{Swish}(x) = (xW_1 \odot \sigma(xW_1))W_2$$

- 自门控激活
- 某些模型使用

## GLU 变体

### GLU (Gated Linear Unit)

$$\text{GLU}(x) = (xW) \odot \sigma(xV)$$

包含两个线性变换，一个作为门控。

### SwiGLU

$$\text{SwiGLU}(x) = \text{Swish}(xW) \odot (xV)$$

或等价地：

$$\text{SwiGLU}(x) = (xW \odot \sigma(xW)) \odot (xV)$$

LLaMA 等现代 LLM 使用。

### GeGLU

$$\text{GeGLU}(x) = \text{GELU}(xW) \odot (xV)$$

### 参数量对比

| 变体 | 参数量 | 相比标准 FFN |
|------|--------|-------------|
| 标准 FFN | $2 \times d \times 4d$ | 1x |
| SwiGLU | $3 \times d \times \frac{8}{3}d$ | 1x |
| GeGLU | $3 \times d \times \frac{8}{3}d$ | 1x |

为了保持相同参数量，GLU 变体通常缩小隐藏维度。

## 实现

### 标准 FFN (PyTorch)

```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))
```

### SwiGLU (PyTorch)

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # d_ff 通常设为 2/3 * 4 * d_model ≈ 2.67 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # 门控分支
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU(x) = Swish(xW1) ⊙ (xW3) * W2
        return self.dropout(self.w2(
            F.silu(self.w1(x)) * self.w3(x)
        ))
```

## FFN 的作用

### 1. 非线性变换

Attention 主要是线性操作（加权求和），FFN 提供非线性。

### 2. 知识存储

研究表明，FFN 层存储了大量的"事实知识"：
- 前层存储语法知识
- 后层存储语义知识

### 3. 特征提取

在每个位置独立提取高层特征。

## 与 Attention 的对比

| 特性 | Attention | FFN |
|------|-----------|-----|
| 信息交互 | 跨位置 | 单位置 |
| 主要功能 | 信息聚合 | 特征变换 |
| 非线性 | Softmax | 激活函数 |
| 参数占比 | ~1/3 | ~2/3 |

## 优化策略

### 1. 参数共享

跨层共享 FFN 参数，减少模型大小。

### 2. 分组 FFN

将 FFN 分成多组，减少计算量。

### 3. 稀疏 FFN

只激活部分神经元，类似 MoE。

## 参考文献

1. Vaswani et al. (2017). *Attention Is All You Need*
2. Shazeer (2020). *GLU Variants Improve Transformer*
3. Geva et al. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories*
4. Tolstikhin et al. (2021). *MLP-Mixer: An all-MLP Architecture for Vision*
