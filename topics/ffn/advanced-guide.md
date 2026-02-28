# 前馈网络层（FFN）深入版

> 面向有机器学习基础读者的技术详解

## 概述

Position-wise Feed-Forward Network 是 Transformer 中的关键组件，负责对每个位置的特征进行非线性变换。本文深入分析 FFN 的数学原理、变体设计及其在 LLM 中的应用。

## 标准 FFN

### 数学定义

$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

<a id="formula-ffn-1"></a>
[📖 查看公式附录详解](#formula-ffn-1-detail)

**公式解释**
- **公式含义**：先升维到 $d_{ff}$，经 GELU 激活后再降维回 $d_{model}$。
- **变量说明**：$x$ 为输入向量；$W_1, W_2$ 为权重矩阵；$b_1, b_2$ 为偏置；GELU 为激活函数。
- **��觉/作用**：扩展中间维度增加非线性表达能力，是 Transformer 中每个位置的核心变换。

其中：
- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$
- 通常 $d_{ff} = 4 \times d_{model}$

### Position-wise 含义

对序列中的每个位置独立应用相同的变换：

$$\text{FFN}(X)_{i,:} = \text{FFN}(X_{i,:})$$

<a id="formula-ffn-2"></a>
[📖 查看公式附录详解](#formula-ffn-2-detail)

**公式解释**
- **公式含义**：序列中每个位置的向量独立通过同一个 FFN 变换。
- **变量说明**：$X_{i,:}$ 为第 $i$ 个位置的向量；$\text{FFN}(X)_{i,:}$ 为其变换后输出。
- **直觉/作用**：各位置共享参数但独立处理，保证位置间信息不串扰。

不同位置共享参数，但计算是独立的。

### 参数量分析

单个 FFN 层的参数量：

$$P_{FFN} = 2 \times d_{model} \times d_{ff} + d_{model} + d_{ff}$$

<a id="formula-ffn-3"></a>
[📖 查看公式附录详解](#formula-ffn-3-detail)

**公式解释**
- **公式含义**：FFN 参数量 = 两个权重矩阵 + 两个偏置向量。
- **变量说明**：$d_{model}$ 为输入/输出维度；$d_{ff}$ 为中间隐藏维度（通常 $4d$）。
- **直觉/作用**：参数量与维度平方成正比，FFN 约占 Transformer 总参数的 2/3。

对于 $d_{model} = 768$，$d_{ff} = 3072$：

$$P_{FFN} = 2 \times 768 \times 3072 + 768 + 3072 = 4,722,432$$

<a id="formula-ffn-4"></a>
[📖 查看公式附录详解](#formula-ffn-4-detail)

**公式解释**
- **公式含义**：代入具体数值计算 BERT-Base 每层 FFN 的参数量。
- **变量说明**：$768$ 为隐藏维度；$3072 = 4 \times 768$ 为中间维度。
- **直觉/作用**：约 470 万参数，是 BERT-Base 参数量的重要组成部分。

## 激活函数选择

### ReLU

$$\text{FFN}_{ReLU}(x) = \max(0, xW_1)W_2$$

<a id="formula-ffn-5"></a>
[📖 查看公式附录详解](#formula-ffn-5-detail)

**公式解释**
- **公式含义**：用 ReLU 激活，将负值截断为 0。
- **变量说明**：$xW_1$ 为升维后的结果；$\max(0, \cdot)$ 为逐元素 ReLU。
- **直觉/作用**：计算简单，正区间梯度稳定，是原始 Transformer 的选择。

- 简单高效
- 原始 Transformer 使用

### GELU

$$\text{FFN}_{GELU}(x) = \text{GELU}(xW_1)W_2$$

<a id="formula-ffn-6"></a>
[📖 查看公式附录详解](#formula-ffn-6-detail)

**公式解释**
- **公式含义**：用 GELU 替代 ReLU，在 0 附近更平滑。
- **变量说明**：GELU 为高斯误差线性单元，与正态分布 CDF 相关。
- **直觉/作用**：平滑过渡避免硬截断，训练更稳定，BERT/GPT 使用。

- 平滑的 ReLU 替代
- BERT、GPT 使用
- 性能略好

### Swish/SiLU

$$\text{FFN}_{Swish}(x) = (xW_1 \odot \sigma(xW_1))W_2$$

<a id="formula-ffn-7"></a>
[📖 查看公式附录详解](#formula-ffn-7-detail)

**公式解释**
- **公式含义**：用 Swish 激活，输入与 sigmoid 门控的乘积。
- **变量说明**：$\odot$ 为逐元素乘法；$\sigma$ 为 Sigmoid 函数。
- **直觉/作用**：非单调、有下界无上界，深层网络效果更好。

- 自门控激活
- 某些模型使用

## GLU 变体

### GLU (Gated Linear Unit)

$$\text{GLU}(x) = (xW) \odot \sigma(xV)$$

<a id="formula-ffn-8"></a>
[📖 查看公式附录详解](#formula-ffn-8-detail)

**公式解释**
- **公式含义**：两个并行线性变换，一个作为门控信号控制另一个。
- **变量说明**：$W, V$ 为两组权重；$\sigma(xV)$ 为门控值（0-1）；$\odot$ 为逐元素乘。
- **直觉/作用**：门控机制让模型自适应选择哪些信息通过。

包含两个线性变换，一个作为门控。

### SwiGLU

$$\text{SwiGLU}(x) = \text{Swish}(xW) \odot (xV)$$

<a id="formula-ffn-9"></a>
[📖 查看公式附录详解](#formula-ffn-9-detail)

**公式解释**
- **公式含义**：用 Swish 替代 Sigmoid 作为门控激活。
- **变量说明**：$\text{Swish}(x) = x \cdot \sigma(x)$；$W, V$ 为两组权重。
- **直觉/作用**：Swish 比 Sigmoid 更平滑，LLaMA 等现代 LLM 采用。

或等价地：

$$\text{SwiGLU}(x) = (xW \odot \sigma(xW)) \odot (xV)$$

<a id="formula-ffn-10"></a>
[📖 查看公式附录详解](#formula-ffn-10-detail)

LLaMA 等现代 LLM 使用。

### GeGLU

$$\text{GeGLU}(x) = \text{GELU}(xW) \odot (xV)$$

<a id="formula-ffn-11"></a>
[📖 查看公式附录详解](#formula-ffn-11-detail)

**公式解释**
- **公式含义**：用 GELU 替代 Sigmoid 作为门控激活。
- **变量说明**：GELU 为高斯误差线性单元；$W, V$ 为两组权重。
- **直觉/作用**：GELU 的平滑性带来更好的训练稳定性。

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

