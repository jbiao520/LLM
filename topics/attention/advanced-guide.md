# 注意力机制（Attention）深入版

> 面向有机器学习基础读者的技术详解

## 概述

注意力机制是 Transformer 架构的核心，彻底改变了 NLP 和深度学习领域。本文深入分析 Self-Attention、Multi-Head Attention 和 Flash Attention 的数学原理。

## Scaled Dot-Product Attention

### 数学定义

给定查询矩阵 $Q$、键矩阵 $K$、值矩阵 $V$：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

<a id="formula-attention-1"></a>
[📖 查看公式附录详解](#formula-attention-1-detail)

其中：
- $Q \in \mathbb{R}^{n \times d_k}$（查询）
- $K \in \mathbb{R}^{m \times d_k}$（键）
- $V \in \mathbb{R}^{m \times d_v}$（值）
- $d_k$ 是键的维度

**公式解释**
- **公式含义**：先计算查询与键的相似度 $QK^T$，缩放后做 softmax 得到注意力权重，再与 $V$ 加权求和得到输出。
- **变量说明**：$n$ 为查询序列长度，$m$ 为键/值序列长度；$d_k$ 为键维度；$d_v$ 为值维度。
- **直觉/作用**：每个查询位置都会“看”所有键位置，权重越大表示越相关。

### 为什么除以 $\sqrt{d_k}$？

假设 $q$ 和 $k$ 的元素是独立的随机变量，均值为 0，方差为 1。

点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0，方差为 $d_k$。

当 $d_k$ 很大时，点积的绝对值会很大，导致 softmax 进入饱和区，梯度极小。

除以 $\sqrt{d_k}$ 使方差归一化为 1：

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

<a id="formula-attention-2"></a>
[📖 查看公式附录详解](#formula-attention-2-detail)

**公式解释**
- **公式含义**：缩放后的点积方差被归一到 1。
- **变量说明**：$q \cdot k$ 是点积；$\text{Var}(\cdot)$ 是方差。
- **直觉/作用**：避免 softmax 输入过大导致梯度饱和。

### 逐步计算

1. **计算注意力分数**：$S = QK^T \in \mathbb{R}^{n \times m}$

2. **缩放**：$S = S / \sqrt{d_k}$

3. **Softmax**：$A = \text{softmax}(S)$，其中 $A_{ij} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}$

4. **加权求和**：$O = AV \in \mathbb{R}^{n \times d_v}$

**公式解释**
- **$S = QK^T$**：得到每个查询与每个键的相似度矩阵。
- **$S / \sqrt{d_k}$**：缩放相似度，控制数值范围。
- **$A = \text{softmax}(S)$**：将每一行归一化为权重分布，$A_{ij}$ 表示查询 $i$ 关注键 $j$ 的程度。
- **$O = AV$**：用权重对值向量加权求和得到输出。

### 计算复杂度

- 时间复杂度：$O(n^2 \cdot d)$（$n$ 是序列长度）
- 空间复杂度：$O(n^2)$（存储注意力矩阵）

**公式解释**
- **$O(n^2 \cdot d)$**：序列长度 $n$ 增长时，计算量近似按平方增长。
- **$O(n^2)$**：需要存储 $n \times n$ 的注意力矩阵，因此内存随 $n^2$ 增长。

## Self-Attention

当 $Q$、$K$、$V$ 都来自同一个输入 $X$ 时：

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

<a id="formula-attention-3"></a>
[📖 查看公式附录详解](#formula-attention-3-detail)

其中 $W^Q, W^K, W^V$ 是可学习的投影矩阵。

**公式解释**
- **公式含义**：将同一输入 $X$ 通过三组不同的线性变换得到查询、键和值。
- **变量说明**：$X$ 为输入序列表示；$W^Q, W^K, W^V$ 为参数矩阵。
- **直觉/作用**：同一输入在不同“视角”下被投影，以计算关联性。

### 特点

1. **全局感受野**：每个位置可以直接关注序列中的任何位置
2. **动态权重**：注意力权重根据输入内容动态计算
3. **并行计算**：所有位置的计算可以并行

## Multi-Head Attention

### 数学定义

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

<a id="formula-attention-4"></a>
[📖 查看公式附录详解](#formula-attention-4-detail)

其中：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

<a id="formula-attention-5"></a>
[📖 查看公式附录详解](#formula-attention-5-detail)

**公式解释**
- **公式含义**：将注意力分成 $h$ 个子空间并行计算，最后拼接并线性映射。
- **变量说明**：$h$ 为头数；$W_i^Q, W_i^K, W_i^V$ 为每个头的投影矩阵；$W^O$ 为输出映射。
- **直觉/作用**：不同头关注不同关系，提升表达能力。

### 参数设置

- $h$：头数（如 8、12、16）
- $d_k = d_{model} / h$：每个头的维度
- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

### 为什么多头有效？

1. **多表示空间**：每个头学习不同的表示子空间
2. **并行关注**：同时关注不同位置的不同信息
3. **增强表达能力**：等价于增加网络容量

### 复杂度分析

总参数量不变（与单头相比），但表达能力更强。

## Masked Attention

### Padding Mask

处理变长序列时，填充位置不应该参与计算：

$$S_{ij} = \begin{cases} S_{ij} & \text{if } j \text{ is valid} \\ -\infty & \text{if } j \text{ is padding} \end{cases}$$

<a id="formula-attention-6"></a>
[📖 查看公式附录详解](#formula-attention-6-detail)

**公式解释**
- **公式含义**：对 padding 位置施加 $-\infty$，使 softmax 后权重变为 0。
- **变量说明**：$S_{ij}$ 为注意力分数；$j$ 为被关注位置。
- **直觉/作用**：让模型忽略无效填充位置。

### Causal Mask (Look-ahead Mask)

在解码器中，位置 $i$ 只能看到位置 $1$ 到 $i$：

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

<a id="formula-attention-7"></a>
[📖 查看公式附录详解](#formula-attention-7-detail)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

<a id="formula-attention-8"></a>
[📖 查看公式附录详解](#formula-attention-8-detail)

**公式解释**
- **公式含义**：用掩码 $M$ 把未来位置的注意力分数置为 $-\infty$，保证因果性。
- **变量说明**：$M_{ij}$ 表示位置 $i$ 是否允许看位置 $j$；允许为 0，不允许为 $-\infty$。
- **直觉/作用**：解码器生成时只能看“过去”，避免信息泄露。

## Cross-Attention

编码器-解码器注意力中，Query 来自解码器，Key 和 Value 来自编码器：

$$Q = X_{dec}W^Q, \quad K = X_{enc}W^K, \quad V = X_{enc}W^V$$

<a id="formula-attention-9"></a>
[📖 查看公式附录详解](#formula-attention-9-detail)

这让解码器可以"查看"编码器的输出。

**公式解释**
- **公式含义**：查询来自解码器，键和值来自编码器。
- **变量说明**：$X_{dec}$ 为解码器输入表示；$X_{enc}$ 为编码器输出表示。
- **直觉/作用**：解码器在生成时能对齐并利用源序列信息。

## Flash Attention

### 问题：内存瓶颈

标准注意力的空间复杂度是 $O(n^2)$，长序列会占用大量内存。

### Flash Attention 的思想

1. **分块计算**：将 Q、K、V 分成小块
2. **在线 Softmax**：逐块计算 softmax，不需要存储完整的注意力矩阵
3. **内存重用**：利用 GPU 的高带宽内存 (HBM) 和 SRAM

### 算法要点

```
for each block of Q:
    for each block of K, V:
        compute local attention
        update output incrementally
```

### 效果

- 内存复杂度：$O(n)$（线性）
- 速度提升：2-4 倍
- 数值等价：输出与标准注意力完全相同

## 注意力变体

| 变体 | 特点 | 复杂度 |
|------|------|--------|
| Standard Attention | 全序列注意力 | $O(n^2)$ |
| Flash Attention | 内存优化，分块计算 | $O(n)$ 内存 |
| Sparse Attention | 稀疏注意力模式 | $O(n\sqrt{n})$ |
| Linear Attention | 线性近似 | $O(n)$ |
| Multi-Query Attention | 共享 Key 和 Value | 减少参数 |

**公式解释**
- 表中的 $O(n^2)$、$O(n)$ 是时间/空间复杂度量级，用来描述序列长度增大时的增长速度。

## 实现细节

### 数值稳定性

```python
# Bad: 可能溢出
scores = Q @ K.T / sqrt(d_k)

# Good: 减去最大值
scores = Q @ K.T / sqrt(d_k)
scores = scores - scores.max(dim=-1, keepdim=True)
attn = F.softmax(scores, dim=-1)
```

### 混合精度训练

```python
# 注意力计算通常使用 FP32
with torch.cuda.amp.autocast(enabled=False):
    scores = Q.float() @ K.float().T / sqrt(d_k)
```

## 参考文献

1. Vaswani et al. (2017). *Attention Is All You Need*
2. Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention*
3. Child et al. (2019). *Generating Long Sequences with Sparse Transformers*
4. Shazeer (2019). *Fast Transformer Decoding: One Write-Head is All You Need*

