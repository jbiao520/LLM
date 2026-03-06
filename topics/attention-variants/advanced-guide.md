# 注意力变体（Attention Variants）深入版

> 面向有机器学习基础读者的技术详解

## 概述

本文深入分析 MHA、MQA、GQA 的数学原理，以及 Flash Attention 的算法细节。

## 标准 Multi-Head Attention (MHA)

### 数学定义

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

<a id="formula-attentionvariants-1"></a>

其中每个头：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

<a id="formula-attentionvariants-2"></a>

**公式解释**
- **公式含义**：每个头有独立的 Q、K、V 投影。
- **变量说明**：$h$ 是头数；$W_i^Q, W_i^K, W_i^V$ 是每个头的投影矩阵。
- **直觉/作用**：每个头学习不同的注意力模式。

### KV Cache 大小

对于 $h$ 个头，每个头的维度 $d_k$：

$$\text{KV Cache} = 2 \times L \times h \times d_k$$

<a id="formula-attentionvariants-3"></a>

**公式解释**
- **公式含义**：需要存储每个位置的 K 和 V，共 $h$ 组。
- **变量说明**：$L$ 是序列长度；2 是 K 和 V。
- **直觉/作用**：序列越长、头数越多，KV Cache 越大。

## Multi-Query Attention (MQA)

### 核心思想

所有头共享同一组 K 和 V：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW^K, VW^V)$$

<a id="formula-attentionvariants-4"></a>

**公式解释**
- **公式含义**：Q 保持多头，但 K 和 V 只有一组。
- **变量说明**：$W^K, W^V$ 是共享的投影矩阵。
- **直觉/作用**：大幅减少 KV Cache，但可能损失精度。

### KV Cache 大小

$$\text{KV Cache}_{MQA} = 2 \times L \times d_k$$

<a id="formula-attentionvariants-5"></a>

相比 MHA 减少了 $h$ 倍。

### 优缺点

| 优点 | 缺点 |
|------|------|
| KV Cache 减少 $h$ 倍 | 精度可能下降 |
| 推理速度更快 | 训练需要特殊处理 |
| 内存占用更低 | 不适合所有任务 |

## Grouped Query Attention (GQA)

### 核心思想

将 $h$ 个查询头分成 $g$ 组，每组共享 K 和 V：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_{\lfloor i/g \rfloor}^K, VW_{\lfloor i/g \rfloor}^V)$$

<a id="formula-attentionvariants-6"></a>

**公式解释**
- **公式含义**：查询头 $i$ 使用第 $\lfloor i/g \rfloor$ 组的 K 和 V。
- **变量说明**：$g$ 是组数（$1 \leq g \leq h$）。
- **直觉/作用**：在 MQA 和 MHA 之间平衡。

### KV Cache 大小

$$\text{KV Cache}_{GQA} = 2 \times L \times g \times d_k$$

<a id="formula-attentionvariants-7"></a>

**公式解释**
- **公式含义**：需要存储 $g$ 组 K 和 V。
- **变量说明**：$g$ 是组数。
- **直觉/作用**：相比 MHA 减少 $h/g$ 倍。

### LLaMA 2 的 GQA 配置

- 32 个查询头
- 8 组 KV 头
- 每 4 个查询头共享一组 KV

KV Cache 减少到原来的 1/4。

## Flash Attention

### 标准注意力的内存瓶颈

```
内存访问模式:
Q [HBM] → K [HBM] → S = QK^T [HBM] → softmax(S) [HBM] → P [HBM] → V [HBM] → O [HBM]
         读取      读取       写入/读取        写入/读取      写入/读取   读取      写入

总计: O(N²) 次 HBM 访问
```

### Flash Attention 算法

**核心思想**：分块计算，避免存储完整的注意力矩阵

```
算法:
1. 将 Q, K, V 分成小块 (适合 SRAM 大小)
2. 对每个 Q 块:
   - 初始化输出 O 和统计量 m, l
   - 对每个 K, V 块:
     a. 在 SRAM 中计算 S_ij = Q_i K_j^T
     b. 计算局部 m_ij, l_ij
     c. 在线更新全局 m, l, O
3. 写入最终输出 O 到 HBM
```

### 在线 Softmax

关键技巧：增量计算 softmax

$$\text{softmax}(x \cup y) = \text{softmax}([\text{softmax}(x), \text{softmax}(y)])$$

<a id="formula-attentionvariants-8"></a>

**公式解释**
- **公式含义**：可以分块计算 softmax，然后合并结果。
- **变量说明**：$x, y$ 是不同块的结果。
- **直觉/作用**：不需要存储完整的注意力矩阵。

### 内存复杂度

| 方法 | 时间复杂度 | 空间复杂度 | HBM 访问 |
|------|-----------|-----------|---------|
| 标准注意力 | $O(N^2 d)$ | $O(N^2)$ | $O(N^2)$ |
| Flash Attention | $O(N^2 d)$ | $O(N)$ | $O(N)$ |

**公式解释**
- **$O(N^2 d)$**：计算量不变，但内存访问大大减少。
- **$O(N)$ 空间**：不需要存储 $N \times N$ 的注意力矩阵。

## Sliding Window Attention

### 数学定义

$$\text{Attention}_{SW}(Q, K, V) = \text{softmax}(S \odot M)V$$

<a id="formula-attentionvariants-9"></a>

其中 $M$ 是滑动窗口掩码：

$$M_{ij} = \begin{cases} 0 & \text{if } i - w \leq j \leq i \\ -\infty & \text{otherwise} \end{cases}$$

<a id="formula-attentionvariants-10"></a>

**公式解释**
- **公式含义**：只在窗口大小 $w$ 内计算注意力。
- **变量说明**：$w$ 是窗口大小。
- **直觉/作用**：复杂度从 $O(N^2)$ 降到 $O(Nw)$。

### 实现技巧

使用扩张（dilation）增加感受野：

$$M_{ij} = \begin{cases} 0 & \text{if } i - k \cdot w \leq j \leq i \text{ for some } k \\ -\infty & \text{otherwise} \end{cases}$$

<a id="formula-attentionvariants-11"></a>

## 对比总结

| 方法 | KV Cache | 计算复杂度 | 精度 | 适用场景 |
|------|---------|-----------|------|---------|
| MHA | 100% | $O(N^2)$ | 最好 | 研究、短序列 |
| MQA | 12.5% (8头) | $O(N^2)$ | 较好 | 推理优化 |
| GQA | 25% (8头,2组) | $O(N^2)$ | 好 | 生产环境 |
| Flash | 100% | $O(N^2)$ | 完全相同 | 训练、推理 |
| Sliding | 100% | $O(Nw)$ | 取决于窗口 | 长序列 |

## 参考文献

1. Shazeer (2019). *Fast Transformer Decoding: One Write-Head is All You Need*
2. Ainslie et al. (2023). *GQA: Training Generalized Multi-Query Transformer Models*
3. Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention*
4. Dao (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
5. Beltagy et al. (2020). *Longformer: The Long-Document Transformer*
