# Position Embedding 深入版

> 面向有机器学习基础读者的技术详解

## 概述

位置嵌入（Position Embedding/Encoding）是为序列中每个位置生成唯一表示的技术，使不具备递归结构的模型（如 Transformer）能够捕捉序列顺序信息。

## 为什么需要位置信息？

Transformer 的自注意力机制是**置换不变的**（permutation invariant）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

对于输入序列的任意排列，注意力机制的输出只是对应排列，无法区分顺序。因此需要显式注入位置信息。

## Sinusoidal Position Encoding

原始 Transformer 论文提出的正弦位置编码：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $pos$ 是位置索引
- $i$ 是维度索引
- $d_{model}$ 是嵌入维度

### 设计动机

#### 1. 唯一性

每个位置都有唯一的编码表示。

#### 2. 有界性

所有值都在 $[-1, 1]$ 范围内。

#### 3. 相对位置关系

对于任意固定偏移 $k$，$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数：

$$\sin(x + k) = \sin(x)\cos(k) + \cos(x)\sin(k)$$

这允许模型通过线性变换学习相对位置关系。

#### 4. 外推能力

理论上可以处理任意长度的序列（虽然实践中效果会下降）。

### 频率分析

波长随维度变化：

$$\lambda_i = 2\pi \cdot 10000^{2i/d_{model}}$$

- 低维度：短波长，捕捉局部位置差异
- 高维度：长波长，捕捉全局位置模式

## Learnable Position Embedding

BERT、GPT 等模型使用可学习的位置嵌入：

$$E_{final} = E_{word} + E_{position}$$

其中 $E_{position} \in \mathbb{R}^{L \times d}$ 是可学习的参数矩阵。

### 优缺点

| 方面 | Sinusoidal | Learnable |
|------|------------|-----------|
| 灵活性 | 固定 | 可适应数据 |
| 外推能力 | 理论无限 | 受限于训练长度 |
| 参数量 | 0 | $L \times d$ |

## 相对位置编码

绝对位置编码有一些问题：
- 长度泛化能力差
- 无法很好地捕捉相对距离

### Shaw's Relative Position Encoding

在注意力计算中引入相对位置：

$$e_{ij} = \frac{x_i W^Q (x_j W^K + a_{ij}^K)^T}{\sqrt{d_k}}$$

其中 $a_{ij}^K$ 是位置 $i$ 和 $j$ 之间的相对位置编码。

### T5 Bias

T5 模型直接在注意力分数上添加可学习的相对位置偏置：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right) V$$

其中 $B$ 是相对位置偏置矩阵。

### RoPE (Rotary Position Embedding)

RoPE 通过旋转矩阵将位置信息注入注意力：

$$\text{RoPE}(x_m, m) = \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \end{pmatrix} \otimes \begin{pmatrix} \cos(m\theta) \\ \sin(m\theta) \end{pmatrix}$$

核心性质：

$$\langle \text{RoPE}(x_m, m), \text{RoPE}(x_n, n) \rangle = \text{Re}[\langle x_m, x_n \rangle e^{i(m-n)\theta}]$$

注意力分数只依赖于相对位置 $m - n$。

### ALiBi (Attention with Linear Biases)

ALiBi 在注意力分数上添加线性递减的偏置：

$$\text{score}_{ij} = q_i \cdot k_j - m \cdot |i - j|$$

其中 $m$ 是每个注意力头特定的斜率参数。

## 位置编码的选择

| 方法 | 代表模型 | 优点 | 缺点 |
|------|----------|------|------|
| Sinusoidal | 原始 Transformer | 无参数，可外推 | 灵活性差 |
| Learnable | BERT, GPT | 简单有效 | 长度受限 |
| RoPE | LLaMA, PaLM | 相对位置，长度泛化 | 实现复杂 |
| ALiBi | BLOOM | 简单，长度泛化好 | 绝对位置信息弱 |

## 代码示例

参见 `examples/` 目录。

## 参考文献

1. Vaswani et al. (2017). *Attention Is All You Need*
2. Shaw et al. (2018). *Self-Attention with Relative Position Representations*
3. Raffel et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*
4. Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*
5. Press et al. (2021). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*
