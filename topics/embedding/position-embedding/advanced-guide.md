# Position Embedding 深入版

> 面向有机器学习基础读者的技术详解

## 概述

位置嵌入（Position Embedding/Encoding）是为序列中每个位置生成唯一表示的技术，使不具备递归结构的模型（如 Transformer）能够捕捉序列顺序信息。

## 为什么需要位置信息？

Transformer 的自注意力机制是**置换不变的**（permutation invariant）：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

<a id="formula-position-embedding-1"></a>
[📖 查看公式附录详解](#formula-position-embedding-1-detail)

**公式解释**
- **公式含义**：注意力的输出只依赖 Q、K、V 的值，与序列顺序无关。
- **变量说明**：$Q, K, V$ 为查询、键、值矩阵；$d_k$ 为键维度。
- **直觉/作用**：打乱输入顺序只改变输出顺序，不改变内容；因此需要额外注入位置信息。

对于输入序列的任意排列，注意力机��的输出只是对应排列，无法区分顺序。因此需要显式注入位置信息。

## Sinusoidal Position Encoding

原始 Transformer 论文提出的正弦位置编码：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

<a id="formula-position-embedding-2"></a>
[📖 查看公式附录详解](#formula-position-embedding-2-detail)

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

<a id="formula-position-embedding-3"></a>
[📖 查看公式附录详解](#formula-position-embedding-3-detail)

**公式解释**
- **公式含义**：偶数维度用正弦、奇数维度用余弦编码位置信息，不同维度使用不同频率。
- **变量说明**：$pos$ 为位置索引；$i$ 为维度索引；$d_{model}$ 为嵌入维度；$10000$ 为底数控制频率范围。
- **直觉/作用**：每个位置有唯一编码；低维捕捉局部位置，高维捕捉全局位置；可外推到训练未见长度。

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

<a id="formula-position-embedding-4"></a>
[📖 查看公式附录详解](#formula-position-embedding-4-detail)

**公式解释**
- **公式含义**：正弦函数的加法公式表明位置 $pos+k$ 的编码可由位置 $pos$ 的编码线性变换得到。
- **变量说明**：$x$ 为当前位置；$k$ 为偏移量；$\cos(k), \sin(k)$ 为固定常数。
- **直觉/作用**：模型可通过学习线性变换来"计算"相对位置关系，无需显式编码所有位置对。

这允许模型通过线性变换学习相对位置关系。

#### 4. 外推能力

理论上可以处理任意长度的序列（虽然实践中效果会下降）。

### 频率分析

波长随维度变化：

$$\lambda_i = 2\pi \cdot 10000^{2i/d_{model}}$$

<a id="formula-position-embedding-5"></a>
[📖 查看公式附录详解](#formula-position-embedding-5-detail)

**公式解释**
- **公式含义**：不同维度的正弦波有不同的周期/波长。
- **变量说明**：$\lambda_i$ 为第 $i$ 维的波长；$2\pi$ 为正弦周期系数；$10000^{2i/d_{model}}$ 控制波长增长速度。
- **直觉/作用**：低维度波长短（敏感于位置变化），高维度波长大（捕捉远距离关系）。

- 低维度：短波长，捕捉局部位置差异
- 高维度：长波长，捕捉全局位置模式

## Learnable Position Embedding

BERT、GPT 等模型使用可学习的位置嵌入：

$$E_{final} = E_{word} + E_{position}$$

<a id="formula-position-embedding-6"></a>
[📖 查看公式附录详解](#formula-position-embedding-6-detail)

**公式解释**
- **公式含义**：将词嵌入与位置嵌入相加得到最终输入表示。
- **变量说明**：$E_{word}$ 为词嵌入；$E_{position}$ 为可学习的位置嵌入；$L$ 为最大序列长度；$d$ 为维度。
- **直觉/作用**：��置嵌入作为参数学习，能自适应数据分布，但长度受限于训练时的最大长度。

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

<a id="formula-position-embedding-7"></a>
[📖 查看公式附录详解](#formula-position-embedding-7-detail)

**公式解释**
- **公式含义**：在键向量上加上相对位置编码 $a_{ij}^K$，让注意力考虑位置关系。
- **变量说明**：$x_i, x_j$ 为位置 $i, j$ 的输入；$W^Q, W^K$ 为投影矩阵；$a_{ij}^K$ 为相对位置 $i-j$ 的编码。
- **直觉/作用**：注意力分数不仅依赖内容相似度，还依赖相对位置关系。

其中 $a_{ij}^K$ 是位置 $i$ 和 $j$ 之间的相对位置编码。

### T5 Bias

T5 模型直接在注意力分数上添加可学习的相对位置偏置：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right) V$$

<a id="formula-position-embedding-8"></a>
[📖 查看公式附录详解](#formula-position-embedding-8-detail)

**公式解释**
- **公式含义**：在注意力分数上加一个相对位置偏置矩阵 $B$，再 softmax。
- **变量说明**：$QK^T/\sqrt{d_k}$ 为内容分数；$B$ 为相对位置偏置矩阵（$B_{ij}$ 依赖 $i-j$）。
- **直觉/作用**：简单直接地让模型学习"距离越远/近"应该如何影响注意力。

其中 $B$ 是相对位置偏置矩阵。

### RoPE (Rotary Position Embedding)

RoPE 通过旋转矩阵将位置信息注入注意力：

$$\text{RoPE}(x_m, m) = \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \end{pmatrix} \otimes \begin{pmatrix} \cos(m\theta) \\ \sin(m\theta) \end{pmatrix}$$

<a id="formula-position-embedding-9"></a>
[📖 查看公式附录详解](#formula-position-embedding-9-detail)

**公式解释**
- **公式含义**：对向量每两维一组，乘以位置 $m$ 对应的旋转角度的正弦余弦。
- **变量说明**：$x_m^{(1)}, x_m^{(2)}$ 为第 $m$ 个位置的向量分量；$m\theta$ 为旋转角度；$\otimes$ 为逐元素乘。
- **直觉/作用**：位置信息通过旋转角度注入，相对位置相同则内积也相同。

核心性质：

$$\langle \text{RoPE}(x_m, m), \text{RoPE}(x_n, n) \rangle = \text{Re}[\langle x_m, x_n \rangle e^{i(m-n)\theta}]$$

<a id="formula-position-embedding-10"></a>
[📖 查看公式附录详解](#formula-position-embedding-10-detail)

**公式解释**
- **公式含义**：两个位置的 RoPE 编码的内积只依赖它们的相对位置 $m-n$。
- **变量说明**：$\langle \cdot, \cdot \rangle$ 为内积；$e^{i(m-n)\theta}$ 为复指数形式表示相对位置旋转。
- **直觉/作用**：无论绝对位置在哪，只要相对距离相同，注意力分数的结构就相同。

注意力分数只依赖于相对位置 $m - n$。

### ALiBi (Attention with Linear Biases)

ALiBi 在注意力分数上添加线性递减的偏置：

$$\text{score}_{ij} = q_i \cdot k_j - m \cdot |i - j|$$

<a id="formula-position-embedding-11"></a>
[📖 查看公式附录详解](#formula-position-embedding-11-detail)

**公式解释**
- **公式含义**：在内容分数上减去与距离成正比的惩罚项。
- **变量说明**：$q_i \cdot k_j$ 为内容分数；$m$ 为每个头的斜率（控制距离衰减速度）；$|i-j|$ 为绝对距离。
- **直觉/作用**：距离越远惩罚越大，模型自然偏好关注近处；无参数且外推能力强。

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

