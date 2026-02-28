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

<a id="formula-encoder-1"></a>
[📖 查看公式附录详解](#formula-encoder-1-detail)

**公式解释**
- **公式含义**：残差连接将输入 $x$ 与子层输出相加，再通过层归一化得到最终输出。
- **变量说明**：$x$ 为子层输入；$\text{Sublayer}(x)$ 为注意力或前馈网络的输出；$\text{LayerNorm}$ 为层归一化操作。
- **直觉/作用**：残差连接让梯度可以直接流过，缓解深层网络的梯度消失；层归一化稳定每层的输入分布。

### 完整公式

对于输入序列 $X \in \mathbb{R}^{n \times d}$：

**子层 1：Multi-Head Self-Attention**
$$Z = \text{LayerNorm}(X + \text{MultiHead}(X, X, X))$$

<a id="formula-encoder-2"></a>
[📖 查看公式附录详解](#formula-encoder-2-detail)

**公式解释**
- **公式含义**：对输入 $X$ 做多头自注意力，加残差后再做层归一化。
- **变量说明**：$X$ 为输入序列；$\text{MultiHead}$ 为多头注意力；$Z$ 为子层输出。
- **直觉/作用**：让每个位置都能"看到"全序列信息，残差和归一化保证训练稳定。

**子层 2：Feed-Forward Network**
$$Y = \text{LayerNorm}(Z + \text{FFN}(Z))$$

<a id="formula-encoder-3"></a>
[📖 查看公式附录详解](#formula-encoder-3-detail)

**公式解释**
- **公式含义**：对注意力输出 $Z$ 做前馈网络变换，加残差后再层归一化。
- **变量说明**：$Z$ 为上一层输出；$\text{FFN}$ 为逐位置前馈网络；$Y$ 为最终层输出。
- **直觉/作用**：FFN 对每个位置独立做非线性特征提取，增强表达能力。

其中 FFN 定义为：
$$\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2$$

<a id="formula-encoder-4"></a>
[📖 查看公式附录详解](#formula-encoder-4-detail)

**公式解释**
- **公式含义**：先升维再做 GELU 激活，最后降维回原维度。
- **变量说明**：$W_1$ 升维到 $4d$，$W_2$ 降回 $d$；GELU 为平滑激活函数。
- **直觉/作用**：扩展中间维度增加非线性表达能力，是 Transformer 的核心计算模块之一。

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

<a id="formula-encoder-5"></a>
[📖 查看公式附录详解](#formula-encoder-5-detail)

**公式解释**
- **公式含义**：第 $l$ 层的梯度等于最后一层直接传回的梯度加上中间各子层的梯度贡献。
- **变量说明**：$x_l$ 为第 $l$ 层输入；$L$ 为总层数；$f_i$ 为第 $i$ 个子层变换。
- **直觉/作用**：第一项（直连梯度）保证即使网络很深，梯度也能无损传回，防止梯度消失。

第一项保证梯度不会消失。

## 位置编码

由于自注意力没有顺序概念，需要位置编码：

### Sinusoidal（原始）

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$

<a id="formula-encoder-6"></a>
[📖 查看公式附录详解](#formula-encoder-6-detail)
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

<a id="formula-encoder-7"></a>
[📖 查看公式附录详解](#formula-encoder-7-detail)

**公式解释**
- **公式含义**：偶数维度用正弦、奇数维度用余弦编码位置信息。
- **变量说明**：$pos$ 为位置索引；$i$ 为维度索引；$d$ 为嵌入维度。
- **直觉/作用**：不同频率的正弦波组合，让模型能区分相对位置关系；无参数且可外推到任意长度。

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

<a id="formula-encoder-8"></a>
[📖 查看公式附录详解](#formula-encoder-8-detail)

**公式解释**
- **公式含义**：BERT 输入由三种嵌入相加得到：词嵌入、句子嵌入、位置嵌入。
- **变量说明**：Token Embedding 表示词本身；Segment Embedding 区分句子对；Position Embedding 编码位置。
- **直觉/作用**：将词语义、句子归属、位置信息统一编码到同一向量空间。

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

<a id="formula-encoder-9"></a>
[📖 查看公式附录详解](#formula-encoder-9-detail)

**公式解释**
- **公式含义**：用 `[MASK]` 位置的隐藏状态 $h$ 与词嵌入矩阵 $E$ 计算相似度，再用 softmax 得到词概率分布。
- **变量说明**：$h$ 为 `[MASK]` 位置的表示向量；$E$ 为词嵌入矩阵；$E^T$ 为其转置。
- **直觉/作用**：将隐藏状态映射回词表空间，预测被遮盖的原始词。

其中 $h$ 是 `[MASK]` 位置的隐藏状态，$E$ 是词嵌入矩阵。

#### Next Sentence Prediction (NSP)

判断两个句子是否连续：

$$P(isNext) = \text{sigmoid}(h_{[CLS]} \cdot w)$$

<a id="formula-encoder-10"></a>
[📖 查看公式附录详解](#formula-encoder-10-detail)

**公式解释**
- **公式含义**：用 `[CLS]` 位置的隐藏状态做二分类，判断两句子是否连续。
- **变量说明**：$h_{[CLS]}$ 为 `[CLS]` token 的表示；$w$ 为分类权重向量；$\text{sigmoid}$ 将输出压缩到 $(0,1)$。
- **直觉/作用**：`[CLS]` 聚合了全句信息，用于句子级别的预测任务。

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

<a id="formula-encoder-11"></a>
[📖 查看公式附录详解](#formula-encoder-11-detail)

**公式解释**
- **公式含义**：编码器中每个位置的表示都依赖全序列信息。
- **变量说明**：$h_i$ 为位置 $i$ 的输出表示；$x_1, ..., x_n$ 为序列所有位置的输入。
- **直觉/作用**：双向建模让每个词都能"看到"前后文，更适合理解任务。

对比 RNN/LSTM 只能看到左侧上下文。

### 2. 并行计算

所有位置可以同时计算，不像 RNN 需要顺序处理。

### 3. 长距离依赖

任意两个位置之间的路径长度为 O(1)，不像 RNN 是 O(n)。

## 实现细节

### 参数共享

在 BERT 中，词嵌入矩阵和输出分类器共享：

$$\text{MLM\_logits} = h \cdot E^T + b$$

<a id="formula-encoder-12"></a>
[📖 查看公式附录详解](#formula-encoder-12-detail)

**公式解释**
- **公式含义**：预测被遮盖词时，直接复用输入词嵌入矩阵作为输出投影。
- **变量说明**：$h$ 为隐藏状态；$E^T$ 为词嵌入矩阵转置；$b$ 为偏置。
- **直觉/作用**：共享矩阵减少参数量，同时输入和输出语义对齐更一致。

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

