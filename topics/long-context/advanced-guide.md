# 长上下文（Long Context）深入版

> 面向有机器学习基础读者的技术详解

## 概述

扩展 LLM 上下文长度是当前研究热点。本文深入分析 RoPE 缩放、YaRN、ALiBi 等长上下文技术的数学原理。

## RoPE 回顾

### 旋转位置编码

$$\text{RoPE}(x_m, m) = \begin{pmatrix} x_m^{(1)} \\ x_m^{(2)} \\ \vdots \\ x_m^{(d-1)} \\ x_m^{(d)} \end{pmatrix} \odot \begin{pmatrix} \cos(m\theta_1) \\ \cos(m\theta_1) \\ \vdots \\ \cos(m\theta_{d/2}) \\ \cos(m\theta_{d/2}) \end{pmatrix} + \begin{pmatrix} -x_m^{(2)} \\ x_m^{(1)} \\ \vdots \\ -x_m^{(d)} \\ x_m^{(d-1)} \end{pmatrix} \odot \begin{pmatrix} \sin(m\theta_1) \\ \sin(m\theta_1) \\ \vdots \\ \sin(m\theta_{d/2}) \\ \sin(m\theta_{d/2}) \end{pmatrix}$$

<a id="formula-longcontext-1"></a>

其中 $\theta_i = 10000^{-2i/d}$。

**公式解释**
- **公式含义**：通过旋转编码位置信息。
- **变量说明**：$m$ 是位置；$\theta_i$ 是频率。
- **直觉/作用**：位置通过旋转角度编码，相对位置可以通过点积恢复。

## 位置插值（Position Interpolation）

### 线性缩放

将位置 $m \in [0, L_{new}]$ 映射到 $m' \in [0, L_{train}]$：

$$m' = \frac{m \cdot L_{train}}{L_{new}} = \frac{m}{s}$$

<a id="formula-longcontext-2"></a>

其中 $s = L_{new} / L_{train}$ 是缩放因子。

**公式解释**
- **公式含义**：将新位置线性压缩到训练范围内。
- **变量说明**：$s$ 是缩放因子；$L_{train}$ 是训练长度。
- **直觉/作用**：简单有效，但可能损失分辨率。

### 效果

| 缩放因子 | 扩展后长度 | 效果 |
|---------|-----------|------|
| 1x | 4K | 基线 |
| 2x | 8K | 几乎无损 |
| 4x | 16K | 轻微下降 |
| 8x | 32K | 可接受 |
| 16x+ | 64K+ | 明显下降 |

## 动态 NTK 缩放

### 动机

线性缩放的问题：固定缩放因子不够灵活。

### 方法

根据当前序列长度动态计算缩放因子：

$$s = \begin{cases} 1 & \text{if } L \leq L_{train} \\ \frac{\alpha \cdot L_{train}}{L} + 1 - \alpha & \text{if } L > L_{train} \end{cases}$$

<a id="formula-longcontext-3"></a>

或者使用 NTK 感知的缩放：

$$\theta'_i = \theta_i \cdot \left( \frac{s \cdot L_{train}}{L_{train}} \right)^{d/(d-2)}$$

<a id="formula-longcontext-4"></a>

**公式解释**
- **公式含义**：根据长度动态调整频率基数。
- **变量说明**：$\alpha$ 是混合系数。
- **直觉/作用**：短序列保持原始编码，长序列动态调整。

## YaRN（Yet another RoPE extensioN）

### 组合技术

YaRN 结合了三种技术：

1. **温度缩放**：调整 softmax 温度
2. **厚度缩放**：调整 RoPE 频率
3. **动态缩放**：根据长度调整

### 温度缩放

$$\text{Attention}(q, k) = \text{softmax}\left(\frac{q \cdot k^T}{\sqrt{d} \cdot t}\right)$$

<a id="formula-longcontext-5"></a>

其中 $t > 1$ 是温度参数，随缩放因子增大：

$$t = 1 + 0.32 \cdot \log(s)$$

<a id="formula-longcontext-6"></a>

**公式解释**
- **公式含义**：放大温度使注意力分布更平滑。
- **变量说明**：$t$ 是温度；$s$ 是缩放因子。
- **直觉/作用**：补偿位置编码拉伸带来的注意力变化。

### 厚度缩放

$$\theta'_i = \theta_i \cdot \lambda^i$$

<a id="formula-longcontext-7"></a>

其中 $\lambda$ 根据目标扩展比例计算。

### YaRN 效果

| 方法 | 4K→8K | 4K→32K | 4K→128K |
|------|-------|--------|---------|
| 线性 | 良好 | 下降 | 差 |
| 动态 NTK | 良好 | 良好 | 下降 |
| YaRN | 良好 | 良好 | 良好 |

## ALiBi（Attention with Linear Biases）

### 原理

不使用位置编码，而是在注意力计算中加入线性偏置：

$$\text{Attention}(q_i, k_j) = q_i \cdot k_j - m \cdot |i - j|$$

<a id="formula-longcontext-8"></a>

其中 $m$ 是每个头不同的斜率。

**公式解释**
- **公式含义**：距离越远，注意力分数越低。
- **变量说明**：$m$ 是斜率参数（每个头不同）。
- **直觉/作用**：位置信息直接编码在注意力偏置中。

### 斜率设置

$$m_h = \frac{1}{2^{\frac{8}{n} \cdot (h+1)}}$$

<a id="formula-longcontext-9"></a>

其中 $h$ 是头的索引，$n$ 是总头数。

### 外推能力

ALiBi 天然支持外推到更长序列，因为偏置是相对的。

## LongLoRA

### 思想

将长序列分成短块，在块内做注意力，通过滑动窗口覆盖全局：

$$\text{Complexity}: O(n \cdot w) \text{ instead of } O(n^2)$$

<a id="formula-longcontext-10"></a>

其中 $w$ 是窗口大小。

## Flash Attention

虽然不直接增加上下文，但大幅降低内存：

$$\text{Memory}: O(n) \text{ instead of } O(n^2)$$

<a id="formula-longcontext-11"></a>

## 参考文献

1. Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*
2. Chen et al. (2023). *Extending Context Window of LLMs via Positional Interpolation*
3. Peng et al. (2023). *YaRN: Efficient Context Window Extension of Large Language Models*
4. Press et al. (2021). *Train Short, Test Long: Attention with Linear Biases*
5. Chen et al. (2023). *LongLoRA: Efficient Fine-tuning of Long-Context LLMs*
