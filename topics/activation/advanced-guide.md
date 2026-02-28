# 激活函数（Activation Function）深入版

> 面向有机器学习基础读者的技术详解

## 概述

激活函数是神经网络的灵魂，为网络引入非线性变换能力。本文深入分析主流激活函数的数学原理、梯度特性及其在深度学习中的应用。

## 为什么需要激活函数？

### 万能近似定理

具有非线性激活函数的前馈网络，只要有足够的隐藏单元，就可以以任意精度近似任何连续函数。

$$f(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

<a id="formula-activation-1"></a>
[📖 查看公式附录详解](#formula-activation-1-detail)

没有 $\sigma$ 的非线性，网络只能表示线性变换。

**公式解释**
- **公式含义**：一个两层前馈网络的计算过程：先做线性变换 $W_1 x + b_1$，再经过激活函数 $\sigma$，最后再线性变换得到输出。
- **变量说明**：$x$ 为输入向量；$W_1, W_2$ 为权重矩阵；$b_1, b_2$ 为偏置；$\sigma$ 为激活函数（逐元素作用）。
- **直觉/作用**：$\sigma$ 提供非线性，使模型可以表达复杂函数；若没有 $\sigma$，多层线性变换仍等价于一次线性变换。

## ReLU (Rectified Linear Unit)

### 数学定义

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

<a id="formula-activation-2"></a>
[📖 查看公式附录详解](#formula-activation-2-detail)

**公式解释**
- **公式含义**：ReLU 把所有负值截断为 0，正值保持不变。
- **变量说明**：$x$ 是输入标量或向量（逐元素应用）。
- **直觉/作用**：像一个“闸门”，只让正信号通过，使激活稀疏且计算简单。

### 导数

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

<a id="formula-activation-3"></a>
[📖 查看公式附录详解](#formula-activation-3-detail)

**公式解释**
- **公式含义**：ReLU 在正区间导数为 1，负区间导数为 0。
- **变量说明**：$\text{ReLU}'(x)$ 表示输出对输入的变化率。
- **直觉/作用**：正区间梯度稳定、易训练；负区间梯度为 0，可能导致神经元“死亡”。

### 优点

1. **计算高效**：只需要比较和赋值
2. **缓解梯度消失**：正区间的梯度恒为 1
3. **稀疏激活**：负区间输出为 0，产生稀疏性

### 问题：Dead ReLU

当输入始终为负时，神经元"死亡"，梯度永远为 0。

**解决方案**：
- Leaky ReLU: $\max(\alpha x, x)$，其中 $\alpha \approx 0.01$
- Parametric ReLU: $\alpha$ 可学习

**公式解释**
- **Leaky ReLU**：当 $x<0$ 时仍保留 $\alpha x$ 的小斜率，避免完全“关死”。
- **Parametric ReLU**：$\alpha$ 由训练数据自动学习，适配不同层的最佳负区间斜率。

## Sigmoid

### 数学定义

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

<a id="formula-activation-4"></a>
[📖 查看公式附录详解](#formula-activation-4-detail)

**公式解释**
- **公式含义**：将任意实数映射到 $(0, 1)$ 的概率式输出。
- **变量说明**：$x$ 为输入；$e$ 为自然常数。
- **直觉/作用**：像一个“概率开关”，常用于二分类输出。

### 导数

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

<a id="formula-activation-5"></a>
[📖 查看公式附录详解](#formula-activation-5-detail)

**公式解释**
- **公式含义**：Sigmoid 的梯度与其输出成正相关，在 0 附近最大。
- **变量说明**：$\sigma'(x)$ 表示输出对输入的敏感度。
- **直觉/作用**：当 $\sigma(x)$ 接近 0 或 1 时，梯度趋近 0，容易导致梯度消失。

### 问题

1. **梯度消失**：当 $|x|$ 较大时，$\sigma'(x) \approx 0$
2. **非零中心**：输出恒为正，导致权重梯度符号一致
3. **指数计算**：计算成本较高

### 变体：Hard Sigmoid

$$\text{HardSigmoid}(x) = \max(0, \min(1, \frac{x + 1}{2}))$$

<a id="formula-activation-6"></a>
[📖 查看公式附录详解](#formula-activation-6-detail)

**公式解释**
- **公式含义**：用分段线性函数近似 Sigmoid，把输出限制在 $[0,1]$。
- **变量说明**：当 $x \le -1$ 输出 0；$x \ge 1$ 输出 1；中间线性变化。
- **直觉/作用**：牺牲精度换取更快计算和更稳定的梯度。

用分段线性近似 sigmoid，计算更快。

## Tanh (Hyperbolic Tangent)

### 数学定义

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

<a id="formula-activation-7"></a>
[📖 查看公式附录详解](#formula-activation-7-detail)

**公式解释**
- **公式含义**：双曲正切把输入映射到 $(-1, 1)$，是“零中心”的 Sigmoid 变体。
- **变量说明**：$x$ 为输入；等式右侧说明它与 Sigmoid 的关系。
- **直觉/作用**：输出均值更接近 0，有利于优化，但仍可能梯度消失。

### 导数

$$\tanh'(x) = 1 - \tanh^2(x)$$

<a id="formula-activation-8"></a>
[📖 查看公式附录详解](#formula-activation-8-detail)

**公式解释**
- **公式含义**：tanh 的梯度由输出值决定，输出越接近 $\pm1$，梯度越小。
- **变量说明**：$\tanh'(x)$ 是导数，衡量输出对输入的变化率。
- **直觉/作用**：当激活饱和时（接近 -1 或 1），学习速度变慢。

### 特点

- 输出范围：$(-1, 1)$，零中心
- 梯度比 sigmoid 更大（在 0 附近）
- 仍存在梯度消失问题

## GELU (Gaussian Error Linear Unit)

### 数学定义

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \leq x)$$

<a id="formula-activation-9"></a>
[📖 查看公式附录详解](#formula-activation-9-detail)

**公式解释**
- **公式含义**：GELU 将输入 $x$ 与其被“保留”的概率相乘。
- **变量说明**：$X \sim \mathcal{N}(0,1)$；$\Phi(x)$ 是标准正态分布 CDF。
- **直觉/作用**：输入越大，保留概率越高；输入为负时仍可能保留一部分。

其中 $\Phi(x)$ 是标准正态分布的 CDF：

$$\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

<a id="formula-activation-10"></a>
[📖 查看公式附录详解](#formula-activation-10-detail)

**公式解释**
- **公式含义**：标准正态分布的累计概率函数（CDF）。
- **变量说明**：$\text{erf}(\cdot)$ 是误差函数。
- **直觉/作用**：给出 $X \le x$ 的概率，用于平滑地“筛选”输入。

### 近似公式

精确的 GELU 计算 expensive，常用近似：

**Tanh 近似**：
$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right]\right)$$

<a id="formula-activation-11"></a>
[📖 查看公式附录详解](#formula-activation-11-detail)

**公式解释**
- **公式含义**：用 $\tanh$ 近似 GELU，避免计算误差函数带来的开销。
- **变量说明**：$0.044715$ 为经验常数，$x^3$ 提供非线性调节。
- **直觉/作用**：在保证形状接近 GELU 的同时提升计算效率。

**Sigmoid 近似**（SiLU/Swish）：
$$\text{SiLU}(x) = x \cdot \sigma(x)$$

<a id="formula-activation-12"></a>
[📖 查看公式附录详解](#formula-activation-12-detail)

**公式解释**
- **公式含义**：输入 $x$ 与 Sigmoid 门控后的结果相乘。
- **变量说明**：$\sigma(x)$ 为 Sigmoid。
- **直觉/作用**：既保留正值，又在负值区保留小响应，平滑且非单调。

### 导数

$$\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)$$

<a id="formula-activation-13"></a>
[📖 查看公式附录详解](#formula-activation-13-detail)

**公式解释**
- **公式含义**：GELU 的梯度由 CDF 和 PDF 两部分组成。
- **变量说明**：$\phi(x)$ 是标准正态分布的概率密度函数（PDF）。
- **直觉/作用**：梯度随 $x$ 平滑变化，训练更稳定。

其中 $\phi(x)$ 是标准正态分布的 PDF。

### 为什么 GELU 更适合 Transformer？

1. **平滑性**：在 0 处可导，梯度更稳定
2. **非单调性**：对负输入有非零响应
3. **随机正则化解释**：可解释为随机 dropout 的确定性版本

## Swish

### 数学定义

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

<a id="formula-activation-14"></a>
[📖 查看公式附录详解](#formula-activation-14-detail)

**公式解释**
- **公式含义**：在 Sigmoid 门控后再乘以输入 $x$。
- **变量说明**：$\beta$ 控制门控的“陡峭度”，可为常数或可学习。
- **直觉/作用**：$\beta$ 越大，函数越接近 ReLU；$\beta$ 越小，越平滑。

当 $\beta = 1$ 时，Swish 等价于 SiLU。

### 特点

- 平滑、非单调
- 无上界、有下界
- 在深层网络中表现优于 ReLU

## Softmax

### 数学定义

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

<a id="formula-activation-15"></a>
[📖 查看公式附录详解](#formula-activation-15-detail)

**公式解释**
- **公式含义**：把一组实数 $x_i$ 变成概率分布（和为 1）。
- **变量说明**：$K$ 为类别数；$x_i$ 为第 $i$ 类的打分。
- **直觉/作用**：分数越大，指数放大效应越强，概率越高。

### 导数

$$\frac{\partial \text{Softmax}(x_i)}{\partial x_j} = \text{Softmax}(x_i)(\delta_{ij} - \text{Softmax}(x_j))$$

<a id="formula-activation-16"></a>
[📖 查看公式附录详解](#formula-activation-16-detail)

其中 $\delta_{ij}$ 是 Kronecker delta。

**公式解释**
- **公式含义**：Softmax 的梯度不仅与自身有关，也与其他类别的概率相关。
- **变量说明**：$\delta_{ij}=1$ 当 $i=j$，否则为 0。
- **直觉/作用**：提升某一类概率会挤压其他类概率，体现“竞争关系”。

### 数值稳定性

为防止指数溢出，通常减去最大值：

$$\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{K} e^{x_j - \max(x)}}$$

<a id="formula-activation-17"></a>
[📖 查看公式附录详解](#formula-activation-17-detail)

**公式解释**
- **公式含义**：在所有 $x_i$ 上减去最大值不改变结果，但避免指数溢出。
- **变量说明**：$\max(x)$ 为所有输入中的最大值。
- **直觉/作用**：数值稳定性技巧，输出概率不变。

### 温度参数 (Temperature)

$$\text{Softmax}(x_i, T) = \frac{e^{x_i/T}}{\sum_{j=1}^{K} e^{x_j/T}}$$

<a id="formula-activation-18"></a>
[📖 查看公式附录详解](#formula-activation-18-detail)

**公式解释**
- **公式含义**：加入温度参数 $T$ 来控制分布的“尖锐程度”。
- **变量说明**：$T$ 越大，分布越平滑；$T$ 越小，分布越尖锐。
- **直觉/作用**：可用于采样时控制随机性。

- $T > 1$：分布更平滑（更随机）
- $T < 1$：分布更尖锐（更确定）
- $T \to 0$：趋近于 argmax
- $T \to \infty$：趋近于均匀分布

## 激活函数对比

| 激活函数 | 公式复杂度 | 梯度问题 | 计算效率 | 适用场景 |
|---------|-----------|---------|---------|---------|
| ReLU | 低 | Dead ReLU | 高 | 通用 |
| LeakyReLU | 低 | 较少 | 高 | 替代 ReLU |
| Sigmoid | 高 | 梯度消失 | 中 | 二分类输出 |
| Tanh | 高 | 梯度消失 | 中 | RNN/LSTM |
| GELU | 高 | 较少 | 中 | Transformer |
| Swish | 高 | 较少 | 中 | 深层网络 |

## 选择建议

1. **隐藏层默认选择**：ReLU
2. **Transformer 架构**：GELU
3. **二分类输出**：Sigmoid
4. **多分类输出**：Softmax
5. **RNN/GRU/LSTM**：Tanh

## 参考文献

1. Nair & Hinton (2010). *Rectified Linear Units Improve Restricted Boltzmann Machines*
2. Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs)*
3. Ramachandran et al. (2017). *Searching for Activation Functions*
4. Maas et al. (2013). *Rectifier Nonlinearities Improve Neural Network Acoustic Models*

