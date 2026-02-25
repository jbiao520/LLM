# 激活函数（Activation Function）深入版

> 面向有机器学习基础读者的技术详解

## 概述

激活函数是神经网络的灵魂，为网络引入非线性变换能力。本文深入分析主流激活函数的数学原理、梯度特性及其在深度学习中的应用。

## 为什么需要激活函数？

### 万能近似定理

具有非线性激活函数的前馈网络，只要有足够的隐藏单元，就可以以任意精度近似任何连续函数。

$$f(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

没有 $\sigma$ 的非线性，网络只能表示线性变换。

## ReLU (Rectified Linear Unit)

### 数学定义

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

### 导数

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

### 优点

1. **计算高效**：只需要比较和赋值
2. **缓解梯度消失**：正区间的梯度恒为 1
3. **稀疏激活**：负区间输出为 0，产生稀疏性

### 问题：Dead ReLU

当输入始终为负时，神经元"死亡"，梯度永远为 0。

**解决方案**：
- Leaky ReLU: $\max(\alpha x, x)$，其中 $\alpha \approx 0.01$
- Parametric ReLU: $\alpha$ 可学习

## Sigmoid

### 数学定义

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### 导数

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

### 问题

1. **梯度消失**：当 $|x|$ 较大时，$\sigma'(x) \approx 0$
2. **非零中心**：输出恒为正，导致权重梯度符号一致
3. **指数计算**：计算成本较高

### 变体：Hard Sigmoid

$$\text{HardSigmoid}(x) = \max(0, \min(1, \frac{x + 1}{2}))$$

用分段线性近似 sigmoid，计算更快。

## Tanh (Hyperbolic Tangent)

### 数学定义

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

### 导数

$$\tanh'(x) = 1 - \tanh^2(x)$$

### 特点

- 输出范围：$(-1, 1)$，零中心
- 梯度比 sigmoid 更大（在 0 附近）
- 仍存在梯度消失问题

## GELU (Gaussian Error Linear Unit)

### 数学定义

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \leq x)$$

其中 $\Phi(x)$ 是标准正态分布的 CDF：

$$\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

### 近似公式

精确的 GELU 计算 expensive，常用近似：

**Tanh 近似**：
$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 \cdot x^3)\right]\right)$$

**Sigmoid 近似**（SiLU/Swish）：
$$\text{SiLU}(x) = x \cdot \sigma(x)$$

### 导数

$$\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x)$$

其中 $\phi(x)$ 是标准正态分布的 PDF。

### 为什么 GELU 更适合 Transformer？

1. **平滑性**：在 0 处可导，梯度更稳定
2. **非单调性**：对负输入有非零响应
3. **随机正则化解释**：可解释为随机 dropout 的确定性版本

## Swish

### 数学定义

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

当 $\beta = 1$ 时，Swish 等价于 SiLU。

### 特点

- 平滑、非单调
- 无上界、有下界
- 在深层网络中表现优于 ReLU

## Softmax

### 数学定义

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

### 导数

$$\frac{\partial \text{Softmax}(x_i)}{\partial x_j} = \text{Softmax}(x_i)(\delta_{ij} - \text{Softmax}(x_j))$$

其中 $\delta_{ij}$ 是 Kronecker delta。

### 数值稳定性

为防止指数溢出，通常减去最大值：

$$\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j=1}^{K} e^{x_j - \max(x)}}$$

### 温度参数 (Temperature)

$$\text{Softmax}(x_i, T) = \frac{e^{x_i/T}}{\sum_{j=1}^{K} e^{x_j/T}}$$

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
