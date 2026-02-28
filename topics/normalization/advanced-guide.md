# 归一化（Normalization）深入版

> 面向有机器学习基础读者的技术详解

## 概述

归一化技术是深度学习中稳定训练、加速收敛的关键组件。本文深入分析 BatchNorm、LayerNorm 和 RMSNorm 的数学原理及其在 LLM 中的应用。

## 为什么需要归一化？

### 内部协变量偏移 (Internal Covariate Shift)

深层网络中，每层输入的分布在训练过程中不断变化，导致：
1. 后层需要不断适应新的输入分布
2. 学习率需要设置得很小
3. 训练收敛慢，容易陷入饱和区

归一化通过稳定每层输入的分布来缓解这个问题。

## Batch Normalization

### 数学定义

对于 mini-batch 中的第 $i$ 个样本的第 $k$ 维特征：

$$\hat{x}_{i,k} = \frac{x_{i,k} - \mu_k}{\sqrt{\sigma_k^2 + \epsilon}}$$

<a id="formula-normalization-1"></a>
[📖 查看公式附录详解](#formula-normalization-1-detail)

$$y_{i,k} = \gamma_k \hat{x}_{i,k} + \beta_k$$

<a id="formula-normalization-2"></a>
[📖 查看公式附录详解](#formula-normalization-2-detail)

**公式解释**
- **公式含义**：对每个特征维度 $k$，减去 batch 内均值、除以标准差，再缩放平移。
- **变量说明**：$\mu_k, \sigma_k^2$ 为第 $k$ 维在 batch 内的均值和方差；$\gamma_k, \beta_k$ 为可学习的缩放和偏移参数。
- **直觉/作用**：将每维特征归一化到标准分布，再通过 $\gamma, \beta$ 恢复表达能力；$\epsilon$ 防止除零。

其中：
- $\mu_k = \frac{1}{m}\sum_{i=1}^{m} x_{i,k}$（batch 内第 $k$ 维的均值）
- $\sigma_k^2 = \frac{1}{m}\sum_{i=1}^{m} (x_{i,k} - \mu_k)^2$（batch 内第 $k$ 维的方差）
- $\gamma_k, \beta_k$ 是可学习参数
- $\epsilon$ 是数值稳定性常数（如 $10^{-5}$）

### 特点

- **归一化方向**：沿 batch 维度
- **训练/推理差异**：推理时使用训练时累积的 running mean/var
- **依赖 batch size**：小 batch 效果差

### 梯度计算

$$\frac{\partial L}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \left( \frac{\partial L}{\partial y_i} - \frac{1}{m}\sum_{j=1}^{m}\frac{\partial L}{\partial y_j} - \frac{\hat{x}_i}{m}\sum_{j=1}^{m}\frac{\partial L}{\partial y_j}\hat{x}_j \right)$$

<a id="formula-normalization-3"></a>
[📖 查看公式附录详解](#formula-normalization-3-detail)

**公式解释**
- **公式含义**：BatchNorm 反向传播时，梯度不仅依赖自身输出，还与 batch 内所有样本相关。
- **变量说明**：$m$ 为 batch 大小；$\hat{x}_i$ 为归一化后的值；$\partial L / \partial y_j$ 为上游梯度。
- **直觉/作用**：归一化操作的梯度会"分散"到整个 batch，增加训练的正规化效果。

## Layer Normalization

### 数学定义

对单个样本的所有特征进行归一化：

$$\mu = \frac{1}{H}\sum_{i=1}^{H} x_i$$

<a id="formula-normalization-4"></a>
[📖 查看公式附录详解](#formula-normalization-4-detail)

$$\sigma^2 = \frac{1}{H}\sum_{i=1}^{H} (x_i - \mu)^2$$

<a id="formula-normalization-5"></a>
[📖 查看公式附录详解](#formula-normalization-5-detail)

$$y = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \odot (x - \mu) + \beta$$

<a id="formula-normalization-6"></a>
[📖 查看公式附录详解](#formula-normalization-6-detail)

**公式解释**
- **公式含义**：对单个样本的所有特征维计算均值和方差，然后归一化。
- **变量说明**：$H$ 为特征维度数；$\mu, \sigma^2$ 为单样本内的统计量；$\gamma, \beta$ 为可学习参数。
- **直觉/作用**：与 BatchNorm 不同，不依赖 batch 大小，适合序列模型和 NLP 任务。

其中 $H$ 是特征维度数。

### 与 BatchNorm 的区别

| 特性 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化维度 | batch 维 | feature 维 |
| 对 batch size 依赖 | 强 | 无 |
| 训练/推理差异 | 有 | 无 |
| 适用场景 | CV | NLP/Transformer |

### Transformer 中的应用

在 Transformer 中，LayerNorm 通常有两种位置：

1. **Post-LN**（原始 Transformer）：
   ```
   x = LayerNorm(x + Sublayer(x))
   ```

2. **Pre-LN**（现代 Transformer）：
   ```
   x = x + Sublayer(LayerNorm(x))
   ```

Pre-LN 训练更稳定，梯度更平滑。

## RMS Normalization

### 数学定义

RMSNorm 简化了 LayerNorm，去掉了均值中心化：

$$\text{RMS}(x) = \sqrt{\frac{1}{H}\sum_{i=1}^{H} x_i^2}$$

<a id="formula-normalization-7"></a>
[📖 查看公式附录详解](#formula-normalization-7-detail)

$$y = \frac{x}{\text{RMS}(x)} \cdot \gamma$$

<a id="formula-normalization-8"></a>
[📖 查看公式附录详解](#formula-normalization-8-detail)

**公式解释**
- **公式含义**：只用均方��归一化，省去均值计算，再乘以可学习缩放因子。
- **变量说明**：$\text{RMS}(x)$ 为均方根值；$\gamma$ 为缩放参数；$H$ 为特征维度数。
- **直觉/作用**：简化计算但保持归一化效果，LLaMA 等现代 LLM 广泛使用。

或等价地：

$$y = \frac{\gamma}{\sqrt{\frac{1}{H}\sum_{i=1}^{H} x_i^2 + \epsilon}} \cdot x$$

<a id="formula-normalization-9"></a>
[📖 查看公式附录详解](#formula-normalization-9-detail)

**公式解释**
- **公式含义**：将 RMSNorm 写成与 LayerNorm 类似的形式，方便对比。
- **变量说明**：分母是均方根加 $\epsilon$，分子是缩放参数 $\gamma$。
- **直觉/作用**：与 LayerNorm 的区别在于没有 $(x - \mu)$，即不做中心化。

### 为什么 RMSNorm 有效？

1. **理论分析**：中心化操作对于再缩放（re-scaling）不变性并非必需
2. **计算效率**：省去均值计算，减少约 25% 计算量
3. **实践效果**：在 LLM 中效果与 LayerNorm 相当或更好

### LLaMA 中的 RMSNorm 实现

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [batch, seq_len, dim]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

## 数值稳定性

### Epsilon 的重要性

$\epsilon$ 的作用是防止除零错误，但也影响数值精度：
- 太大：归一化效果被削弱
- 太小：可能出现数值不稳定

常用值：$\epsilon = 10^{-5}$ 或 $10^{-6}$

### 混合精度训练中的归一化

在 FP16 训练中，归一化层通常保持 FP32 精度：
- 方差计算涉及平方，容易溢出
- 除法操作需要高精度

## 不同归一化方法的对比

| 方法 | 公式复杂度 | 计算量 | 参数量 | LLM 使用率 |
|------|-----------|--------|--------|-----------|
| BatchNorm | 高 | 中 | 2H | 低 |
| LayerNorm | 中 | 中 | 2H | 中 |
| RMSNorm | 低 | 低 | H | 高 |

## 参考文献

1. Ioffe & Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*
2. Ba et al. (2016). *Layer Normalization*
3. Zhang & Sennrich (2019). *Root Mean Square Layer Normalization*
4. Xiong et al. (2020). *On Layer Normalization in the Transformer Architecture*

