# 预训练（Pre-training）深入版

> 面向有机器学习基础读者的技术详解

## 概述

预训练是 LLM 获得通用语言能力的关键阶段。本文深入分析预训练的目标函数、数据处理、优化策略和缩放定律。

## 自回归语言建模

### 目标函数

给定文本序列 $x = (x_1, x_2, ..., x_T)$，最大化似然：

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$

<a id="formula-pretraining-1"></a>

**公式解释**
- **公式含义**：最大化每个位置预测正确 token 的对数概率之和。
- **变量说明**：$x_t$ 是第 $t$ 个 token；$x_{<t}$ 是之前的 token；$\theta$ 是模型参数。
- **直觉/作用**：让模型学会预测下一个词，从而掌握语言规律。

### 交叉熵损失

等价于：

$$\mathcal{L} = -\sum_{t=1}^{T} \log \frac{e^{z_{x_t}}}{\sum_{v \in V} e^{z_v}}$$

<a id="formula-pretraining-2"></a>

其中 $z$ 是模型输出的 logits。

**公式解释**
- **公式解释**：这是 softmax 交叉熵的展开形式。
- **变量说明**：$z_{x_t}$ 是正确 token 的 logit；$V$ 是词汇表。
- **直觉/作用**：正确 token 的概率越高，损失越低。

## 困惑度（Perplexity）

### 定义

$$\text{PPL} = \exp\left(\frac{1}{T} \sum_{t=1}^{T} -\log P(x_t | x_{<t})\right)$$

<a id="formula-pretraining-3"></a>

**公式解释**
- **公式含义**：平均每个位置的负对数概率的指数。
- **变量说明**：$T$ 是序列长度。
- **直觉/作用**：PPL 越低，模型越好；可以理解为"平均分支因子"。

### 直观理解

PPL = 10 意味着模型在每个位置平均在 10 个候选中犹豫。

| PPL | 质量 |
|-----|------|
| 100+ | 很差 |
| 30-50 | 一般 |
| 10-20 | 良好 |
| <10 | 优秀 |

## 数据组成

### 典型数据配比

| 数据源 | LLaMA 比例 | 作用 |
|--------|-----------|------|
| CommonCrawl | 67% | 通用知识 |
| C4 | 15% | 高质量网页 |
| GitHub | 5% | 代码能力 |
| Wikipedia | 4.5% | 事实知识 |
| Books | 4.5% | 长文本理解 |
| arXiv | 2.5% | 科学知识 |
| StackExchange | 2% | 问答能力 |

### 数据质量评分

使用小模型对数据质量打分：

$$\text{score}(d) = \text{Model}_{small}(d) \cdot \text{heuristics}(d)$$

保留高质量数据，丢弃低质量数据。

## 优化策略

### 学习率调度

**Warmup + Cosine Decay**:

$$\eta_t = \begin{cases}
\eta_{max} \cdot \frac{t}{T_{warmup}} & t < T_{warmup} \\
\eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t - T_{warmup}}{T_{total} - T_{warmup}} \pi)) & t \geq T_{warmup}
\end{cases}$$

<a id="formula-pretraining-4"></a>

**公式解释**
- **公式含义**：先线性升温，再余弦降温。
- **变量说明**：$\eta_t$ 是第 $t$ 步的学习率；$T_{warmup}$ 是预热步数。
- **直觉/作用**：稳定训练初期，后期逐渐收敛。

### 梯度裁剪

防止梯度爆炸：

$$g' = \begin{cases}
g & \|g\| \leq c \\
\frac{c \cdot g}{\|g\|} & \|g\| > c
\end{cases}$$

<a id="formula-pretraining-5"></a>

**公式解释**
- **公式含义**：当梯度范数超过阈值时，按比例缩小。
- **变量说明**：$c$ 是裁剪阈值（通常 1.0）。
- **直觉/作用**：防止单次更新步长过大。

### AdamW 优化器

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \eta \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)$$

<a id="formula-pretraining-6"></a>

典型参数：$\beta_1 = 0.9$, $\beta_2 = 0.95$, $\lambda = 0.1$

## 缩放定律

### Kaplan 定律

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

<a id="formula-pretraining-7"></a>

**公式解释**
- **公式含义**：损失与模型大小的幂次关系。
- **变量说明**：$N$ 是参数量；$N_c$, $\alpha_N$ 是拟合常数。
- **直觉/作用**：模型越大，损失越低（幂次衰减）。

### Chinchilla 定律

最优计算分配：

$$N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$$

<a id="formula-pretraining-8"></a>

**公式解释**
- **公式含义**：给定计算预算 $C$，最优的模型大小和数据量。
- **变量说明**：$N_{opt}$ 是最优参数量；$D_{opt}$ 是最优数据量。
- **直觉/作用**：模型和数据应该平衡增长。

具体公式：

$$N_{opt} = 0.6 \cdot C^{0.56}, \quad D_{opt} = 0.3 \cdot C^{0.54}$$

### 计算预算

训练总计算量：

$$C \approx 6 \cdot N \cdot D$$

<a id="formula-pretraining-9"></a>

**公式解释**
- **公式含义**：总 FLOPs 约为 6 倍的参数量乘以数据量。
- **变量说明**：$N$ 是参数量；$D$ 是训练 token 数。
- **直觉/作用**：用于估算训练成本。

## 批次大小

### 最优批次大小

$$B_{opt} \approx \frac{C}{6 \cdot N \cdot \eta_{max}}$$

实际中常使用梯度累积实现大批次：

```
effective_batch_size = batch_size × gradient_accumulation_steps × num_gpus
```

### 动态批次

训练过程中逐渐增大批次：

$$B_t = B_{min} \cdot \frac{t}{T} + B_{max} \cdot (1 - \frac{t}{T})$$

## 训练稳定性

### 损失尖峰

训练中可能出现突然的损失上升：

**原因**：
- 数据中的异常样本
- 梯度爆炸
- 学习率过高

**解决**：
- 跳过异常批次
- 降低学习率
- 增加梯度裁剪

### LayerNorm 位置

Pre-LN vs Post-LN：

```
Post-LN:  x → Attention → Add → LN → FFN → Add → LN
Pre-LN:   x → LN → Attention → Add → LN → FFN → Add
```

Pre-LN 训练更稳定，现代模型普遍采用。

## 参考文献

1. Kaplan et al. (2020). *Scaling Laws for Neural Language Models*
2. Hoffmann et al. (2022). *Training Compute-Optimal Large Language Models*
3. Brown et al. (2020). *Language Models are Few-Shot Learners*
4. Radford et al. (2019). *Language Models are Unsupervised Multitask Learners*
5. Touvron et al. (2023). *LLaMA: Open and Efficient Foundation Language Models*
