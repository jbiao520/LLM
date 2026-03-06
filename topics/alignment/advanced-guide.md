# 对齐（Alignment）深入版

> 面向有机器学习基础读者的技术详解

## 概述

对齐技术让 LLM 从单纯的文本生成器变成有帮助、诚实、无害的助手。本文深入分析 SFT、RLHF 和 DPO 的数学原理。

## SFT（监督微调）

### 目标函数

给定指令-回答对 $(x, y)$，最大化条件概率：

$$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \log \pi_\theta(y | x) \right]$$

<a id="formula-alignment-1"></a>

**公式解释**
- **公式含义**：标准的自回归语言建模损失。
- **变量说明**：$x$ 是指令；$y$ 是期望的回答；$\pi_\theta$ 是策略（语言模型）。
- **直觉/作用**：让模型学会在给定指令下生成期望的回答。

### 数据格式

```json
{
  "instruction": "解释什么是机器学习",
  "input": "",
  "output": "机器学习是人工智能的一个分支..."
}
```

### 训练细节

- 学习率：通常比预训练小（1e-5 到 5e-5）
- Epoch：3-5 个
- 数据量：10K - 100K 高质量样本

## 奖励模型

### Bradley-Terry 模型

给定两个回答 $y_w$（更好）和 $y_l$（更差），$y_w$ 被偏好的概率：

$$P(y_w \succ y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))$$

<a id="formula-alignment-2"></a>

**公式解释**
- **公式含义**：奖励差越大，好回答被选中的概率越高。
- **变量说明**：$r(x, y)$ 是奖励函数；$\sigma$ 是 sigmoid 函数。
- **直觉/作用**：将奖励差异映射为选择概率。

### 损失函数

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

<a id="formula-alignment-3"></a>

**公式解释**
- **公式含义**：最大化好回答得分高于坏回答得分的对数概率。
- **变量说明**：$y_w$ 是更好的回答；$y_l$ 是更差的回答。
- **直觉/作用**：让奖励模型学会区分回答质量。

### 实现架构

```
输入 (x, y) → LM (冻结) → 最后一层隐藏状态 → 线性层 → 标量奖励 r(x, y)
```

## RLHF（PPO）

### 目标函数

$$\mathcal{L}_{PPO} = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right]$$

<a id="formula-alignment-4"></a>

**公式解释**
- **公式含义**：最大化奖励，同时控制与参考模型的偏离。
- **变量说明**：$\pi_{ref}$ 是参考模型（SFT 后）；$\beta$ 是 KL 惩罚系数。
- **直觉/作用**：在获得高奖励和保持语言能力之间平衡。

### PPO 裁剪

$$\mathcal{L}_{clip} = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

<a id="formula-alignment-5"></a>

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ 是重要性采样比率。

**公式解释**
- **公式含义**：限制策略更新幅度，避免不稳定。
- **变量说明**：$\hat{A}_t$ 是优势函数；$\epsilon$ 是裁剪参数（通常 0.2）。
- **直觉/作用**：防止策略在一次更新中变化太大。

### 完整 PPO 损失

$$\mathcal{L} = \mathcal{L}_{clip} - c_1 \mathcal{L}_{value} + c_2 \mathcal{L}_{entropy}$$

<a id="formula-alignment-6"></a>

**公式解释**
- **公式含义**：组合裁剪损失、值函数损失和熵奖励。
- **变量说明**：$c_1$, $c_2$ 是权重系数。
- **直觉/作用**：平衡策略优化、值估计和探索。

## DPO（直接偏好优化）

### 核心洞察

可以从偏好数据中直接推导出最优策略，无需显式的奖励模型。

### 理论推导

最优策略与奖励函数的关系：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

<a id="formula-alignment-7"></a>

**公式解释**
- **公式含义**：奖励可以用策略比的对数表示（加上配分函数）。
- **变量说明**：$Z(x)$ 是配分函数（在 DPO 中会被消去）。
- **直觉/作用**：奖励模型和策略可以互相转换。

### DPO 损失函数

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

<a id="formula-alignment-8"></a>

简化为：

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right) \right]$$

<a id="formula-alignment-9"></a>

**公式解释**
- **公式含义**：直接优化策略，使好回答的概率相对提高。
- **变量说明**：$\beta$ 控制偏离程度；$y_w$ 是更好的回答；$y_l$ 是更差的回答。
- **直觉/作用**：无需奖励模型，直接用偏好数据优化。

### DPO 优势

1. **简单**：只需一个策略模型，无需奖励模型
2. **稳定**：没有 RL 的不稳定性
3. **高效**：计算成本更低
4. **有效**：效果与 RLHF 相当或更好

## 其他对齐方法

### IPO（Identity Preference Optimization）

使用平方损失替代 sigmoid：

$$\mathcal{L}_{IPO} = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \frac{1}{\beta} \right)^2 \right]$$

<a id="formula-alignment-10"></a>

### KTO（Kahneman-Tversky Optimization）

基于前景理论，不需要成对偏好数据：

$$\mathcal{L}_{KTO} = \mathbb{E}_{(x, y)} \left[ \lambda_y (1 - v(x, y, \beta)) \right]$$

<a id="formula-alignment-11"></a>

其中 $v$ 是价值函数，$\lambda_y$ 根据回答好坏取不同值。

### ORPO（Odds Ratio Preference Optimization）

在 SFT 中直接加入偏好优化：

$$\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} + \lambda \mathcal{L}_{OR}$$

<a id="formula-alignment-12"></a>

## 参考文献

1. Ouyang et al. (2022). *Training language models to follow instructions with human feedback*
2. Schulman et al. (2017). *Proximal Policy Optimization Algorithms*
3. Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
4. Bai et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*
5. Azar et al. (2023). *A General Theoretical Paradigm for RLHF*
