# 文本生成（Generation）深入版

> 面向有机器学习基础读者的技术详解

## 概述

文本生成是 LLM 的核心能力。本文深入分析各种采样策略的数学原理，包括 Temperature Scaling、Top-K、Top-P (Nucleus) Sampling、Beam Search 等。

## 自回归生成

### 基本形式

给定输入序列 $x = (x_1, ..., x_n)$，自回归生成：

$$P(y | x) = \prod_{t=1}^{T} P(y_t | x, y_{<t})$$

<a id="formula-generation-1"></a>

**公式解释**
- **公式含义**：输出序列的概率是每个位置条件概率的乘积。
- **变量说明**：$y_t$ 是第 $t$ 步生成的 token；$y_{<t}$ 是之前生成的所有 token。
- **直觉/作用**：每一步的生成依赖于之前的生成结果。

### 解码目标

$$y^* = \arg\max_{y} P(y | x)$$

<a id="formula-generation-2"></a>

**公式解释**
- **公式含义**：找到概率最大的输出序列。
- **变量说明**：$y^*$ 是最优序列。
- **直觉/作用**：理论上最优的解码策略，但实际难以精确计算。

## Softmax 与 Logits

### Logits 到概率

模型输出 logits $z \in \mathbb{R}^V$（$V$ 是词汇表大小），通过 softmax 转为概率：

$$P(y_t = v | context) = \frac{e^{z_v}}{\sum_{j=1}^{V} e^{z_j}}$$

<a id="formula-generation-3"></a>

**公式解释**
- **公式含义**：将 logits 转换为归一化的概率分布。
- **变量说明**：$z_v$ 是词 $v$ 的 logit 值；$V$ 是词汇表大小。
- **直觉/作用**：概率越高，该词被选中的可能性越大。

### 数值稳定性

使用 log-softmax 避免数值下溢：

$$\log P(y_t = v) = z_v - \log \sum_{j=1}^{V} e^{z_j}$$

<a id="formula-generation-4"></a>

**公式解释**
- **公式含义**：在对数空间计算，避免极小概率的数值问题。
- **变量说明**：$\log \sum e^{z_j}$ 是 log-sum-exp。
- **直觉/作用**：计算机友好的概率计算方式。

## Temperature Scaling

### 定义

$$P_T(y_t = v) = \frac{e^{z_v / T}}{\sum_{j=1}^{V} e^{z_j / T}}$$

<a id="formula-generation-5"></a>

**公式解释**
- **公式含义**：用温度 $T$ 调整概率分布的"尖锐度"。
- **变量说明**：$T$ 是温度参数；$T > 0$。
- **直觉/作用**：$T \to 0$ 趋近贪婪；$T \to \infty$ 趋近均匀分布。

### 温度对熵的影响

分布的熵：

$$H(P_T) = -\sum_v P_T(v) \log P_T(v)$$

<a id="formula-generation-6"></a>

- $T \to 0$：$H \to 0$（确定性）
- $T = 1$：原始分布的熵
- $T \to \infty$：$H \to \log V$（均匀分布）

**公式解释**
- **公式含义**：熵衡量分布的不确定性。
- **变量说明**：$H$ 是熵；$V$ 是词汇表大小。
- **直觉/作用**：温度越高，熵越大，输出越不确定。

### 实现细节

```python
def temperature_sampling(logits, temperature):
    if temperature == 0:
        return torch.argmax(logits)  # 贪婪

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

## Top-K Sampling

### 算法

1. 保留概率最高的 K 个 token
2. 将其余 token 的概率设为 0
3. 重新归一化
4. 从新分布中采样

### 数学表达

$$P_{\text{top-k}}(v) = \begin{cases}
\frac{P(v)}{\sum_{v' \in V_k} P(v')} & \text{if } v \in V_k \\
0 & \text{otherwise}
\end{cases}$$

<a id="formula-generation-7"></a>

其中 $V_k$ 是概率最高的 K 个 token 的集合。

**公式解释**
- **公式含义**：只在 top-K 候选中重新分配概率。
- **变量说明**：$V_k$ 是概率最高的 K 个词的集合。
- **直觉/作用**：避免采样到极不可能的词。

### 复杂度

- 时间：$O(V \log K)$（使用堆）
- 空间：$O(K)$

## Top-P (Nucleus) Sampling

### 动机

Top-K 的问题：
- 概率分布平坦时，K 可能太小
- 概率分布尖锐时，K 可能太大

Top-P 自适应地选择候选集大小。

### 算法

1. 按概率降序排列 token
2. 选择最小的集合 $V_p$，使得 $\sum_{v \in V_p} P(v) \geq p$
3. 重新归一化
4. 采样

### 数学表达

$$V_p = \min\{V' \subseteq V : \sum_{v \in V'} P(v) \geq p\}$$

<a id="formula-generation-8"></a>

$$P_{\text{top-p}}(v) = \begin{cases}
\frac{P(v)}{\sum_{v' \in V_p} P(v')} & \text{if } v \in V_p \\
0 & \text{otherwise}
\end{cases}$$

<a id="formula-generation-9"></a>

**公式解释**
- **公式含义**：选择累计概率达到 $p$ 的最小词集合。
- **变量说明**：$p$ 是阈值（如 0.9）；$V_p$ 是选中的词集合。
- **直觉/作用**：自适应地确定候选集大小。

### 实现

```python
def top_p_sampling(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # 计算累计概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 找到需要移除的位置
    sorted_indices_to_remove = cumulative_probs > p

    # 保留第一个超过阈值的
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # 移除低概率 token
    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # 采样
    probs = F.softmax(sorted_logits, dim=-1)
    return sorted_indices[torch.multinomial(probs, 1)]
```

## Typical Sampling

### 动机

人类文本的"典型性"：倾向于选择信息量接近期望信息量的 token。

### 算法

基于信息论，选择"典型"的 token：

$$H = -\sum_v P(v) \log P(v)$$

<a id="formula-generation-10"></a>

选择满足 $|-\log P(v) - H| \leq \epsilon$ 的 token。

**公式解释**
- **公式含义**：选择信息量接近期望信息量的词。
- **变量说明**：$H$ 是熵；$\epsilon$ 是阈值。
- **直觉/作用**：避免选择过于罕见或过于常见的词。

## Beam Search

### 基本算法

维护 $k$ 个最有可能的候选序列（beam）：

```
初始: beam = [("", 0)]  # (序列, log概率)

for each step:
    candidates = []
    for seq, score in beam:
        for token, prob in top_tokens(seq):
            candidates.append((seq + token, score + log(prob)))
    beam = top_k(candidates, k)  # 保留 k 个最佳

return best_sequence(beam)
```

### 数学表达

$$\text{Beam}_t = \text{TopK}_{y_{<t}} \left( \sum_{i=1}^{t} \log P(y_i | y_{<i}, x) \right)$$

<a id="formula-generation-11"></a>

**公式解释**
- **公式含义**：每步保留累积概率最高的 $k$ 条路径。
- **变量说明**：$\text{Beam}_t$ 是第 $t$ 步的候选集；$k$ 是 beam 宽度。
- **直觉/作用**：多路径搜索，找到近似最优解。

### 变体

1. **Length-normalized Beam Search**

$$\text{score}(y) = \frac{\sum_{t} \log P(y_t | y_{<t})}{|y|^\alpha}$$

<a id="formula-generation-12"></a>

**公式解释**
- **公式含义**：用长度归一化，避免偏向短序列。
- **变量说明**：$|y|$ 是序列长度；$\alpha$ 是归一化系数（通常 0.6-1.0）。
- **直觉/作用**：公平比较不同长度的候选序列。

2. **Diverse Beam Search**

将 beam 分成 $G$ 组，组间施加多样性惩罚。

### 复杂度

- 时间：$O(k \cdot V \cdot T)$
- 空间：$O(k \cdot T)$

## Repetition Penalty

### 实现

对已出现的 token 调整其 logits：

$$z'_v = \begin{cases}
z_v / \alpha & \text{if } z_v > 0 \text{ and } v \in \text{history} \\
z_v \cdot \alpha & \text{if } z_v \leq 0 \text{ and } v \in \text{history} \\
z_v & \text{otherwise}
\end{cases}$$

<a id="formula-generation-13"></a>

**公式解释**
- **公式含义**：降低已出现词的概率。
- **变量说明**：$\alpha$ 是惩罚系数（通常 1.1-1.5）。
- **直觉/作用**：防止重复生成相同内容。

## 对比总结

| 方法 | 复杂度 | 优点 | 缺点 |
|------|--------|------|------|
| Greedy | $O(V)$ | 快速、确定 | 重复、质量差 |
| Temperature | $O(V)$ | 简单、可控 | 需调参 |
| Top-K | $O(V \log K)$ | 简单 | 固定 K 可能不优 |
| Top-P | $O(V \log V)$ | 自适应 | 稍复杂 |
| Beam Search | $O(k \cdot V \cdot T)$ | 高质量 | 慢、重复 |

## 参考文献

1. Holtzman et al. (2020). *The Curious Case of Neural Text Degeneration*
2. Fan et al. (2018). *Hierarchical Neural Story Generation*
3. Radford et al. (2019). *Language Models are Unsupervised Multitask Learners*
4. Meister et al. (2022). *Typical Decoding for Natural Language Generation*
5. Vijayakumar et al. (2018). *Diverse Beam Search*
