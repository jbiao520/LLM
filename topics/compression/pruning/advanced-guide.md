# 剪枝（Pruning）深入版

> 面向有机器学习基础读者的模型剪枝深度指南

## 1. 重要性评估方法

### 1.1 幅度剪枝（Magnitude Pruning）

最简单有效的方法：删除绝对值最小的权重。

$$\text{重要性}(w) = |w|$$

**阈值确定：**
$$\tau = \text{Percentile}(|W|, p)$$

其中 $p$ 是目标稀疏度（如 80%）。

**问题：** 小权重不一定不重要！

### 1.2 梯度敏感度

基于损失函数对权重的敏感度：

$$\text{重要性}(w) = |w \cdot \frac{\partial L}{\partial w}|$$

即：权重 × 梯度（Taylor 展开一阶近似）

**直觉：** 大权重 + 大梯度 = 对损失影响大

### 1.3 二阶敏感度（Optimal Brain Surgeon）

使用 Hessian 矩阵评估：

$$\Delta L \approx \frac{1}{2} w^T H w$$

其中 $H$ 是 Hessian 矩阵。

**计算代价高**，但理论上最优。

### 1.4 基于激活的方法

根据神经元的激活频率：

$$\text{重要性}_i = \sum_{j=1}^{N} \mathbb{1}[|a_i^{(j)}| > \epsilon]$$

其中 $a_i$ 是神经元 $i$ 的激活值。

## 2. 结构化剪枝算法

### 2.1 L1-norm 通道剪枝

对于卷积层，按通道的 L1 范数排序：

$$\text{重要性}_c = \sum_{w \in \text{channel}_c} |w|$$

删除 $\text{重要性}_c$ 最小的通道。

### 2.2 ThiNet

基于下一层激活值选择通道：

1. 对每个通道，计算其对下一层输出的贡献
2. 贡献小的通道被删除
3. 使用贪心算法选择要保留的通道

### 2.3 Network Slimming

利用 BN 层的缩放参数 $\gamma$：

$$y = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

训练时加 L1 正则：
$$L = L_{task} + \lambda \sum_{\gamma} |\gamma|$$

$\gamma$ 小的通道被剪掉。

## 3. 非结构化剪枝算法

### 3.1 GraSP (Gradient-based Pruning)

保留梯度的梯度流动：

$$\text{分数}_w = -w \cdot H \cdot g$$

目标：最大化剪枝后的梯度流。

### 3.2 SNIP (Single-shot Pruning)

训练开始前，基于连接敏感度：

$$S_w = |w \cdot \frac{\partial L}{\partial w}|$$

只计算一次，不需要训练。

### 3.3 RigL (Rigorous Lottery)

动态稀疏训练：
1. 从稀疏网络开始
2. 周期性：剪掉小权重，重新激活大梯度连接
3. 允许网络"探索"更好的结构

## 4. 剪枝调度策略

### 4.1 一次性剪枝（One-shot）

```
训练 ────▶ 剪枝 ────▶ 微调
```

简单，但大稀疏度时精度损失大。

### 4.2 迭代剪枝（Iterative）

```
训练 ─▶ 剪枝10% ─▶ 微调 ─▶ 剪枝10% ─▶ 微调 ─▶ ...
```

精度好，但耗时。

### 4.3 渐进剪枝（Gradual）

训练过程中逐步增加稀疏度：

$$s_t = s_{initial} + (s_{final} - s_{initial}) \cdot (1 - (1 - \frac{t}{T})^3)$$

三次方调度，平滑过渡。

## 5. 稀疏矩阵存储与计算

### 5.1 CSR/CSC 格式

**Compressed Sparse Row (CSR):**
```
原始: [1, 0, 0, 2, 0, 3, 0, 0, 4]
       ↓
values: [1, 2, 3, 4]
indices: [0, 3, 5, 8]
indptr: [0, 2, 3, 4]
```

存储开销：$2 \times nnz + n + 1$

### 5.2 块稀疏格式

按块存储非零区域：

```
┌────┬────┬────┬────┐
│████│    │    │████│
│████│    │    │████│
├────┼────┼────┼────┤
│    │████│    │    │
│    │████│    │    │
└────┴────┴────┴────┘
只存储有 █ 的块
```

**N:M 稀疏：** 每 M 个元素中只有 N 个非零。

### 5.3 硬件加速

| 格式 | GPU 支持 | 加速效果 |
|------|----------|----------|
| 非结构化 | 需要稀疏库 | 1-2x |
| 块稀疏 (2:4) | Ampere 原生 | 2x |
| 通道剪枝 | 标准 CUDA | 2-4x |

## 6. LLM 剪枝实践

### 6.1 LLM 剪枝挑战

| 挑战 | 原因 |
|------|------|
| 结构破坏 | LLM 对结构变化敏感 |
| 微调成本 | 重新训练需要大量计算 |
| 层间依赖 | 不能独立剪各层 |

### 6.2 LLM-Pruner

针对 LLM 的结构化剪枝：

1. **耦合剪枝：** 同时考虑多头注意力和 FFN
2. **重要性估计：** 基于一阶泰勒展开
3. **轻量微调：** 使用 LoRA 恢复精度

### 6.3 Wanda (Pruning by Weights and Activations)

无需重新训练！

$$\text{分数}_{ij} = |W_{ij}| \cdot \sqrt{\sum_x X_{j}^2}$$

按输入激活的范数调整权重重要性。

**结果：** 50% 稀疏度，无需微调，性能接近原始。

## 7. 代码示例

### 7.1 幅度剪枝

```python
import torch

def magnitude_prune(weight, sparsity):
    """幅度剪枝"""
    # 计算阈值
    threshold = torch.quantile(weight.abs().flatten(), sparsity)
    # 创建掩码
    mask = weight.abs() > threshold
    # 应用剪枝
    return weight * mask, mask

# 使用示例
weight = torch.randn(1024, 1024)
pruned_weight, mask = magnitude_prune(weight, 0.8)  # 80% 稀疏度
print(f"非零元素: {mask.sum().item()} / {mask.numel()}")
```

### 7.2 渐进剪枝调度

```python
def gradual_sparsity_schedule(t, T, s_init, s_final):
    """三次方渐进调度"""
    return s_init + (s_final - s_init) * (1 - (1 - t/T)**3)

# 训练循环
for t in range(total_steps):
    current_sparsity = gradual_sparsity_schedule(t, total_steps, 0, 0.9)
    prune_with_sparsity(current_sparsity)
```

## 8. 实践建议

### 8.1 选择剪枝方法

| 场景 | 推荐方法 |
|------|----------|
| 快速部署 | 结构化 + 微调 |
| 追求精度 | 非结构化 + 迭代剪枝 |
| LLM | Wanda 或 LLM-Pruner |

### 8.2 稀疏度选择

| 稀疏度 | 精度影响 | 加速 |
|--------|----------|------|
| 50% | 很小 | 1.5-2x |
| 70% | 可接受 | 2-3x |
| 90% | 明显下降 | 3-5x |

## 参考文献

- [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) - Han et al., 2015
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) - Frankle & Carbin, 2018
- [Network Slimming](https://arxiv.org/abs/1708.06519) - Liu et al., 2017
- [Wanda](https://arxiv.org/abs/2306.11695) - Sun et al., 2023
- [SNIP](https://arxiv.org/abs/1810.02340) - Lee et al., 2018
