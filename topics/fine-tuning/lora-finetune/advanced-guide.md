# LoRA 微调深入版

> 面向有机器学习基础读者的技术详解

## 概述

LoRA (Low-Rank Adaptation) 是一种参数高效微调方法，通过低秩分解来近似权重更新。对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 冻结 $W_0$ 并使用两个低秩矩阵 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$ 来表示更新：

$$W = W_0 + \Delta W = W_0 + BA$$

<a id="formula-lora-finetune-1"></a>
[📖 查看公式附录详解](#formula-lora-finetune-1-detail)

**公式解释**
- **公式含义**：冻结原权重 $W_0$，新增两个低秩矩阵 $B$ 和 $A$ 的乘积来表示权重更新。
- **变量说明**：$W_0$ 为预训练权重（冻结）；$B \in \mathbb{R}^{d \times r}$ 为降维矩阵；$A \in \mathbb{R}^{r \times k}$ 为升维矩阵；$r$ 为秩。
- **直觉/作用**：用极少的可训练参数（$r \times (d+k)$）近似全参数更新（$d \times k$），大幅降低显存需求。

其中秩 $r \ll \min(d, k)$，通常 $r \in \{1, 2, 4, 8, 16, 64\}$。

## 动机：内在维度假设

预训练模型具有低内在维度（low intrinsic dimension）。研究表明，模型适应新任务时，实际需要的自由度远小于参数总量。

Aghajanyan et al. (2020) 发现，只需在子空间中优化少量参数就能达到良好效果。LoRA 正是利用这一特性。

## 数学推导

### 权重更新的低秩表示

假设全参数微调的权重更新为 $\Delta W$。LoRA 将其分解为：

$$\Delta W = BA$$

<a id="formula-lora-finetune-2"></a>
[📖 查看公式附录详解](#formula-lora-finetune-2-detail)

其中：
- $B \in \mathbb{R}^{d \times r}$：降维矩阵，初始化为随机高斯
- $A \in \mathbb{R}^{r \times k}$：升维矩阵，初始化为零

初始化 $A = 0$ 确保 $BA = 0$，即训练开始时模型行为与预训练模型一致。

### 前向传播

$$h = W_0 x + \frac{\alpha}{r} BAx$$

<a id="formula-lora-finetune-3"></a>
[📖 查看公式附录详解](#formula-lora-finetune-3-detail)

**公式解释**
- **公式含义**：输出 = 冻结权重的输出 + LoRA 更新的输出（带缩放因子）。
- **变量说明**：$W_0 x$ 为原模型输出；$BAx$ 为 LoRA 增量；$\alpha$ 为缩放因子；$r$ 为秩。
- **直觉/作用**：$\alpha/r$ 控制 LoRA 更新的影响强度；除以 $r$ 让调 $\alpha$ 时无需因秩不同而重新调参。

其中 $\alpha$ 是缩放因子，用于控制 LoRA 的影响强度。除以 $r$ 使得调整 $\alpha$ 时无需重新调参。

### 参数量对比

对于 $W_0 \in \mathbb{R}^{d \times k}$：

| 方法 | 可训练参数 |
|------|-----------|
| 全参数微调 | $d \times k$ |
| LoRA | $r \times (d + k)$ |

压缩比：$\frac{r(d+k)}{dk} = \frac{r}{d} + \frac{r}{k} \approx \frac{2r}{d}$（假设 $d \approx k$）

当 $r=8, d=4096$ 时，压缩比约为 0.4%，即减少 99.6% 的可训练参数。

## LoRA 配置详解

### Rank (r)

秩 $r$ 决定了 LoRA 的表达能力：

- **$r=1-4$**：简单任务（风格迁移、简单分类）
- **$r=8-16$**：中等复杂度任务（指令微调）
- **r=32-64**：复杂任务（新知识学习、多任务）

**建议**：从小秩开始，逐步增大直到性能饱和。

### Alpha (α)

缩放因子 $\alpha$ 控制 LoRA 更新的强度：

$$\text{effective\_lr} \propto \frac{\alpha}{r} \times \text{learning\_rate}$$

<a id="formula-lora-finetune-4"></a>
[📖 查看公式附录详解](#formula-lora-finetune-4-detail)

常见设置：
- $\alpha = 2r$（保守）
- $\alpha = r$（标准）
- $\alpha = 16$（固定，常用于 r=8 时）

### Target Modules

选择哪些层应用 LoRA：

```python
# 常见配置
target_modules = ["q_proj", "v_proj"]  # 只微调注意力
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 全部注意力
target_modules = ["all_linear"]  # 所有线性层
```

## QLoRA：量化 + LoRA

QLoRA (Dettmers et al., 2023) 结合了量化和 LoRA：

### 核心技术

1. **4-bit NormalFloat (NF4)**
   - 信息论最优的量化数据类型
   - 对于正态分布权重，NF4 比 4-bit 整数更精确

2. **双重量化 (Double Quantization)**
   - 对量化常数再次量化
   - 进一步减少内存（约 0.5GB/65B模型）

3. **分页优化器 (Paged Optimizers)**
   - 使用 NVIDIA Unified Memory
   - 处理内存峰值，避免 OOM

### QLoRA vs LoRA

| 特性 | LoRA | QLoRA |
|------|------|-------|
| 基础模型精度 | FP16/BF16 | 4-bit NF4 |
| 65B模型显存 | ~130GB | ~48GB |
| 训练速度 | 快 | 略慢（需反量化）|
| 精度损失 | 无 | 极小 |

## 训练技巧

### 学习率选择

```python
# LoRA 学习率通常比全参数微调高
learning_rate = 1e-4  # 起始点
learning_rate = 5e-4  # 可尝试更高
```

### Dropout

```python
# LoRA 层可以添加 dropout
lora_dropout = 0.05  # 常用值
```

### 合并策略

训练完成后，可以将 LoRA 权重合并到基础模型：

$$W_{merged} = W_0 + \frac{\alpha}{r} BA$$

<a id="formula-lora-finetune-5"></a>
[📖 查看公式附录详解](#formula-lora-finetune-5-detail)

**公式解释**
- **公式含义**：将 LoRA 的低秩更新合并到原权重，得到新的完整权重矩阵。
- **变量说明**：$W_0$ 为预训练权重；$B, A$ 为训练后的 LoRA 矩阵；$\alpha/r$ 为缩放因子。
- **直觉/作用**：合并后模型与原模型结构相同，部署时无需 LoRA 代码，无额外推理开销。

```python
# 合并后的模型可独立使用，无需 LoRA 代码
model = model.merge_and_unload()
```

## 实验结果

### GLUE 基准测试

| 方法 | 可训练参数 | RTE | MRPC | CoLA |
|------|-----------|-----|------|------|
| Fine-tuning | 100% | 70.2 | 87.6 | 63.6 |
| LoRA (r=8) | 0.1% | 69.8 | 87.4 | 63.2 |
| QLoRA | 0.1% | 69.5 | 87.1 | 62.8 |

### LLM 指令微调

在 LLaMA 7B 上的指令微调结果：

| 方法 | MMLU | 训练显存 |
|------|------|----------|
| 全参数微调 | 42.1 | 60GB |
| LoRA | 41.8 | 24GB |
| QLoRA | 41.5 | 12GB |

## 局限性

1. **知识更新有限**：低秩约束限制了学习新知识的能力
2. **任务冲突**：多个 LoRA 模块可能相互干扰
3. **层选择敏感**：需要选择合适的 target modules

## 最佳实践

```python
# 推荐配置（LLaMA 类模型）
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

## 参考文献

1. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*
2. Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*
3. Aghajanyan et al. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*

