# 全参数微调深入版

> 面向有机器学习基础读者的技术详解

## 概述

全参数微调（Full Fine-tuning）是指在预训练模型 $W_{pre}$ 的基础上，使用目标任务数据 $\mathcal{D}_{task}$，通过梯度下降更新所有参数：

$$W_{ft} = W_{pre} - \eta \sum_{t=1}^{T} \nabla_{W} \mathcal{L}(f_{W_{pre}}(x_t), y_t)$$

其中 $\eta$ 是学习率，$\mathcal{L}$ 是损失函数，$T$ 是训练步数。

## 训练动力学

### 学习率调度

全参数微调的学习率调度至关重要：

```python
# 常见的学习率调度策略
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,      # 预热步数
    num_training_steps=10000   # 总训练步数
)
```

**关键点**：
- **预热（Warmup）**：从小学习率逐渐增大，避免初期震荡
- **衰减（Decay）**：后期减小学习率，稳定收敛

### 学习率选择

| 阶段 | 学习率范围 | 说明 |
|------|-----------|------|
| 预训练 | 1e-4 ~ 6e-4 | 较高学习率 |
| 全参数微调 | 1e-5 ~ 5e-5 | 通常比预训练低 10x |
| PEFT 微调 | 1e-4 ~ 1e-3 | 可以更高（参数少）|

## 优化技术

### 梯度累积

显存不足时，用时间换空间：

```python
gradient_accumulation_steps = 4  # 累积 4 步后更新

# 有效批次大小 = batch_size × accumulation_steps
effective_batch_size = 4 × 4 = 16
```

### 混合精度训练

使用 FP16/BF16 减少显存，加速计算：

```python
from transformers import TrainingArguments

args = TrainingArguments(
    fp16=True,              # 使用 FP16
    fp16_opt_level="O1",    # 优化级别
    # 或使用 BF16（Ampere 及以上 GPU）
    bf16=True,
    tf32=True,              # 启用 TF32
)
```

### 梯度检查点

以计算换显存：

```python
# 只保存部分中间激活，需要时重新计算
model.gradient_checkpointing_enable()
```

**显存节省**：约 30-50%
**代价**：训练速度下降约 20%

### DeepSpeed / FSDP

分布式训练策略：

| 技术 | 特点 | 适用场景 |
|------|------|----------|
| DDP | 数据并行，每卡复制完整模型 | 小模型 |
| FSDP | 分片并行，模型切分到多卡 | 大模型 |
| DeepSpeed ZeRO | 优化器状态分片 | 超大模型 |

## 灾难性遗忘

### 问题定义

微调后模型在原任务上性能下降：

$$\text{Forgetting} = \text{Perf}_{pre}(T_{orig}) - \text{Perf}_{ft}(T_{orig})$$

### 缓解策略

1. **学习率调低**
   ```python
   learning_rate = 2e-5  # 比预训练低 10-20 倍
   ```

2. **早停（Early Stopping）**
   ```python
   # 在验证集上监控，防止过拟合
   early_stopping_patience = 3
   ```

3. **混合数据训练**
   ```python
   # 新任务数据 + 原始预训练数据混合
   dataset = ConcatDataset([task_dataset, pretrain_subset])
   ```

4. **弹性权重固化（EWC）**
   ```python
   # 对重要参数施加约束
   loss = task_loss + λ * EWC_penalty
   ```

## 训练配置最佳实践

### LLaMA-2 7B 全参数微调

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./full-ft-output",

    # 批次设置
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # 有效批次 = 16

    # 学习率
    learning_rate=2e-5,
    weight_decay=0.01,

    # 调度
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    # 精度
    bf16=True,
    tf32=True,

    # 序列长度
    max_length=2048,

    # 梯度检查点
    gradient_checkpointing=True,

    # 训练轮数
    num_train_epochs=3,

    # 保存策略
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    # 日志
    logging_steps=10,
    eval_steps=500,
)
```

### 显存估算

全参数微调显存需求（优化器状态 + 梯度 + 激活）：

$$\text{Memory} \approx 4 \times \text{Params} \times \text{Bytes}$$

| 模型 | 参数量 | FP16 模型 | 全参数微调显存 |
|------|--------|----------|---------------|
| 7B | 7×10⁹ | 14GB | ~60GB |
| 13B | 13×10⁹ | 26GB | ~100GB |
| 70B | 70×10⁹ | 140GB | ~500GB |

## 与 PEFT 的对比实验

### 实验设置
- 模型：LLaMA-2 7B
- 任务：指令微调（52K 样本）
- 评估：MT-Bench

### 结果

| 方法 | 可训练参数 | 显存 | MT-Bench |
|------|-----------|------|----------|
| 全参数微调 | 7B (100%) | 60GB | 6.8 |
| LoRA (r=64) | 70M (1%) | 24GB | 6.5 |
| LoRA (r=16) | 18M (0.3%) | 16GB | 6.3 |

**结论**：全参数微调效果最好，但 LoRA 以 1% 的参数达到 95% 的效果。

## 实践建议

1. **先尝试 PEFT**：大多数场景 LoRA 足够
2. **全参数微调条件**：
   - 任务与预训练差异大
   - 有充足标注数据（>10K）
   - 追求极致效果
3. **监控遗忘**：在原始任务上定期评估
4. **备份原始模型**：全参数微调不可逆

## 参考文献

1. Howard & Ruder (2018). *Universal Language Model Fine-tuning for Text Classification*
2. Kirkpatrick et al. (2017). *Overcoming catastrophic forgetting in neural networks*
3. Ouyang et al. (2022). *Training language models to follow instructions with human feedback*
