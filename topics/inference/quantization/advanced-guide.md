# 量化深入版

> 面向有机器学习基础读者的技术详解

## 概述

量化（Quantization）是将高精度浮点数映射到低精度表示的技术。对于权重矩阵 $W \in \mathbb{R}^{n \times m}$，量化操作可表示为：

$$\hat{W} = Q(W) = s \cdot \text{clamp}\left(\text{round}\left(\frac{W}{s}\right), -2^{b-1}, 2^{b-1}-1\right)$$

**公式解释**
- **公式含义**：将浮点数除以缩放因子、四舍五入、截断到目标范围，再乘回复原。
- **变量说明**：$s$ 为缩放因子（scale）；$b$ 为目标位宽；$\text{round}$ 为舍入；$\text{clamp}$ 为截断到 $[-2^{b-1}, 2^{b-1}-1]$。
- **直觉/作用**：用有限的整数近似连续浮点数，牺牲少量精度换取大幅压缩。

其中：
- $s$ 是量化缩放因子（scale）
- $b$ 是目标位宽（如 4, 8）
- $\text{round}$ 是舍入函数

## 量化基础

### 量化类型

#### 1. 对称量化

$$s = \frac{\max(|W|)}{2^{b-1}}$$

$$Q(x) = s \cdot \text{round}\left(\frac{x}{s}\right)$$

**公式解释**
- **公式含义**：缩放因子由最大绝对值决定，量��范围关于 0 对称。
- **变量说明**：$s$ 为缩放因子；$b$ 为位宽；$2^{b-1}$ 为量化级别数的一半。
- **直觉/作用**：零点保持不变，实现简单，但非均匀分布时会浪费部分量化范围。

**优点**：简单，零点不变
**缺点**：非均匀分布时浪费动态范围

#### 2. 非对称量化

$$s = \frac{\max(W) - \min(W)}{2^b - 1}$$

$$z = \text{round}\left(-\frac{\min(W)}{s}\right)$$

$$Q(x) = s \cdot (\text{round}(x/s) + z)$$

**公式解释**
- **公式含义**：缩放因子由值域范围决定，引入零点偏移 $z$ 适配非对称分布。
- **变量说明**：$s$ 为缩放因子；$z$ 为零点（zero point）；$2^b - 1$ 为量化级别数。
- **直觉/作用**：充分利用整个量化范围，精度更高，但需要额外存储零点。

**优点**：更充分利用动态范围
**缺点**：需要额外存储零点

### 量化粒度

| 粒度 | 说明 | 参数量 |
|------|------|--------|
| Per-tensor | 整个张量一个 scale | 最少 |
| Per-channel | 每个通道一个 scale | 中等 |
| Per-group | 每组元素一个 scale | 较多 |

## GPTQ

### 原理

GPTQ (Frantar et al., 2023) 基于 Optimal Brain Quantization，逐层量化权重：

**目标**：找到量化权重 $\hat{W}$，使得输出误差最小：

$$\arg\min_{\hat{W}} \|WX - \hat{W}X\|^2$$

**公式解释**
- **公式含义**：寻找量化后的���重 $\hat{W}$，使其输出与原始权重 $W$ 的输出尽可能接近。
- **变量说明**：$W$ 为原始权重；$\hat{W}$ 为量化后权重；$X$ 为输入激活；$\|\cdot\|^2$ 为 L2 范数平方。
- **直觉/作用**：直接优化任务相关误差（输出差异），而非单纯的权重误差。

### 算法流程

对于权重矩阵的每一列 $w_i$：

1. **量化**：$\hat{w}_i = Q(w_i)$
2. **计算误差**：$\delta_i = w_i - \hat{w}_i$
3. **更新未量化的权重**：
   $$W_{[:, i+1:]} \leftarrow W_{[:, i+1:]} - \frac{\delta_i \cdot (X_i^T X_{[i+1:]})}{X_i^T X_i + \epsilon}$$

**公式解释**
- **公式含义**：将当前列的量化误差通过 Hessian 逆矩阵传播到后续未量化的列。
- **变量说明**：$\delta_i$ 为第 $i$ 列的量化误差；$X_i^T X_{[i+1:]}$ 为激活协方差；$\epsilon$ 为数值稳定性常数。
- **直觉/作用**：用后续列的调整来补偿当前列量化带来的输出误差。

### 实现

```python
from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,              # 目标位宽
    group_size=128,      # 分组大小
    desc_act=True,       # 激活感知
)

# 加载并量化模型
model = AutoGPTQForCausalLM.from_quantized(
    model_name,
    quantize_config=quantize_config
)
```

### 特点

- **离线量化**：量化后无需校准
- **高压缩率**：4-bit 可达原始大小的 1/4
- **精度保持**：在大多数任务上损失 < 1%

## AWQ

### 原理

AWQ (Lin et al., 2023) 发现：**只有 1% 的权重对模型输出影响最大**。

**核心思想**：
1. 分析哪些权重"更重要"（基于激活值大小）
2. 对重要权重保持更高精度
3. 对其他权重激进量化

### 算法

$$\hat{W} = Q(W \cdot s) \cdot s^{-1}$$

**公式解释**
- **公式含义**：先对权重做缩放，量化后再逆向缩放，使重要权重获得更高精度。
- **变量说明**：$W$ 为原始权重；$s$ 为每通道缩放因子；$Q(\cdot)$ 为量化函数；$s^{-1}$ 为逐元素除法。
- **直觉/作用**：缩放后重要权重的值域扩大，��化时被分配更多量化级别。

其中 $s$ 是每通道的缩放因子，基于激活值分布计算：

$$s_i = \max(|X_i|)^\alpha, \quad \alpha \in [0, 1]$$

**公式解释**
- **公式含义**：缩放因子由激活值的幅值决定，激活越大的通道权重越重要。
- **变量说明**：$X_i$ 为第 $i$ 通道的激活值；$\alpha$ 为调节幂次（通常接近 1）。
- **直觉/作用**：激活大的通道对输出影响更大，应保留更高精度的权重。

### 实现

```python
from awq import AutoAWQForCausalLM

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_name)

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4
}

# 量化（需要校准数据）
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_data
)
```

## GGUF

### 特点

GGUF 是 llama.cpp 使用的量化格式：

| 量化类型 | 说明 | 压缩比 |
|----------|------|--------|
| Q4_0 | 4-bit，无额外缩放 | ~4x |
| Q4_K_M | 4-bit，K-量化，中等 | ~4x |
| Q5_K_M | 5-bit，K-量化，中等 | ~3x |
| Q8_0 | 8-bit，无额外缩放 | ~2x |

### 使用

```bash
# 下载 GGUF 模型
wget https://huggingface.co/.../model-Q4_K_M.gguf

# 使用 llama.cpp 运行
./main -m model-Q4_K_M.gguf -p "你好"
```

## 精度对比实验

### LLaMA-2 70B 量化结果

| 方法 | 位宽 | 模型大小 | MMLU | Perplexity |
|------|------|----------|------|------------|
| FP16 | 16 | 140 GB | 69.8 | 3.32 |
| GPTQ | 4 | 35 GB | 68.9 | 3.45 |
| AWQ | 4 | 35 GB | 69.2 | 3.41 |
| GGUF Q4_K_M | 4 | 40 GB | 68.5 | 3.52 |

### 消费级显卡可用性

| 模型 | FP16 | INT8 | INT4 |
|------|------|------|------|
| 7B | 14 GB | 7 GB | 4 GB |
| 13B | 26 GB | 13 GB | 7 GB |
| 70B | 140 GB | 70 GB | 35 GB |

**结论**：70B 模型 INT4 量化后可在单张 A100 40GB 上运行。

## 混合精度量化

不同层使用不同精度：

```python
# 关键层（如 Attention）使用更高精度
# 其他层使用较低精度

quant_config = {
    "q_proj": {"bits": 8},    # 注意力使用 8-bit
    "k_proj": {"bits": 8},
    "v_proj": {"bits": 8},
    "o_proj": {"bits": 8},
    "gate_proj": {"bits": 4},  # FFN 使用 4-bit
    "up_proj": {"bits": 4},
    "down_proj": {"bits": 4},
}
```

## 最佳实践

### 选择建议

| 场景 | 推荐方法 |
|------|----------|
| GPU 推理（追求速度） | AWQ |
| GPU 推理（追求精度） | GPTQ |
| CPU 推理 | GGUF |
| 边缘设备 | GGUF Q4_K_M |

### 量化流程

1. **选择精度**：INT8（无损）或 INT4（高压缩）
2. **选择方法**：AWQ（快）或 GPTQ（准）
3. **准备校准数据**：~512 样本
4. **量化并测试**：验证关键任务性能

## 参考文献

1. Frantar et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*
2. Lin et al. (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*
3. Xiao et al. (2023). *SmoothQuant: Accurate and Efficient Post-Training Quantization*
