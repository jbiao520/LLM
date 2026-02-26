# 推理加速深入版

> 面向有机器学习基础读者的技术详解

## 概述

LLM 推理的核心挑战：

1. **内存带宽瓶颈**：模型参数和 KV Cache 的传输比计算更慢
2. **自回归生成**：每个 token 都需要完整前向传播
3. **动态长度**：不同请求的生成长度不同，难以批处理

## KV Cache 详解

### 原理

在自回归生成中，计算第 $t$ 个 token 的注意力：

$$\text{Attention}(Q_t, K_{1:t}, V_{1:t})$$

KV Cache 存储 $K_{1:t-1}$ 和 $V_{1:t-1}$，避免重复计算。

### 内存需求

对于 $L$ 层、$h$ 头、$d_{head}$ 维度、序列长度 $s$：

$$\text{KV Cache} = 2 \times L \times h \times d_{head} \times s \times \text{bytes}$$

**示例**（LLaMA-2 70B, FP16）：
- $L = 80$, $h = 64$, $d_{head} = 128$
- 序列长度 2048：~40GB KV Cache
- 序列长度 4096：~80GB KV Cache

### PagedAttention

vLLM 的核心创新，将 KV Cache 分页管理：

```
传统方式:
请求1: [连续大块内存]
请求2: [连续大块内存] → 预留最大长度，浪费严重

PagedAttention:
请求1: [页1][页3][页5]... 按需分配
请求2: [页2][页4]...
```

**优势**：
- 内存利用率从 ~50% 提升到 ~95%
- 支持更长上下文
- 减少内存碎片

## 连续批处理（Continuous Batching）

### 静态批处理问题

```python
# 传统批处理
batch = [req1, req2, req3]  # 长度: 50, 100, 20 tokens

# 等待最长请求完成
# req3 生成 20 tokens 后就空闲了，但要等 req2
```

### 连续批处理

```python
# 迭代级调度
iteration_1: [req1, req2, req3]  # 都在生成
iteration_2: [req1, req2, req3]  # ...
iteration_3: [req1, req2]        # req3 完成，退出
iteration_4: [req1, req4]        # 新请求 req4 加入
```

**吞吐量提升**：2-4x

## Flash Attention

### 标准注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

问题：
- $O(N^2)$ 内存（$N \times N$ 的注意力矩阵）
- 多次 HBM 访问

### Flash Attention 优化

**核心思想**：分块计算，避免存储完整注意力矩阵

```python
# 伪代码
for block_q in Q:
    for block_k, block_v in K, V:
        # 在 SRAM 中计算
        local_attention = block_q @ block_k.T
        local_output = local_attention @ block_v
        # 累加到全局输出
```

**性能**：
- 内存：$O(N)$ 而非 $O(N^2)$
- 速度：2-4x 加速（减少 HBM 访问）

## 主流框架对比

### vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
params = SamplingParams(max_tokens=100, temperature=0.7)

outputs = llm.generate(["Hello", "Hi there"], params)
```

**特点**：
- PagedAttention
- 连续批处理
- 优化的 CUDA 内核

### TensorRT-LLM

```python
import tensorrt_llm

# 构建引擎
builder = tensorrt_llm.Builder()
engine = builder.create_engine(model_config)

# 推理
outputs = engine.generate(inputs)
```

**特点**：
- NVIDIA 官方
- 极致优化
- 支持多 GPU

### 性能对比

| 框架 | 吞吐量 | 延迟 | 易用性 |
|------|--------|------|--------|
| HuggingFace | 1x | 基准 | ⭐⭐⭐⭐⭐ |
| vLLM | 10-20x | 0.7x | ⭐⭐⭐⭐ |
| TensorRT-LLM | 15-25x | 0.5x | ⭐⭐⭐ |

## 推理优化技术总结

### 计算优化

| 技术 | 加速比 | 复杂度 |
|------|--------|--------|
| Flash Attention | 2-4x | 低 |
| 算子融合 | 1.5-2x | 中 |
| INT8/FP8 推理 | 2-3x | 中 |

### 内存优化

| 技术 | 内存节省 | 复杂度 |
|------|----------|--------|
| PagedAttention | 50% | 中 |
| KV Cache 量化 | 50-75% | 中 |
| 模型卸载 | 极大 | 低 |

### 调度优化

| 技术 | 吞吐提升 | 复杂度 |
|------|----------|--------|
| 连续批处理 | 2-4x | 中 |
| 投机解码 | 1.5-2x | 高 |
| 分离式架构 | 2-3x | 高 |

## 最佳实践

### 选择框架

```python
# 场景选择
if production_server:
    use vLLM  # 最佳性价比
elif nvidia_gpu and max_performance:
    use TensorRT-LLM  # 极致性能
elif cpu_only:
    use llama.cpp  # CPU 优化
```

### 配置建议

```python
# vLLM 推荐配置
llm = LLM(
    model="model-name",
    tensor_parallel_size=2,  # 2 GPU 并行
    gpu_memory_utilization=0.9,  # 90% 显存利用率
    max_model_len=4096,  # 最大上下文长度
    enforce_eager=True,  # 禁用 CUDA 图（调试用）
)
```

## 参考文献

1. Kwon et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*
2. Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention*
3. NVIDIA (2023). *TensorRT-LLM: High-Performance Inference*
