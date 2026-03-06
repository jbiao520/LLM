# 分布式训练（Distributed Training）深入版

> 面向有机器学习基础读者的分布式训练深度指南

## 概述

本文档深入讲解分布式训练的数学原理、通信模式和实现细节。

## 1. 数据并行的数学原理

### 1.1 梯度同步

数据并行的核心是**梯度聚合**。每个 GPU 计算局部梯度，然后汇总。

**单卡梯度计算：**
$$g_i = \frac{1}{B_i} \sum_{j=1}^{B_i} \nabla L(f(x_j; \theta), y_j)$$

其中：
- $g_i$：第 $i$ 个 GPU 上的梯度
- $B_i$：第 $i$ 个 GPU 上的 batch size
- $\theta$：模型参数
- $L$：损失函数

**全局梯度聚合：**
$$g_{global} = \frac{1}{N} \sum_{i=1}^{N} g_i$$

其中 $N$ 是 GPU 数量。

### 1.2 AllReduce 操作

AllReduce 是分布式训练的核心通信原语。

```
┌─────────────────────────────────────────────────────────────────┐
│                     AllReduce 操作                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   初始状态：每个 GPU 有一个值                                      │
│   GPU 0: [1]    GPU 1: [2]    GPU 2: [3]    GPU 3: [4]         │
│                                                                 │
│   After AllReduce (Sum)：每个 GPU 都有总和                        │
│   GPU 0: [10]   GPU 1: [10]   GPU 2: [10]   GPU 3: [10]        │
│                                                                 │
│   实现：Ring AllReduce                                           │
│   ┌───────────────────────────────────────────────────────┐     │
│   │  GPU 0 ──▶ GPU 1 ──▶ GPU 2 ──▶ GPU 3 ──▶ GPU 0       │     │
│   │    ◀──       ◀──       ◀──       ◀──       ◀──        │     │
│   └───────────────────────────────────────────────────────┘     │
│   通信量：O((N-1) × M / N)，M = 数据大小                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 同步开销分析

通信时间估算：
$$T_{comm} = 2 \times (N-1) \times \frac{M}{N \times B}$$

其中：
- $M$：模型参数量 × 4（FP32）或 × 2（FP16）
- $B$：网络带宽
- $N$：GPU 数量

**优化策略：梯度压缩**
- 通信量减少 10-100 倍
- 方法：量化、稀疏化、Top-K

## 2. 张量并行的数学原理

### 2.1 矩阵分块乘法

对于 $Y = XW$，将 $W$ 按列切分：

$$W = [W_1 | W_2 | ... | W_N]$$

$$Y = XW = [XW_1 | XW_2 | ... | XW_N]$$

每个 GPU 计算 $XW_i$，最后拼接。

### 2.2 Transformer 层的 TP

对于 MLP 层：
$$Y = \text{GELU}(XW_1) W_2$$

**切分策略：**
1. $W_1$ 按列切分：$W_1 = [W_1^{(1)} | W_1^{(2)}]$
2. 每个 GPU 计算：$Y_i = \text{GELU}(XW_1^{(i)})$
3. $W_2$ 按行切分：$W_2 = \begin{bmatrix} W_2^{(1)} \\ W_2^{(2)} \end{bmatrix}$
4. 每个 GPU 计算：$Z_i = Y_i W_2^{(i)}$
5. AllReduce 求和：$Z = Z_1 + Z_2$

### 2.3 通信分析

每层需要 2 次 AllReduce：
- Forward：1 次求和
- Backward：1 次广播梯度

**通信量：**
$$T_{comm} = 2 \times L \times \frac{H \times S \times 2}{B}$$

其中 $L$ 是层数，$H$ 是隐藏维度，$S$ 是序列长度。

## 3. 流水线并行的数学原理

### 3.1 流水线气泡

气泡比例（GPipe 调度）：
$$\text{Bubble Ratio} = \frac{P - 1}{M}$$

其中 $P$ 是 stage 数（GPU 数），$M$ 是 micro-batch 数。

### 3.2 1F1B 调度

1F1B（One Forward One Backward）减少气泡：

```
┌─────────────────────────────────────────────────────────────────┐
│                     1F1B 调度                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   GPU 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 ...                  │
│   GPU 1:    F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 ...               │
│   GPU 2:       F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 ...            │
│   GPU 3:          F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 ...         │
│                                                                 │
│   F = Forward, B = Backward                                     │
│   气泡减少到 (P-1) 个 micro-batch 时间                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 4. ZeRO 优化详解

### 4.1 内存分析

假设模型参数量为 $\Psi$，使用 Adam 优化器：

| 状态 | 大小 | 说明 |
|------|------|------|
| 参数 | $2\Psi$ | FP16 |
| 梯度 | $2\Psi$ | FP16 |
| 动量 | $4\Psi$ | FP32 |
| 方差 | $4\Psi$ | FP32 |
| FP32 参数副本 | $4\Psi$ | 主参数 |
| **总计** | $16\Psi$ | |

**示例：** 7B 模型需要 $16 \times 7 = 112$ GB 内存

### 4.2 ZeRO-3 内存节省

ZeRO-3 将所有状态分片：
$$\text{每卡内存} = \frac{16\Psi}{N}$$

**示例：** 7B 模型，8 卡
- 原始：112 GB / 卡
- ZeRO-3：14 GB / 卡

### 4.3 通信开销

ZeRO 增加通信量：

| Stage | 通信增加 |
|-------|----------|
| ZeRO-1 | 0x |
| ZeRO-2 | 1.5x |
| ZeRO-3 | 3x |

但通过 **通信重叠**（computation-communication overlap），实际开销可以隐藏。

## 5. 混合精度训练

### 5.1 FP16 动态范围

| 格式 | 最大值 | 最小正值 |
|------|--------|----------|
| FP32 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ |
| FP16 | $65504$ | $6.1 \times 10^{-5}$ |
| BF16 | $3.4 \times 10^{38}$ | $1.2 \times 10^{-38}$ |

**问题：** FP16 容易上溢（> 65504）或下溢（< 6.1e-5）

### 5.2 Loss Scaling

解决梯度下溢：

$$\tilde{g} = g \times S$$

其中 $S$ 是缩放因子（如 32768）。

**动态缩放：** 自动调整 $S$
- 梯度正常：保持或增大 $S$
- 梯度溢出：减小 $S$，跳过该步

### 5.3 BF16 优势

BF16（Brain Float 16）：
- 和 FP32 相同的指数位（8 bit）
- 精度降低（7 bit mantissa → FP32 的 23 bit）
- 不需要 loss scaling

## 6. 实践建议

### 6.1 选择并行策略

| 模型大小 | GPU 显存 | 推荐策略 |
|----------|----------|----------|
| < 7B | 40GB+ | DP + ZeRO-2 |
| 7B-70B | 80GB | TP + PP 或 ZeRO-3 |
| > 70B | 80GB | 3D 并行 |

### 6.2 通信优化

1. **梯度累积：** 减少通信频率
2. **梯度压缩：** 通信量减少 10x+
3. **通信重叠：** 计算时异步通信

### 6.3 常用框架

| 框架 | 特点 |
|------|------|
| DeepSpeed | ZeRO 实现，成熟稳定 |
| Megatron-LM | TP + PP 实现，适合超大模型 |
| FSDP | PyTorch 原生，易用 |
| Accelerate | Hugging Face，简单易用 |

## 参考文献

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
