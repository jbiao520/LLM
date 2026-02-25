"""
RMS Normalization 示例代码 / RMS Normalization Example Code
===========================================================

本示例展示 RMS Normalization 的原理和实现。
This example demonstrates the principle and implementation of RMS Normalization.

RMSNorm 是 LayerNorm 的简化版本，被 LLaMA 等现代 LLM 广泛采用。
RMSNorm is a simplified version of LayerNorm, widely adopted in modern LLMs like LLaMA.

依赖安装 / Dependencies:
    pip install torch matplotlib
"""

import torch
import torch.nn as nn
import time

# =============================================================================
# 1. RMSNorm 手动实现 / Manual Implementation of RMSNorm
# =============================================================================

class RMSNormManual(nn.Module):
    """
    手动实现的 RMS Normalization / Manual implementation of RMS Normalization

    数学公式 / Mathematical formula:
        y = x / sqrt(mean(x^2) + eps) * gamma

    与 LayerNorm 的区别 / Difference from LayerNorm:
        - 不减去均值 / No mean subtraction
        - 只有一个可学习参数 gamma / Only one learnable parameter (gamma)
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 计算 RMS / Compute RMS
        # x: [batch, seq_len, dim]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并缩放 / Normalize and scale
        return x / rms * self.gamma

# PyTorch 2.0+ 提供官方实现 / PyTorch 2.0+ provides official implementation
# nn.RMSNorm(dim, eps=1e-6)

# =============================================================================
# 2. RMSNorm vs LayerNorm 对比 / Compare RMSNorm vs LayerNorm
# =============================================================================

print("=" * 60)
print("RMSNorm vs LayerNorm 对比 / Comparison")
print("=" * 60)

torch.manual_seed(42)

dim = 64
batch_size, seq_len = 4, 128

# 创建测试数据 / Create test data
x = torch.randn(batch_size, seq_len, dim) * 10 + 5  # 带偏移的数据 / Biased data

# RMSNorm
rms_norm = RMSNormManual(dim)
output_rms = rms_norm(x)

# LayerNorm
layer_norm = nn.LayerNorm(dim)
output_ln = layer_norm(x)

print(f"\n输入统计 / Input statistics:")
print(f"  均值 / Mean: {x.mean().item():.4f}")
print(f"  标准差 / Std: {x.std().item():.4f}")
print(f"  RMS: {torch.sqrt(torch.mean(x ** 2)).item():.4f}")

print(f"\nRMSNorm 输出统计 / RMSNorm output statistics:")
print(f"  均值 / Mean: {output_rms.mean().item():.4f}")
print(f"  标准差 / Std: {output_rms.std().item():.4f}")
print(f"  RMS: {torch.sqrt(torch.mean(output_rms ** 2)).item():.4f}")

print(f"\nLayerNorm 输出统计 / LayerNorm output statistics:")
print(f"  均值 / Mean: {output_ln.mean().item():.4f}")
print(f"  标准差 / Std: {output_ln.std().item():.4f}")
print(f"  RMS: {torch.sqrt(torch.mean(output_ln ** 2)).item():.4f}")

# =============================================================================
# 3. 性能对比 / Performance Comparison
# =============================================================================

print("\n" + "=" * 60)
print("性能对比 / Performance Comparison")
print("=" * 60)

# 大规模测试 / Large-scale test
dim = 4096  # LLaMA-style dimension
batch_size, seq_len = 8, 2048
x_large = torch.randn(batch_size, seq_len, dim)

# 使用 GPU 如果可用 / Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_large = x_large.to(device)

rms_norm_gpu = RMSNormManual(dim).to(device)
layer_norm_gpu = nn.LayerNorm(dim).to(device)

# 预热 / Warmup
for _ in range(10):
    _ = rms_norm_gpu(x_large)
    _ = layer_norm_gpu(x_large)

# 计时 / Timing
torch.cuda.synchronize() if torch.cuda.is_available() else None

start = time.time()
for _ in range(100):
    _ = rms_norm_gpu(x_large)
torch.cuda.synchronize() if torch.cuda.is_available() else None
rms_time = time.time() - start

start = time.time()
for _ in range(100):
    _ = layer_norm_gpu(x_large)
torch.cuda.synchronize() if torch.cuda.is_available() else None
ln_time = time.time() - start

print(f"\n设备 / Device: {device}")
print(f"输入形状 / Input shape: {x_large.shape}")
print(f"RMSNorm 时间 / Time: {rms_time*1000:.2f} ms (100 次迭代)")
print(f"LayerNorm 时间 / Time: {ln_time*1000:.2f} ms (100 次迭代)")
print(f"加速比 / Speedup: {ln_time/rms_time:.2f}x")

# =============================================================================
# 4. 参数量对比 / Parameter Comparison
# =============================================================================

print("\n" + "=" * 60)
print("参数量对比 / Parameter Comparison")
print("=" * 60)

dim = 4096
rms = RMSNormManual(dim)
ln = nn.LayerNorm(dim)

print(f"\n隐藏维度 / Hidden dimension: {dim}")
print(f"RMSNorm 参数量 / Parameters: {sum(p.numel() for p in rms.parameters()):,} (仅 gamma)")
print(f"LayerNorm 参数量 / Parameters: {sum(p.numel() for p in ln.parameters()):,} (gamma + beta)")

# =============================================================================
# 5. LLaMA 风格的 RMSNorm 实现 / LLaMA-style RMSNorm Implementation
# =============================================================================

print("\n" + "=" * 60)
print("LLaMA 风格 RMSNorm / LLaMA-style RMSNorm")
print("=" * 60)

class LLaMARMSNorm(nn.Module):
    """
    LLaMA 风格的 RMSNorm 实现 / LLaMA-style RMSNorm implementation

    特点 / Features:
    1. 使用 weight 而非 gamma 命名 / Uses 'weight' instead of 'gamma'
    2. 精确的数值实现 / Precise numerical implementation
    3. 支持 FP32 归一化计算 / Supports FP32 normalization computation
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        归一化函数 / Normalization function
        在 FP32 中计算以保证数值稳定性
        Compute in FP32 for numerical stability
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 转换为 FP32 计算，然后转回原精度
        # Convert to FP32 for computation, then back to original dtype
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# 测试 LLaMA 风格实现 / Test LLaMA-style implementation
llama_norm = LLaMARMSNorm(dim=4096)
x_test = torch.randn(1, 100, 4096)
output = llama_norm(x_test)

print(f"\nLLaMA RMSNorm:")
print(f"  输入形状 / Input shape: {x_test.shape}")
print(f"  输出形状 / Output shape: {output.shape}")
print(f"  输出 RMS: {torch.sqrt(torch.mean(output ** 2)).item():.4f}")

# =============================================================================
# 6. 梯度流分析 / Gradient Flow Analysis
# =============================================================================

print("\n" + "=" * 60)
print("梯度流分析 / Gradient Flow Analysis")
print("=" * 60)

dim = 8
x = torch.randn(2, 4, dim, requires_grad=True)

# RMSNorm 梯度
rms = RMSNormManual(dim)
y_rms = rms(x)
loss_rms = y_rms.sum()
loss_rms.backward()

rms_grad = x.grad.clone()
x.grad = None

# LayerNorm 梯度
ln = nn.LayerNorm(dim)
y_ln = ln(x)
loss_ln = y_ln.sum()
loss_ln.backward()

ln_grad = x.grad.clone()

print(f"\nRMSNorm 输入梯度范数 / Input gradient norm: {rms_grad.norm().item():.4f}")
print(f"LayerNorm 输入梯度范数 / Input gradient norm: {ln_grad.norm().item():.4f}")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 RMSNorm 的核心概念:
This example demonstrates core concepts of RMSNorm:

1. 简化公式: y = x / sqrt(mean(x^2) + eps) * gamma
   Simplified formula

2. 相比 LayerNorm:
   Compared to LayerNorm:
   - 省去均值计算 / No mean calculation
   - 减少 25% 计算量 / ~25% less computation
   - 参数量减半 / Half the parameters

3. 被现代 LLM 广泛采用:
   Widely adopted in modern LLMs:
   - LLaMA
   - GPT-NeoX
   - PaLM
   - Chinchilla

4. 实践效果与 LayerNorm 相当或更好
   Practical performance equal to or better than LayerNorm

5. 适合大规模模型训练
   Suitable for large-scale model training
"""
