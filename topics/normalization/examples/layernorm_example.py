"""
Layer Normalization 示例代码 / Layer Normalization Example Code
==============================================================

本示例展示 Layer Normalization 的原理和实现。
This example demonstrates the principle and implementation of Layer Normalization.

LayerNorm 是 Transformer 和 LLM 中最常用的归一化方法。
LayerNorm is the most commonly used normalization method in Transformer and LLM.

依赖安装 / Dependencies:
    pip install torch matplotlib
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1. LayerNorm 手动实现 / Manual Implementation of LayerNorm
# =============================================================================

class LayerNormManual(nn.Module):
    """
    手动实现的 Layer Normalization / Manual implementation of Layer Normalization

    数学公式 / Mathematical formula:
        y = (x - mean) / sqrt(var + eps) * gamma + beta
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        # 可学习参数 / Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        # 计算最后一个维度的均值和方差 / Compute mean and variance along last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        # 归一化 / Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 缩放和平移 / Scale and shift
        return self.gamma * x_norm + self.beta

# =============================================================================
# 2. 对比手动实现与 PyTorch 官方实现 / Compare Manual vs PyTorch Implementation
# =============================================================================

print("=" * 60)
print("LayerNorm 实现对比 / LayerNorm Implementation Comparison")
print("=" * 60)

# 设置随机种子 / Set random seed
torch.manual_seed(42)

# 创建测试数据 / Create test data
batch_size, seq_len, hidden_dim = 2, 4, 8
x = torch.randn(batch_size, seq_len, hidden_dim)

# 手动实现 / Manual implementation
ln_manual = LayerNormManual(hidden_dim)
output_manual = ln_manual(x)

# PyTorch 官方实现 / PyTorch official implementation
ln_pytorch = nn.LayerNorm(hidden_dim)
# 复制参数 / Copy parameters
ln_pytorch.weight.data = ln_manual.gamma.data.clone()
ln_pytorch.bias.data = ln_manual.beta.data.clone()
output_pytorch = ln_pytorch(x)

# 比较结果 / Compare results
print(f"\n输入形状 / Input shape: {x.shape}")
print(f"输出形状 / Output shape: {output_manual.shape}")
print(f"最大差异 / Max difference: {(output_manual - output_pytorch).abs().max().item():.2e}")

# =============================================================================
# 3. 可视化归一化效果 / Visualize Normalization Effect
# =============================================================================

print("\n" + "=" * 60)
print("归一化效果可视化 / Normalization Effect Visualization")
print("=" * 60)

# 创建一个有偏移的数据 / Create biased data
x_biased = torch.tensor([[
    [10.0, 0.1, -5.0, 3.0, 8.0],   # 第一个词 / First token
    [0.5, 8.0, 0.2, -2.0, 1.0],     # 第二个词 / Second token
    [-3.0, 0.01, 12.0, 6.0, -1.0],  # 第三个词 / Third token
]])

ln = nn.LayerNorm(5)
x_normalized = ln(x_biased)

print(f"\n原始数据 (第一个词) / Original data (first token):")
print(f"  {x_biased[0, 0].tolist()}")
print(f"  均值 / Mean: {x_biased[0, 0].mean().item():.4f}")
print(f"  标准差 / Std: {x_biased[0, 0].std().item():.4f}")

print(f"\n归一化后 (第一个词) / After normalization (first token):")
print(f"  {x_normalized[0, 0].tolist()}")
print(f"  均值 / Mean: {x_normalized[0, 0].mean().item():.4f}")
print(f"  标准差 / Std: {x_normalized[0, 0].std().item():.4f}")

# 绘图 / Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 原始数据热力图 / Original data heatmap
ax1 = axes[0]
im1 = ax1.imshow(x_biased[0].numpy(), cmap='RdBu_r', aspect='auto')
ax1.set_title('Before LayerNorm / 归一化前', fontsize=12)
ax1.set_xlabel('Hidden Dimension / 隐藏维度')
ax1.set_ylabel('Token Position / Token 位置')
plt.colorbar(im1, ax=ax1)

# 归一化后热力图 / Normalized data heatmap
ax2 = axes[1]
im2 = ax2.imshow(x_normalized.detach().numpy()[0], cmap='RdBu_r', aspect='auto')
ax2.set_title('After LayerNorm / 归一化后', fontsize=12)
ax2.set_xlabel('Hidden Dimension / 隐藏维度')
ax2.set_ylabel('Token Position / Token 位置')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('/tmp/layernorm_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n图像已保存至 / Image saved to: /tmp/layernorm_comparison.png")

# =============================================================================
# 4. Pre-LN vs Post-LN / Pre-LN vs Post-LN
# =============================================================================

print("\n" + "=" * 60)
print("Pre-LN vs Post-LN")
print("=" * 60)

class PostLNSublayer(nn.Module):
    """
    Post-LN: LayerNorm 在残差连接之后 / LayerNorm after residual connection
    原始 Transformer 使用的方式 / Used in original Transformer
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)

class PreLNSublayer(nn.Module):
    """
    Pre-LN: LayerNorm 在子层之前 / LayerNorm before sublayer
    现代 Transformer 常用方式 / Commonly used in modern Transformers
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, sublayer_fn):
        return x + sublayer_fn(self.norm(x))

print("""
Pre-LN 优势 / Advantages of Pre-LN:
1. 训练更稳定 / More stable training
2. 梯度流动更平滑 / Smoother gradient flow
3. 不需要 warmup / No warmup needed
4. 现代大模型普遍采用 / Widely adopted in modern LLMs
""")

# =============================================================================
# 5. 在 Transformer 中的使用 / Usage in Transformer
# =============================================================================

print("=" * 60)
print("Transformer 中的 LayerNorm / LayerNorm in Transformer")
print("=" * 60)

class TransformerBlock(nn.Module):
    """
    简化的 Transformer Block（使用 Pre-LN）
    Simplified Transformer Block (using Pre-LN)
    """
    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim)
        )

    def forward(self, x):
        # Pre-LN Attention
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Pre-LN FFN
        x = x + self.ffn(self.norm2(x))
        return x

# 创建并测试 / Create and test
block = TransformerBlock(hidden_dim=64, num_heads=4, ff_dim=256)
x = torch.randn(2, 10, 64)  # [batch, seq_len, hidden_dim]
output = block(x)

print(f"\nTransformer Block:")
print(f"  输入形状 / Input shape: {x.shape}")
print(f"  输出形状 / Output shape: {output.shape}")
print(f"  参数量 / Parameters: {sum(p.numel() for p in block.parameters()):,}")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 LayerNorm 的核心概念:
This example demonstrates core concepts of LayerNorm:

1. 数学原理: y = (x - mean) / sqrt(var + eps) * gamma + beta
   Mathematical principle

2. 沿特征维度归一化，而非 batch 维度
   Normalize along feature dimension, not batch dimension

3. 适合变长序列（NLP 任务）
   Suitable for variable-length sequences (NLP tasks)

4. Pre-LN 比 Post-LN 训练更稳定
   Pre-LN is more stable than Post-LN for training

5. 是 Transformer 和 LLM 的核心组件
   Core component of Transformer and LLM
"""
