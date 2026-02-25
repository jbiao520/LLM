"""
前馈网络层示例代码 / Feed-Forward Network Layer Example Code
=============================================================

本示例展示 Transformer 中前馈网络层的原理和实现。
This example demonstrates the principle and implementation of FFN in Transformer.

依赖安装 / Dependencies:
    pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. 标准 FFN 实现 / Standard FFN Implementation
# =============================================================================

class StandardFFN(nn.Module):
    """
    标准的 Transformer FFN 层 / Standard Transformer FFN Layer

    结构: Linear(d_model, d_ff) -> Activation -> Linear(d_ff, d_model)
    """
    def __init__(self, d_model, d_ff, activation='gelu', dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # 选择激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'silu':
            self.activation = F.silu
        else:
            self.activation = F.gelu

    def forward(self, x):
        """
        参数 / Args:
            x: [batch, seq_len, d_model]

        返回 / Returns:
            output: [batch, seq_len, d_model]
        """
        # 升维 + 激活
        x = self.activation(self.w1(x))
        # Dropout
        x = self.dropout(x)
        # 降维
        x = self.w2(x)
        return x

# =============================================================================
# 2. 测试标准 FFN / Test Standard FFN
# =============================================================================

print("=" * 60)
print("标准 FFN 测试 / Standard FFN Test")
print("=" * 60)

torch.manual_seed(42)

d_model = 768
d_ff = 3072  # 通常是 d_model 的 4 倍
batch_size, seq_len = 2, 128

# 创建 FFN
ffn = StandardFFN(d_model, d_ff, activation='gelu')

# 创建输入
x = torch.randn(batch_size, seq_len, d_model)

# 前向传播
output = ffn(x)

print(f"\n配置 / Configuration:")
print(f"  d_model: {d_model}")
print(f"  d_ff: {d_ff}")
print(f"  扩展比例 / Expansion ratio: {d_ff / d_model}x")

print(f"\n输入形状 / Input shape: {x.shape}")
print(f"输出形状 / Output shape: {output.shape}")

# 参数统计
params = sum(p.numel() for p in ffn.parameters())
print(f"\n参数量 / Parameters: {params:,}")

# 理论参数量
theoretical_params = d_model * d_ff + d_ff + d_ff * d_model + d_model
print(f"理论参数量 / Theoretical parameters: {theoretical_params:,}")

# =============================================================================
# 3. 不同激活函数对比 / Compare Different Activations
# =============================================================================

print("\n" + "=" * 60)
print("不同激活函数对比 / Activation Function Comparison")
print("=" * 60)

activations = ['relu', 'gelu', 'silu']
ffn_layers = {act: StandardFFN(d_model, d_ff, act) for act in activations}

# 对比输出
print(f"\n相同输入下不同激活函数的输出统计:")
for act, layer in ffn_layers.items():
    with torch.no_grad():
        out = layer(x)
    print(f"  {act.upper()}: mean={out.mean().item():.4f}, std={out.std().item():.4f}")

# 可视化激活函数
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

x_vis = torch.linspace(-3, 3, 100)

for ax, (act_name, act_fn) in zip(axes, [
    ('ReLU', F.relu),
    ('GELU', F.gelu),
    ('SiLU/Swish', F.silu)
]):
    y = act_fn(x_vis).numpy()
    ax.plot(x_vis.numpy(), y, linewidth=2)
    ax.set_title(act_name, fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/ffn_activations.png', dpi=150, bbox_inches='tight')
print(f"\n激活函数图像已保存至 / Image saved to: /tmp/ffn_activations.png")

# =============================================================================
# 4. FFN 中的信息流分析 / Information Flow Analysis in FFN
# =============================================================================

print("\n" + "=" * 60)
print("FFN 信息流分析 / FFN Information Flow Analysis")
print("=" * 60)

class FFNWithAnalysis(nn.Module):
    """带中间层分析的 FFN / FFN with intermediate analysis"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # 保存中间结果
        self.input = x
        self.hidden = F.gelu(self.w1(x))
        self.output = self.w2(self.hidden)
        return self.output

ffn_analysis = FFNWithAnalysis(d_model, d_ff)
ffn_analysis.eval()

with torch.no_grad():
    out = ffn_analysis(x)

print(f"\n各层张量统计:")
print(f"  输入: shape={ffn_analysis.input.shape}, "
      f"mean={ffn_analysis.input.mean().item():.4f}")
print(f"  隐藏层: shape={ffn_analysis.hidden.shape}, "
      f"mean={ffn_analysis.hidden.mean().item():.4f}")
print(f"  输出: shape={ffn_analysis.output.shape}, "
      f"mean={ffn_analysis.output.mean().item():.4f}")

# 分析隐藏层激活分布
hidden_flat = ffn_analysis.hidden.flatten().numpy()
print(f"\n隐藏层激活分布:")
print(f"  零值比例 / Zero ratio: {(hidden_flat == 0).mean():.2%}")
print(f"  小于0.1的比例 / <0.1 ratio: {(np.abs(hidden_flat) < 0.1).mean():.2%}")

# =============================================================================
# 5. FFN 在 Transformer 中的位置 / FFN Position in Transformer
# =============================================================================

print("\n" + "=" * 60)
print("FFN 在 Transformer 中 / FFN in Transformer")
print("=" * 60)

class TransformerBlockWithFFN(nn.Module):
    """展示 FFN 在 Transformer 中的位置"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Attention 部分
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        # FFN 部分
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = StandardFFN(d_model, d_ff, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention with Pre-LN
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = residual + self.dropout(x)

        # FFN with Pre-LN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        return x

# 创建并测试
block = TransformerBlockWithFFN(d_model=256, num_heads=8, d_ff=1024)
x_test = torch.randn(2, 64, 256)
out = block(x_test)

print(f"\nTransformer Block:")
print(f"  输入形状: {x_test.shape}")
print(f"  输出形状: {out.shape}")

# 参数分布
attn_params = sum(p.numel() for p in block.attention.parameters())
ffn_params = sum(p.numel() for p in block.ffn.parameters())
total_params = attn_params + ffn_params

print(f"\n参数分布:")
print(f"  Attention: {attn_params:,} ({attn_params/total_params:.1%})")
print(f"  FFN: {ffn_params:,} ({ffn_params/total_params:.1%})")

# =============================================================================
# 6. 偏置项的影响 / Effect of Bias Terms
# =============================================================================

print("\n" + "=" * 60)
print("偏置项影响分析 / Bias Term Analysis")
print("=" * 60)

class FFNWithBias(nn.Module):
    """带偏置的 FFN / FFN with bias"""
    def __init__(self, d_model, d_ff, use_bias=True):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class FFNWithoutBias(nn.Module):
    """不带偏置的 FFN / FFN without bias"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

ffn_with_bias = FFNWithBias(256, 1024)
ffn_no_bias = FFNWithoutBias(256, 1024)

params_with_bias = sum(p.numel() for p in ffn_with_bias.parameters())
params_no_bias = sum(p.numel() for p in ffn_no_bias.parameters())

print(f"\n参数量对比:")
print(f"  有偏置: {params_with_bias:,}")
print(f"  无偏置: {params_no_bias:,}")
print(f"  差异: {params_with_bias - params_no_bias:,} (偏置参数)")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 FFN 的核心概念:
This example demonstrates core concepts of FFN:

1. 结构 / Structure:
   - 升维: d_model -> d_ff (通常是 4 倍)
   - 激活: GELU/ReLU/SiLU
   - 降维: d_ff -> d_model

2. 参数量 / Parameters:
   - P ≈ 2 * d_model * d_ff
   - 约占 Transformer 参数的 2/3

3. 作用 / Function:
   - 提供非线性变换
   - 存储知识
   - 特征提取

4. 变体 / Variants:
   - 标准 FFN: Linear -> Activation -> Linear
   - SwiGLU: 门控机制
   - GeGLU: GELU 门控

5. 优化 / Optimization:
   - 可以移除偏置减少参数
   - 分组计算减少计算量
"""
