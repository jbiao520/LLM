"""
Transformer Encoder 示例代码 / Transformer Encoder Example Code
===============================================================

本示例展示 Transformer Encoder 的原理和实现。
This example demonstrates the principle and implementation of Transformer Encoder.

依赖安装 / Dependencies:
    pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. 基础组件 / Basic Components
# =============================================================================

class MultiHeadAttention(nn.Module):
    """多头自注意力 / Multi-Head Self-Attention"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # 计算 Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)

        return output, attn


class FeedForward(nn.Module):
    """前馈网络 / Feed-Forward Network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))

# =============================================================================
# 2. Encoder Layer 实现 / Encoder Layer Implementation
# =============================================================================

class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer（使用 Pre-LN）
    Transformer Encoder Layer (using Pre-LN)
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Pre-LN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Sub-layers
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention with Pre-LN
        residual = x
        x = self.norm1(x)
        x, attn = self.attention(x, mask)
        x = residual + self.dropout(x)

        # FFN with Pre-LN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)

        return x, attn

# =============================================================================
# 3. 完整的 Transformer Encoder / Complete Transformer Encoder
# =============================================================================

class TransformerEncoder(nn.Module):
    """
    完整的 Transformer Encoder
    Complete Transformer Encoder
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_seq_len=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Encoder Layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        参数 / Args:
            x: [batch, seq_len] - token indices
            mask: [batch, seq_len, seq_len] - attention mask

        返回 / Returns:
            output: [batch, seq_len, d_model]
            attentions: list of attention weights from each layer
        """
        batch_size, seq_len = x.shape

        # Token + Position Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        # 通过所有 Encoder 层
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attentions.append(attn)

        # 最终 LayerNorm
        x = self.norm(x)

        return x, attentions

# =============================================================================
# 4. 测试 Encoder / Test Encoder
# =============================================================================

print("=" * 60)
print("Transformer Encoder 测试 / Transformer Encoder Test")
print("=" * 60)

torch.manual_seed(42)

# 配置 / Configuration
vocab_size = 10000
d_model = 256
num_heads = 8
d_ff = 1024
num_layers = 6
max_seq_len = 128
batch_size = 4
seq_len = 32

# 创建 Encoder / Create Encoder
encoder = TransformerEncoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    max_seq_len=max_seq_len
)

# 创建随机输入 / Create random input
x = torch.randint(0, vocab_size, (batch_size, seq_len))

# 前向传播 / Forward pass
output, attentions = encoder(x)

print(f"\n配置 / Configuration:")
print(f"  vocab_size: {vocab_size}")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  d_ff: {d_ff}")
print(f"  num_layers: {num_layers}")

print(f"\n输入形状 / Input shape: {x.shape}")
print(f"输出形状 / Output shape: {output.shape}")
print(f"注意力层数 / Number of attention layers: {len(attentions)}")
print(f"每层注意力形状 / Attention shape per layer: {attentions[0].shape}")

# 参数统计 / Parameter statistics
total_params = sum(p.numel() for p in encoder.parameters())
print(f"\n总参数量 / Total parameters: {total_params:,}")

# =============================================================================
# 5. 可视化各层注意力 / Visualize Attention from Each Layer
# =============================================================================

print("\n" + "=" * 60)
print("注意力可视化 / Attention Visualization")
print("=" * 60)

# 可视化不同层的注意力模式 / Visualize attention patterns from different layers
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (ax, attn) in enumerate(zip(axes, attentions)):
    # 取第一个样本的第一个头
    attn_map = attn[0, 0].detach().numpy()
    im = ax.imshow(attn_map, cmap='Blues')
    ax.set_title(f'Layer {i+1} (Head 1)')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax)

plt.suptitle('Self-Attention Patterns Across Layers', fontsize=14)
plt.tight_layout()
plt.savefig('/tmp/encoder_attention_layers.png', dpi=150, bbox_inches='tight')
print(f"\n注意力图像已保存至 / Attention image saved to: /tmp/encoder_attention_layers.png")

# =============================================================================
# 6. 编码器输出分析 / Encoder Output Analysis
# =============================================================================

print("\n" + "=" * 60)
print("编码器输出分析 / Encoder Output Analysis")
print("=" * 60)

# 分析输出表示 / Analyze output representations
print(f"\n输出统计 / Output statistics:")
print(f"  均值 / Mean: {output.mean().item():.4f}")
print(f"  标准差 / Std: {output.std().item():.4f}")
print(f"  最小值 / Min: {output.min().item():.4f}")
print(f"  最大值 / Max: {output.max().item():.4f}")

# 分析不同位置的表示 / Analyze representations at different positions
print(f"\n不同位置的表示范数 / Norm of representations at different positions:")
for pos in [0, seq_len//4, seq_len//2, seq_len-1]:
    norm = output[0, pos].norm().item()
    print(f"  位置 {pos}: {norm:.4f}")

# =============================================================================
# 7. 与 PyTorch 内置实现对比 / Compare with PyTorch Implementation
# =============================================================================

print("\n" + "=" * 60)
print("与 PyTorch 内置实现对比 / Compare with PyTorch Implementation")
print("=" * 60)

# PyTorch 内置的 Transformer Encoder
pytorch_encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=num_heads,
    dim_feedforward=d_ff,
    dropout=0.1,
    batch_first=True
)
pytorch_encoder = nn.TransformerEncoder(pytorch_encoder_layer, num_layers=num_layers)

# 创建嵌入后的输入 / Create embedded input
x_embedded = encoder.token_embedding(x) + encoder.position_embedding(
    torch.arange(seq_len).unsqueeze(0)
)

# 前向传播 / Forward pass
output_pytorch = pytorch_encoder(x_embedded)

print(f"\nPyTorch Encoder 输出形状 / Output shape: {output_pytorch.shape}")
print(f"PyTorch Encoder 参数量 / Parameters: {sum(p.numel() for p in pytorch_encoder.parameters()):,}")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 Transformer Encoder 的核心概念:
This example demonstrates core concepts of Transformer Encoder:

1. 结构 / Structure:
   - Multi-Head Self-Attention
   - Feed-Forward Network
   - Layer Normalization
   - Residual Connections

2. Pre-LN vs Post-LN:
   - Pre-LN 训练更稳定
   - 梯度流动更平滑

3. 特点 / Features:
   - 双向上下文理解
   - 并行计算
   - 长距离依赖

4. 应用 / Applications:
   - BERT
   - 文本分类
   - 命名实体识别
   - 问答系统
"""
