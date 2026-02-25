"""
Multi-Head Attention 示例代码 / Multi-Head Attention Example Code
=================================================================

本示例展示 Multi-Head Attention 的原理和实现。
This example demonstrates the principle and implementation of Multi-Head Attention.

依赖安装 / Dependencies:
    pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Multi-Head Attention 手动实现 / Manual Implementation of MHA
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    手动实现的 Multi-Head Attention / Manual implementation of Multi-Head Attention

    数学公式 / Mathematical formula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
        where head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性投影层（合并所有头）/ Linear projections (combined for all heads)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, query, key, value, mask=None):
        """
        参数 / Args:
            query: [batch, seq_len_q, d_model]
            key: [batch, seq_len_k, d_model]
            value: [batch, seq_len_v, d_model] (seq_len_k == seq_len_v)
            mask: [batch, seq_len_q, seq_len_k] 或 [seq_len_q, seq_len_k]

        返回 / Returns:
            output: [batch, seq_len_q, d_model]
            attn_weights: [batch, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # 1. 线性投影 / Linear projections
        Q = self.W_q(query)  # [batch, seq_len_q, d_model]
        K = self.W_k(key)    # [batch, seq_len_k, d_model]
        V = self.W_v(value)  # [batch, seq_len_v, d_model]

        # 2. 分割成多个头 / Split into multiple heads
        # [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 3. 计算注意力分数 / Compute attention scores
        # [batch, num_heads, seq_len_q, d_k] @ [batch, num_heads, d_k, seq_len_k]
        # = [batch, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 4. 应用 mask / Apply mask
        if mask is not None:
            # 扩展 mask 以匹配多头维度 / Expand mask to match multi-head dimensions
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # [1, seq_len_q, seq_len_k]
            elif mask.dim() == 3:
                pass  # [batch, seq_len_q, seq_len_k]
            # 扩展到头维度 / Expand to head dimension
            mask = mask.unsqueeze(1)  # [batch, 1, seq_len_q, seq_len_k]
            scores = scores.masked_fill(mask == 1, float('-inf'))

        # 5. Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. 加权求和 / Weighted sum
        # [batch, num_heads, seq_len_q, seq_len_k] @ [batch, num_heads, seq_len_v, d_k]
        # = [batch, num_heads, seq_len_q, d_k]
        output = torch.matmul(attn_weights, V)

        # 7. 合并多头 / Concatenate heads
        # [batch, num_heads, seq_len_q, d_k] -> [batch, seq_len_q, num_heads, d_k]
        # -> [batch, seq_len_q, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 8. 最终线性投影 / Final linear projection
        output = self.W_o(output)

        return output, attn_weights

# =============================================================================
# 2. 测试 Multi-Head Attention / Test MHA
# =============================================================================

print("=" * 60)
print("Multi-Head Attention 测试 / MHA Test")
print("=" * 60)

torch.manual_seed(42)

d_model = 512
num_heads = 8
seq_len = 10
batch_size = 2

# 创建输入 / Create input
x = torch.randn(batch_size, seq_len, d_model)

# 创建 MHA 层 / Create MHA layer
mha = MultiHeadAttention(d_model, num_heads)

# 前向传播 / Forward pass
output, attn_weights = mha(x, x, x)  # Self-Attention

print(f"\n配置 / Configuration:")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  d_k (per head): {d_model // num_heads}")

print(f"\n输入形状 / Input shape: {x.shape}")
print(f"输出形状 / Output shape: {output.shape}")
print(f"注意力权重形状 / Attention weights shape: {attn_weights.shape}")

# 参数统计 / Parameter statistics
total_params = sum(p.numel() for p in mha.parameters())
print(f"\n参数量 / Total parameters: {total_params:,}")

# =============================================================================
# 3. 可视化多头注意力 / Visualize Multi-Head Attention
# =============================================================================

print("\n" + "=" * 60)
print("多头注意力可视化 / Multi-Head Attention Visualization")
print("=" * 60)

# 创建一个小型示例用于可视化 / Create a small example for visualization
d_model_small = 64
num_heads_small = 4
seq_len_small = 6

mha_small = MultiHeadAttention(d_model_small, num_heads_small)
x_small = torch.randn(1, seq_len_small, d_model_small)

# 获取注意力权重 / Get attention weights
with torch.no_grad():
    _, attn_small = mha_small(x_small, x_small, x_small)

# 可视化每个头的注意力 / Visualize attention from each head
fig, axes = plt.subplots(1, num_heads_small, figsize=(16, 4))

tokens = [f"T{i}" for i in range(seq_len_small)]

for head in range(num_heads_small):
    ax = axes[head]
    attn_matrix = attn_small[0, head].numpy()
    im = ax.imshow(attn_matrix, cmap='Blues', vmin=0, vmax=1)
    ax.set_title(f'Head {head + 1}', fontsize=12)
    ax.set_xticks(range(seq_len_small))
    ax.set_yticks(range(seq_len_small))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    ax.set_xlabel('Key')
    ax.set_ylabel('Query')
    plt.colorbar(im, ax=ax)

plt.suptitle('Multi-Head Attention Weights', fontsize=14)
plt.tight_layout()
plt.savefig('/tmp/mha_heads.png', dpi=150, bbox_inches='tight')
print(f"\n多头注意力图像已保存至 / MHA image saved to: /tmp/mha_heads.png")

# =============================================================================
# 4. 平均注意力 vs 各头注意力 / Average vs Individual Heads
# =============================================================================

print("\n" + "=" * 60)
print("注意力模式分析 / Attention Pattern Analysis")
print("=" * 60)

# 计算平均注意力 / Compute average attention
avg_attn = attn_small[0].mean(dim=0).numpy()

print(f"\n各头注意力模式的特点 / Characteristics of each head's attention pattern:")

for head in range(num_heads_small):
    attn_head = attn_small[0, head].numpy()
    # 找到每个 query 最关注的 key / Find most attended key for each query
    max_attended = attn_head.argmax(axis=1)
    print(f"\n  Head {head + 1}:")
    print(f"    每个 query 最关注的位置 / Most attended position for each query:")
    for q, k in enumerate(max_attended):
        print(f"      Query {q} -> Key {k} (权重 {attn_head[q, k]:.3f})")

# 可视化平均注意力 / Visualize average attention
plt.figure(figsize=(6, 5))
plt.imshow(avg_attn, cmap='Blues')
plt.colorbar(label='Attention Weight')
plt.xticks(range(seq_len_small), tokens)
plt.yticks(range(seq_len_small), tokens)
plt.xlabel('Key')
plt.ylabel('Query')
plt.title('Average Attention Across All Heads')

# 添加数值 / Add values
for i in range(seq_len_small):
    for j in range(seq_len_small):
        color = 'white' if avg_attn[i, j] > 0.3 else 'black'
        plt.text(j, i, f'{avg_attn[i, j]:.2f}', ha='center', va='center', color=color, fontsize=8)

plt.tight_layout()
plt.savefig('/tmp/mha_average.png', dpi=150, bbox_inches='tight')
print(f"\n平均注意力图像已保存至 / Average attention image saved to: /tmp/mha_average.png")

# =============================================================================
# 5. 与 PyTorch 内置实现对比 / Compare with PyTorch Implementation
# =============================================================================

print("\n" + "=" * 60)
print("与 PyTorch 内置 MHA 对比 / Compare with PyTorch MHA")
print("=" * 60)

# PyTorch 内置 MHA
mha_pytorch = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

# 测试数据
x_test = torch.randn(batch_size, seq_len, d_model)

# 我们的实现
with torch.no_grad():
    output_custom, attn_custom = mha(x_test, x_test, x_test)

# PyTorch 实现
with torch.no_grad():
    output_pytorch, attn_pytorch = mha_pytorch(x_test, x_test, x_test)

print(f"\n输出形状对比 / Output shape comparison:")
print(f"  自定义实现 / Custom: {output_custom.shape}")
print(f"  PyTorch 实现 / PyTorch: {output_pytorch.shape}")

print(f"\n注意力形状对比 / Attention shape comparison:")
print(f"  自定义实现 / Custom: {attn_custom.shape}")
print(f"  PyTorch 实现 / PyTorch: {attn_pytorch.shape}")

# 参数量对比
params_custom = sum(p.numel() for p in mha.parameters())
params_pytorch = sum(p.numel() for p in mha_pytorch.parameters())

print(f"\n参数量对比 / Parameter comparison:")
print(f"  自定义实现 / Custom: {params_custom:,}")
print(f"  PyTorch 实现 / PyTorch: {params_pytorch:,}")

# =============================================================================
# 6. Cross-Attention 示例 / Cross-Attention Example
# =============================================================================

print("\n" + "=" * 60)
print("Cross-Attention 示例 / Cross-Attention Example")
print("=" * 60)

# Cross-Attention: Query 来自解码器，Key/Value 来自编码器
# Cross-Attention: Query from decoder, Key/Value from encoder

encoder_output = torch.randn(batch_size, 8, d_model)  # 编码器输出 (8 个 token)
decoder_input = torch.randn(batch_size, 5, d_model)   # 解码器输入 (5 个 token)

# Cross-Attention
cross_attn_output, cross_attn_weights = mha(
    query=decoder_input,    # 解码器作为 Query
    key=encoder_output,     # 编码器作为 Key
    value=encoder_output    # 编码器作为 Value
)

print(f"\nCross-Attention:")
print(f"  编码器输出形状 / Encoder output shape: {encoder_output.shape}")
print(f"  解码器输入形状 / Decoder input shape: {decoder_input.shape}")
print(f"  Cross-Attention 输出形状 / Output shape: {cross_attn_output.shape}")
print(f"  注意力权重形状 / Attention weights shape: {cross_attn_weights.shape}")
print(f"  (5 个解码器位置各关注 8 个编码器位置)")

# =============================================================================
# 7. 性能测试 / Performance Test
# =============================================================================

print("\n" + "=" * 60)
print("性能测试 / Performance Test")
print("=" * 60)

import time

def benchmark_attention(mha, seq_len, batch_size, num_iterations=100, device='cpu'):
    """基准测试注意力计算 / Benchmark attention computation"""
    x = torch.randn(batch_size, seq_len, mha.d_model, device=device)
    mha = mha.to(device)

    # 预热 / Warmup
    for _ in range(10):
        _ = mha(x, x, x)

    # 计时 / Timing
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    for _ in range(num_iterations):
        _ = mha(x, x, x)
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = time.time() - start

    return elapsed / num_iterations * 1000  # ms per iteration

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mha_perf = MultiHeadAttention(d_model=512, num_heads=8)

print(f"\n设备 / Device: {device}")
print(f"d_model=512, num_heads=8")

for seq_len in [64, 128, 256, 512]:
    avg_time = benchmark_attention(mha_perf, seq_len, batch_size=4, device=device)
    print(f"  seq_len={seq_len:4d}: {avg_time:.2f} ms per forward pass")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 Multi-Head Attention 的核心概念:
This example demonstrates core concepts of Multi-Head Attention:

1. 多头的作用 / Role of multiple heads:
   - 每个头学习不同的表示子空间
   - 可以同时关注不同位置的不同信息

2. 实现要点 / Implementation details:
   - 将 Q, K, V 分割成 num_heads 个头
   - 每个头独立计算注意力
   - 最后合并所有头的输出

3. 参数效率 / Parameter efficiency:
   - 参数量与单头相同（因为每个头的维度减小）
   - 但表达能力更强

4. Cross-Attention:
   - Query 来自一个序列
   - Key/Value 来自另一个序列
   - 用于编码器-解码器结构

5. 复杂度 / Complexity:
   - 时间: O(n^2 * d)
   - 空间: O(n^2 * h) 用于存储每头的注意力权重
"""
