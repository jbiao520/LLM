"""
Self-Attention 示例代码 / Self-Attention Example Code
=====================================================

本示例展示 Self-Attention 的原理和实现。
This example demonstrates the principle and implementation of Self-Attention.

依赖安装 / Dependencies:
    pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Self-Attention 手动实现 / Manual Implementation of Self-Attention
# =============================================================================

class SelfAttentionManual(nn.Module):
    """
    手动实现的 Self-Attention / Manual implementation of Self-Attention

    数学公式 / Mathematical formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    def __init__(self, d_model, d_k=None, d_v=None):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model

        # 线性投影层 / Linear projection layers
        self.W_q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, self.d_v, bias=False)

        self.scale = self.d_k ** -0.5

    def forward(self, x, mask=None):
        """
        参数 / Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len, seq_len] 或 [seq_len, seq_len]
                  True/1 表示要 mask 的位置

        返回 / Returns:
            output: [batch, seq_len, d_v]
            attn_weights: [batch, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # 计算 Q, K, V / Compute Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_k]
        K = self.W_k(x)  # [batch, seq_len, d_k]
        V = self.W_v(x)  # [batch, seq_len, d_v]

        # 计算注意力分数 / Compute attention scores
        # [batch, seq_len, d_k] @ [batch, d_k, seq_len] = [batch, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 应用 mask / Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))

        # Softmax / Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和 / Weighted sum
        # [batch, seq_len, seq_len] @ [batch, seq_len, d_v] = [batch, seq_len, d_v]
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

# =============================================================================
# 2. 测试 Self-Attention / Test Self-Attention
# =============================================================================

print("=" * 60)
print("Self-Attention 测试 / Self-Attention Test")
print("=" * 60)

torch.manual_seed(42)

d_model = 8
seq_len = 5
batch_size = 2

# 创建输入 / Create input
x = torch.randn(batch_size, seq_len, d_model)

# 创建 Self-Attention 层 / Create Self-Attention layer
self_attn = SelfAttentionManual(d_model)

# 前向传播 / Forward pass
output, attn_weights = self_attn(x)

print(f"\n输入形状 / Input shape: {x.shape}")
print(f"输出形状 / Output shape: {output.shape}")
print(f"注意力权重形状 / Attention weights shape: {attn_weights.shape}")

# 验证注意力权重和为 1 / Verify attention weights sum to 1
print(f"\n注意力权重和 / Attention weights sum (应为 1):")
print(f"  {attn_weights[0, 0].sum().item():.6f}")

# =============================================================================
# 3. 可视化注意力权重 / Visualize Attention Weights
# =============================================================================

print("\n" + "=" * 60)
print("注意力权重可视化 / Attention Weights Visualization")
print("=" * 60)

# 创建一个更有意义的示例 / Create a more meaningful example
# 模拟句子 "The cat sat on the mat"
sentence = ["The", "cat", "sat", "on", "the", "mat"]
seq_len = len(sentence)

# 创建词嵌入（随机）/ Create word embeddings (random)
embedding = nn.Embedding(seq_len, d_model)
indices = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]
x_sentence = embedding(indices)  # [1, seq_len, d_model]

# 计算注意力 / Compute attention
_, attn = self_attn(x_sentence)

# 可视化 / Visualize
plt.figure(figsize=(8, 6))
plt.imshow(attn[0].detach().numpy(), cmap='Blues')
plt.colorbar(label='Attention Weight')
plt.xticks(range(seq_len), sentence, rotation=45)
plt.yticks(range(seq_len), sentence)
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.title('Self-Attention Weights')

# 添加数值标注 / Add value annotations
for i in range(seq_len):
    for j in range(seq_len):
        value = attn[0, i, j].item()
        color = 'white' if value > 0.3 else 'black'
        plt.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontsize=8)

plt.tight_layout()
plt.savefig('/tmp/self_attention_weights.png', dpi=150, bbox_inches='tight')
print(f"\n注意力权重图像已保存至 / Attention weights image saved to: /tmp/self_attention_weights.png")

# =============================================================================
# 4. Causal Mask 示例 / Causal Mask Example
# =============================================================================

print("\n" + "=" * 60)
print("Causal Mask 示例 / Causal Mask Example")
print("=" * 60)

def create_causal_mask(seq_len):
    """
    创建因果掩码（上三角矩阵）
    Create causal mask (upper triangular matrix)

    位置 i 只能看到位置 0 到 i
    Position i can only see positions 0 to i
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask

causal_mask = create_causal_mask(6)
print(f"\n因果掩码 / Causal mask (1 表示要遮蔽):")
print(causal_mask.numpy())

# 应用因果掩码的注意力 / Attention with causal mask
output_masked, attn_masked = self_attn(x_sentence, mask=causal_mask)

print(f"\n带因果掩码的注意力权重 / Masked attention weights:")
print(attn_masked[0].detach().numpy().round(3))

# 可视化因果注意力 / Visualize causal attention
plt.figure(figsize=(8, 6))
plt.imshow(attn_masked[0].detach().numpy(), cmap='Blues')
plt.colorbar(label='Attention Weight')
plt.xticks(range(seq_len), sentence, rotation=45)
plt.yticks(range(seq_len), sentence)
plt.xlabel('Key Position (can attend to)')
plt.ylabel('Query Position')
plt.title('Causal Self-Attention Weights')

plt.tight_layout()
plt.savefig('/tmp/causal_attention_weights.png', dpi=150, bbox_inches='tight')
print(f"\n因果注意力图像已保存至 / Causal attention image saved to: /tmp/causal_attention_weights.png")

# =============================================================================
# 5. PyTorch 内置 MultiheadAttention 对比 / Compare with PyTorch MHA
# =============================================================================

print("\n" + "=" * 60)
print("与 PyTorch 内置 MHA 对比 / Compare with PyTorch MHA")
print("=" * 60)

# PyTorch 的 MultiheadAttention
mha = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

# 复制权重 / Copy weights
with torch.no_grad():
    # PyTorch MHA 的权重组织方式不同
    # 这里只是演示使用方法
    pass

# 使用 PyTorch MHA / Use PyTorch MHA
x_test = torch.randn(2, 5, d_model)
output_pytorch, attn_pytorch = mha(x_test, x_test, x_test)

print(f"\nPyTorch MHA 输出形状 / Output shape: {output_pytorch.shape}")
print(f"PyTorch MHA 注意力形状 / Attention shape: {attn_pytorch.shape}")

# =============================================================================
# 6. 注意力分数分析 / Attention Score Analysis
# =============================================================================

print("\n" + "=" * 60)
print("注意力分数分析 / Attention Score Analysis")
print("=" * 60)

# 分析缩放因子的影响 / Analyze the effect of scaling factor
d_k = 64
q = torch.randn(1, 10, d_k)
k = torch.randn(1, 10, d_k)

# 不缩放的点积 / Unscaled dot product
scores_unscaled = torch.matmul(q, k.transpose(-2, -1))
print(f"\n不缩放的分数 / Unscaled scores:")
print(f"  均值 / Mean: {scores_unscaled.mean().item():.4f}")
print(f"  标准差 / Std: {scores_unscaled.std().item():.4f}")
print(f"  最大值 / Max: {scores_unscaled.max().item():.4f}")
print(f"  最小值 / Min: {scores_unscaled.min().item():.4f}")

# 缩放后的点积 / Scaled dot product
scores_scaled = scores_unscaled / (d_k ** 0.5)
print(f"\n缩放后的分数 / Scaled scores:")
print(f"  均值 / Mean: {scores_scaled.mean().item():.4f}")
print(f"  标准差 / Std: {scores_scaled.std().item():.4f}")
print(f"  最大值 / Max: {scores_scaled.max().item():.4f}")
print(f"  最小值 / Min: {scores_scaled.min().item():.4f}")

# Softmax 后的分布对比 / Compare softmax distributions
softmax_unscaled = F.softmax(scores_unscaled, dim=-1)
softmax_scaled = F.softmax(scores_scaled, dim=-1)

print(f"\nSoftmax 分布熵 / Softmax distribution entropy:")
print(f"  不缩放 / Unscaled: {-(softmax_unscaled * torch.log(softmax_unscaled + 1e-9)).sum().item():.4f}")
print(f"  缩放后 / Scaled: {-(softmax_scaled * torch.log(softmax_scaled + 1e-9)).sum().item():.4f}")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 Self-Attention 的核心概念:
This example demonstrates core concepts of Self-Attention:

1. 计算步骤 / Computation steps:
   - Q = X @ W_q (查询投影)
   - K = X @ W_k (键投影)
   - V = X @ W_v (值投影)
   - Scores = Q @ K^T / sqrt(d_k) (缩放点积)
   - Attention = softmax(Scores) @ V (加权求和)

2. 缩放因子 / Scaling factor:
   - 除以 sqrt(d_k) 防止点积过大
   - 使 softmax 梯度更稳定

3. Causal Mask:
   - 解码器中防止看到未来信息
   - 上三角矩阵实现

4. 复杂度 / Complexity:
   - 时间: O(n^2 * d)
   - 空间: O(n^2)
"""
