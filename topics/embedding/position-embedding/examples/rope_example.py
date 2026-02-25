"""
RoPE (Rotary Position Embedding) 示例代码
==========================================

本示例展示旋转位置编码 (RoPE) 的实现，这是现代 LLM (如 LLaMA, PaLM) 中广泛使用��位置编码方法。
This example demonstrates the implementation of Rotary Position Embedding (RoPE),
widely used in modern LLMs like LLaMA and PaLM.

论文 / Paper: RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)

RoPE 的核心思想是通过旋转矩阵将位置信息注入注意力计算中。
The core idea of RoPE is to inject positional information into attention computation
through rotation matrices.

关键优势 / Key Advantages:
1. 相对位置编码：注意力只依赖于相对位置 / Relative position encoding
2. 长度外推：可以处理比训练时更长的序列 / Length extrapolation
3. 无额外参数：不需要可学习的位置嵌入 / No additional parameters

依赖安装 / Dependencies:
    pip install torch numpy
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

# =============================================================================
# 1. RoPE 基础实现 / Basic RoPE Implementation
# =============================================================================

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    预计算 RoPE 的复数频率 / Precompute complex frequencies for RoPE

    RoPE 使用复数域中的旋转来编码位置:
    RoPE encodes position through rotation in the complex domain:

        f(θ) = e^(iθ) = cos(θ) + i*sin(θ)

    参数 / Args:
        dim: 嵌入维度（必须是偶数）/ Embedding dimension (must be even)
        max_seq_len: 最大序列长度 / Maximum sequence length
        theta: 基础频率（默认 10000）/ Base frequency (default 10000)

    返回 / Returns:
        复数频率张量 (max_seq_len, dim//2) / Complex frequency tensor
    """
    # 计算每个维度的频率 / Compute frequency for each dimension
    # freqs = 1 / (theta^(2i/d)) for i in [0, d/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 创建位置索引 / Create position indices
    t = torch.arange(max_seq_len)

    # 外积得到每个位置每个维度的角度 / Outer product to get angle for each position and dimension
    freqs = torch.outer(t, freqs)

    # 转换为复数形式 e^(i*θ) = cos(θ) + i*sin(θ)
    # Convert to complex form e^(i*θ) = cos(θ) + i*sin(θ)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 Query 和 Key 应用旋转位置编码 / Apply rotary positional encoding to Query and Key

    这是 RoPE 的核心操作。将 Query 和 Key 向量视为复数，乘以位置相关的旋转矩阵。
    This is the core operation of RoPE. Treat Query and Key vectors as complex numbers
    and multiply by position-dependent rotation matrices.

    参数 / Args:
        xq: Query 张量 (batch, seq_len, n_heads, head_dim)
        xk: Key 张量 (batch, seq_len, n_heads, head_dim)
        freqs_cis: 预计算的复数频率 (seq_len, head_dim//2)

    返回 / Returns:
        旋转后的 Query 和 Key / Rotated Query and Key
    """
    # 将实数向量重塑为复数形式 / Reshape real vectors to complex form
    # (batch, seq_len, n_heads, head_dim) -> (batch, seq_len, n_heads, head_dim//2, 2)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 获取当前序列长度的频率 / Get frequencies for current sequence length
    freqs_cis = freqs_cis[:xq.shape[1]]

    # 复数乘法实现旋转 / Complex multiplication implements rotation
    # freqs_cis 需要 broadcast 到 (batch, seq_len, n_heads, head_dim//2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# =============================================================================
# 2. RoPE 模块封装 / RoPE Module Wrapper
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE 模块 / RoPE Module

    封装了预计算频率和应用旋转编码的功能
    Wraps precomputed frequencies and rotary encoding application
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        """
        参数 / Args:
            dim: 每个注意力头的维度 / Dimension per attention head
            max_seq_len: 最大序列长度 / Maximum sequence length
            theta: 基础频率 / Base frequency
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 预计算频率并注册为 buffer / Precompute frequencies and register as buffer
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, xq: torch.Tensor, xk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置编码 / Apply rotary positional encoding

        参数 / Args:
            xq: Query 张量 / Query tensor
            xk: Key 张量 / Key tensor

        返回 / Returns:
            旋转后的 Query 和 Key / Rotated Query and Key
        """
        return apply_rotary_emb(xq, xk, self.freqs_cis)


# =============================================================================
# 3. 基本使用示例 / Basic Usage Example
# =============================================================================

# 模型参数 / Model parameters
batch_size = 2
seq_len = 16
n_heads = 8
head_dim = 64
d_model = n_heads * head_dim  # 512

print("="*60)
print("RoPE 基本使用示例 / Basic RoPE Usage Example")
print("="*60)

# 创建 RoPE 模块 / Create RoPE module
rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=2048)

# 模拟 Query 和 Key / Simulated Query and Key
xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
xk = torch.randn(batch_size, seq_len, n_heads, head_dim)

print(f"\n输入形状 / Input shapes:")
print(f"  Query: {xq.shape}")
print(f"  Key: {xk.shape}")

# 应用 RoPE / Apply RoPE
xq_rotated, xk_rotated = rope(xq, xk)

print(f"\n输出形状 / Output shapes:")
print(f"  Rotated Query: {xq_rotated.shape}")
print(f"  Rotated Key: {xk_rotated.shape}")

# =============================================================================
# 4. RoPE 的数学原理 / Mathematical Principles of RoPE
# =============================================================================

def explain_rope_math():
    """
    解释 RoPE 的数学原理 / Explain mathematical principles of RoPE
    """
    explanation = """
    RoPE 数学原理 / Mathematical Principles of RoPE
    ================================================

    1. 复数旋转 / Complex Rotation:
       对于 2D 向量 (x, y)，可以表示为复数 x + iy
       For a 2D vector (x, y), it can be represented as complex number x + iy

       旋转 θ 角度后的向量:
       Vector after rotating by angle θ:
           (x', y') = (x*cos(θ) - y*sin(θ), x*sin(θ) + y*cos(θ))

       复数形式:
       Complex form:
           (x' + iy') = (x + iy) * e^(iθ) = (x + iy) * (cos(θ) + i*sin(θ))

    2. RoPE 的位置编码 / Positional Encoding in RoPE:
       对于位置 m 的 token，其 Query/Key 向量 x_m 的第 i 个维度对 (2i, 2i+1):
       For token at position m, the i-th dimension pair (2i, 2i+1) of its Q/K vector x_m:

           θ_i = m / (base^(2i/d))

       旋转后:
       After rotation:
           x_m[2i]   = x_m[2i] * cos(θ_i) - x_m[2i+1] * sin(θ_i)
           x_m[2i+1] = x_m[2i] * sin(θ_i) + x_m[2i+1] * cos(θ_i)

    3. 相对位置特性 / Relative Position Property:
       两个位置 m 和 n 的点积:
       Dot product between positions m and n:

           <R_m * q, R_n * k> = f(m - n, q, k)

       只依赖于相对位置 (m - n)，不依赖于绝对位置
       Only depends on relative position (m - n), not absolute positions
    """
    print(explanation)


explain_rope_math()

# =============================================================================
# 5. 验证相对位置特性 / Verify Relative Position Property
# =============================================================================

def verify_relative_position():
    """
    验证 RoPE 的相对位置特性 / Verify relative position property of RoPE
    """
    print("\n" + "="*60)
    print("验证相对位置特性 / Verifying Relative Position Property")
    print("="*60)

    head_dim = 64
    max_len = 100
    rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=max_len)

    # 创建固定的 Query 和 Key 向量 / Create fixed Query and Key vectors
    q = torch.randn(1, 1, 1, head_dim)  # 单个 query
    k = torch.randn(1, 1, 1, head_dim)  # 单个 key

    # 测试不同位置对的注意力分数 / Test attention scores for different position pairs
    print("\n不同位置对的注意力分数 / Attention scores for different position pairs:")
    print("(理论上，相同相对距离的位置对应该有相似的注意力分数)")
    print("(Theoretically, position pairs with same relative distance should have similar scores)")

    relative_distances = [1, 5, 10, 20]

    for dist in relative_distances:
        scores = []
        for pos1 in [0, 10, 20, 30, 40]:
            pos2 = pos1 + dist
            if pos2 < max_len:
                # 为位置 pos1 创建 Query，位置 pos2 创建 Key
                q_pos = q.expand(1, max_len, 1, head_dim).clone()
                k_pos = k.expand(1, max_len, 1, head_dim).clone()

                # 应用 RoPE
                q_rot, k_rot = rope(q_pos, k_pos)

                # 计算注意力分数 (点积)
                score = (q_rot[0, pos1] * k_rot[0, pos2]).sum().item()
                scores.append(score)

        if scores:
            avg_score = sum(scores) / len(scores)
            variance = sum((s - avg_score)**2 for s in scores) / len(scores)
            print(f"  相对距离 {dist:2d}: 平均分数={avg_score:8.4f}, 方差={variance:.6f}")


verify_relative_position()

# =============================================================================
# 6. 完整的注意力层示例 / Complete Attention Layer Example
# =============================================================================

class RoPEAttention(nn.Module):
    """
    使用 RoPE 的多头注意力层 / Multi-head Attention Layer with RoPE

    这是 LLaMA 等模型中注意力层的简化实现
    Simplified implementation of attention layer in models like LLaMA
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ):
        """
        参数 / Args:
            d_model: 模型维度 / Model dimension
            n_heads: 注意力头数量 / Number of attention heads
            max_seq_len: 最大序列长度 / Maximum sequence length
            dropout: Dropout 概率 / Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Query, Key, Value 投影层 / Q, K, V projection layers
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        # 输出投影 / Output projection
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播 / Forward pass

        参数 / Args:
            x: 输入张量 (batch, seq_len, d_model) / Input tensor
            mask: 注意力掩码 (可选) / Attention mask (optional)

        返回 / Returns:
            输出张量 (batch, seq_len, d_model) / Output tensor
        """
        batch_size, seq_len, _ = x.shape

        # 计算 Q, K, V / Compute Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # 重塑为多头形式 / Reshape to multi-head form
        # (batch, seq_len, d_model) -> (batch, seq_len, n_heads, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # 应用 RoPE / Apply RoPE
        q, k = self.rope(q, k)

        # 转置用于注意力计算 / Transpose for attention computation
        # (batch, seq_len, n_heads, head_dim) -> (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数 / Compute attention scores
        # scores: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用掩码（如果有）/ Apply mask if provided
        if mask is not None:
            scores = scores + mask

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和 / Weighted sum
        # (batch, n_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # 转置回来 / Transpose back
        # (batch, seq_len, n_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # 合并多头 / Merge heads
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # 输出投影 / Output projection
        output = self.wo(attn_output)

        return output


# 使用示例 / Usage example
print("\n" + "="*60)
print("RoPE 注意力层示例 / RoPE Attention Layer Example")
print("="*60)

attention = RoPEAttention(d_model=512, n_heads=8, max_seq_len=2048)
x = torch.randn(2, 32, 512)
output = attention(x)

print(f"\n输入形状 / Input shape: {x.shape}")
print(f"输出形状 / Output shape: {output.shape}")
print(f"参数量 / Parameters: {sum(p.numel() for p in attention.parameters()):,}")

# =============================================================================
# 7. 与正弦位置编码的比较 / Comparison with Sinusoidal PE
# =============================================================================

def compare_rope_vs_sinusoidal():
    """
    比较 RoPE 和正弦位置编码 / Compare RoPE and Sinusoidal PE
    """
    comparison = """
    RoPE vs 正弦位置编码 / RoPE vs Sinusoidal Positional Encoding
    =============================================================

    | 特性 | RoPE | 正弦编码 |
    |------|------|----------|
    | 应用位置 | 仅 Q, K | 所有嵌入 |
    | 相对位置 | ✓ 精确 | △ 近似 |
    | 长度外推 | ✓ 强 | △ 中等 |
    | 计算复杂度 | 较高（复数运算）| 较低（加法）|
    | 参数量 | 0 | 0 |
    | 使用模型 | LLaMA, PaLM, GPT-NeoX | 原始 Transformer |

    RoPE 的优势 / Advantages of RoPE:
    1. 相对位置编码更精确 / More precise relative position encoding
    2. 更好的长度外推能力 / Better length extrapolation
    3. 在长序列上表现更好 / Better performance on long sequences

    正弦编码的优势 / Advantages of Sinusoidal:
    1. 实现更简单 / Simpler implementation
    2. 计算效率更高 / Higher computational efficiency
    3. 对所有嵌入统一处理 / Uniform treatment for all embeddings
    """
    print(comparison)


compare_rope_vs_sinusoidal()

# =============================================================================
# 8. 不同 theta 值的影响 / Effect of Different Theta Values
# =============================================================================

def analyze_theta_effect():
    """
    分析不同 theta 值对位置编码的影响
    Analyze effect of different theta values on positional encoding
    """
    print("\n" + "="*60)
    print("Theta 参数分析 / Theta Parameter Analysis")
    print("="*60)

    thetas = [100.0, 1000.0, 10000.0, 100000.0]
    head_dim = 64
    seq_len = 100

    print("\n不同 theta 值的频率范围 / Frequency ranges for different theta values:")
    print("(theta 越大，低频分量越多，能编码更长的距离关系)")
    print("(Larger theta means more low-frequency components, encoding longer distance relationships)")

    for theta in thetas:
        # 计算频率 / Compute frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

        print(f"\n  theta = {theta:.0f}:")
        print(f"    最高频率 / Highest freq: {freqs[0].item():.6f}")
        print(f"    最低频率 / Lowest freq:  {freqs[-1].item():.10f}")
        print(f"    周期范围 / Period range: {1/freqs[0].item():.1f} ~ {1/freqs[-1].item():.1f} 位置")


analyze_theta_effect()

# =============================================================================
# 9. 实际应用建议 / Practical Application Tips
# =============================================================================

PRACTICAL_TIPS = """
实际应用建议 / Practical Application Tips
==========================================

1. Theta 值选择 / Choosing Theta Value:
   - 标准值: 10000 (大多数模型使用)
   - 长序列: 可增大到 100000 或更多
   - 短序列: 可减小到 1000

2. 与其他位置编码结合 / Combining with Other PE:
   - RoPE 通常单独使用，不需要额外的位置编码
   - 在某些场景下可与 ALiBi 结合使用

3. 实现优化 / Implementation Optimizations:
   - 预计算频率并缓存 / Precompute and cache frequencies
   - 使用半精度 (fp16/bf16) 加速计算 / Use half precision for faster computation
   - 考虑使用 Flash Attention 进一步优化

4. 常见问题 / Common Issues:
   - 维度必须是偶数 / Dimension must be even
   - 序列长度不能超过 max_seq_len / Sequence length cannot exceed max_seq_len
   - 注意 dtype 一致性 / Pay attention to dtype consistency

5. 调试技巧 / Debugging Tips:
   - 检查旋转后的向量范数是否保持不变 / Check if rotated vector norm is preserved
   - 验证相对位置特性 / Verify relative position property
   - 可视化注意力矩阵 / Visualize attention matrix
"""

print(PRACTICAL_TIPS)

# =============================================================================
# 总结 / Summary
# =============================================================================
"""
本示例展示了 RoPE 的完整实现和使用:
This example demonstrates complete implementation and usage of RoPE:

1. 基本原理 / Basic Principles
   - 使用复数旋转编码位置 / Encode position using complex rotation
   - 不同维度使用不同频率 / Different dimensions use different frequencies

2. 核心优势 / Key Advantages
   - 相对位置编码 / Relative position encoding
   - 无额外参数 / No additional parameters
   - 良好的长度外推 / Good length extrapolation

3. 实现要点 / Implementation Points
   - 预计算频率 / Precompute frequencies
   - 复数乘法实现旋转 / Complex multiplication for rotation
   - 仅应用于 Q 和 K / Only apply to Q and K

4. 应用场景 / Use Cases
   - LLaMA, PaLM, GPT-NeoX 等现代 LLM
   - 需要处理长序列的场景
   - 需要相对位置信息的任务

参考资源 / References:
- RoFormer Paper: https://arxiv.org/abs/2104.09864
- LLaMA Implementation: https://github.com/facebookresearch/llama
"""
