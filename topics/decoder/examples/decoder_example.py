"""
Transformer Decoder 示例代码 / Transformer Decoder Example Code
===============================================================

本示例展示 Transformer Decoder 的原理和实现。
This example demonstrates the principle and implementation of Transformer Decoder.

依赖安装 / Dependencies:
    pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. 因果掩码 / Causal Mask
# =============================================================================

def create_causal_mask(seq_len):
    """
    创建因果掩码（上三角矩阵）
    Create causal mask (upper triangular matrix)

    位置 i 只能看到位置 0 到 i-1
    Position i can only see positions 0 to i-1
    """
    # 创建下三角矩阵（包含对角线）
    mask = torch.tril(torch.ones(seq_len, seq_len))
    # 转换为 attention mask 格式: 0 表示要 mask
    mask = mask == 0  # True 表示要 mask 的位置
    return mask

# =============================================================================
# 2. Masked Multi-Head Self-Attention
# =============================================================================

class MaskedMultiHeadAttention(nn.Module):
    """带因果掩码的多头自注意力 / Masked Multi-Head Self-Attention"""
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

        # 应用因果掩码
        if mask is not None:
            # mask: [seq_len, seq_len] -> 扩展到 [batch, num_heads, seq_len, seq_len]
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 加权求和
        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)

        return output, attn

# =============================================================================
# 3. Decoder Layer
# =============================================================================

class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    包含 Masked Self-Attention 和 FFN
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Pre-LN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Sub-layers
        self.attention = MaskedMultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Masked Self-Attention
        residual = x
        x = self.norm1(x)
        x, attn = self.attention(x, mask)
        x = residual + self.dropout(x)

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, attn

# =============================================================================
# 4. 完整的 Transformer Decoder
# =============================================================================

class TransformerDecoder(nn.Module):
    """
    完整的 Transformer Decoder
    用于自回归生成
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_seq_len=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.norm = nn.LayerNorm(d_model)

        # Output projection (语言模型头)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (共享词嵌入和输出层权重)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, x):
        """
        参数 / Args:
            x: [batch, seq_len] - token indices

        返回 / Returns:
            logits: [batch, seq_len, vocab_size]
            attentions: list of attention weights
        """
        batch_size, seq_len = x.shape

        # Token + Position Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        # 创建因果掩码
        causal_mask = create_causal_mask(seq_len).to(x.device)

        # 通过所有 Decoder 层
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, causal_mask)
            attentions.append(attn)

        # 最终 LayerNorm
        x = self.norm(x)

        # 输出 logits
        logits = self.lm_head(x)

        return logits, attentions

# =============================================================================
# 5. 测试 Decoder / Test Decoder
# =============================================================================

print("=" * 60)
print("Transformer Decoder 测试 / Transformer Decoder Test")
print("=" * 60)

torch.manual_seed(42)

# 配置
vocab_size = 1000
d_model = 256
num_heads = 8
d_ff = 1024
num_layers = 6
max_seq_len = 128
batch_size = 2
seq_len = 16

# 创建 Decoder
decoder = TransformerDecoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    max_seq_len=max_seq_len
)

# 创建输入
x = torch.randint(0, vocab_size, (batch_size, seq_len))

# 前向传播
logits, attentions = decoder(x)

print(f"\n配置 / Configuration:")
print(f"  vocab_size: {vocab_size}")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  num_layers: {num_layers}")

print(f"\n输入形状 / Input shape: {x.shape}")
print(f"输出 logits 形状 / Output logits shape: {logits.shape}")
print(f"注意力层数 / Number of attention layers: {len(attentions)}")

# 参数统计
total_params = sum(p.numel() for p in decoder.parameters())
print(f"\n总参数量 / Total parameters: {total_params:,}")

# =============================================================================
# 6. 可视化因果掩码和注意力 / Visualize Causal Mask and Attention
# =============================================================================

print("\n" + "=" * 60)
print("因果掩码可视化 / Causal Mask Visualization")
print("=" * 60)

# 可视化因果掩码
causal_mask = create_causal_mask(8)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(causal_mask.numpy(), cmap='RdBu')
plt.title('Causal Mask\n(True = masked)')
plt.xlabel('Key Position')
plt.ylabel('Query Position')

# 可视化注意力矩阵（应用掩码后）
plt.subplot(1, 3, 2)
attn_example = attentions[0][0, 0].detach().numpy()  # 第一个样本第一个头
plt.imshow(attn_example[:8, :8], cmap='Blues')
plt.title('Attention Weights\n(with Causal Mask)')
plt.xlabel('Key Position')
plt.ylabel('Query Position')

# 检查注意力是否是因果的
plt.subplot(1, 3, 3)
# 检查上三角是否全为0
is_causal = (attn_example[:8, :8] < 1e-6)
plt.imshow(is_causal, cmap='Greens')
plt.title('Causal Check\n(Green = ~0)')
plt.xlabel('Key Position')
plt.ylabel('Query Position')

plt.tight_layout()
plt.savefig('/tmp/decoder_causal_attention.png', dpi=150, bbox_inches='tight')
print(f"\n因果注意力图像已保存至 / Image saved to: /tmp/decoder_causal_attention.png")

# =============================================================================
# 7. 自回归生成 / Autoregressive Generation
# =============================================================================

print("\n" + "=" * 60)
print("自回归生成演示 / Autoregressive Generation Demo")
print("=" * 60)

def generate(model, start_tokens, max_new_tokens=20, temperature=1.0):
    """
    自回归生成文本 / Autoregressive text generation

    参数 / Args:
        model: 语言模型
        start_tokens: 起始 token [batch, seq_len]
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度

    返回 / Returns:
        generated: 生成的 token 序列
    """
    model.eval()
    generated = start_tokens.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 获取最后一个位置的 logits
            logits, _ = model(generated)
            next_token_logits = logits[:, -1, :] / temperature

            # 采样下一个 token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 添加到序列
            generated = torch.cat([generated, next_token], dim=1)

    return generated

# 从单个 token 开始生成
start_token = torch.tensor([[1]])  # 假设 token 1 是开始符

generated = generate(decoder, start_token, max_new_tokens=10, temperature=0.8)

print(f"\n生成过程 / Generation process:")
print(f"  起始 token: {start_token[0].tolist()}")
print(f"  生成的 tokens: {generated[0].tolist()}")

# =============================================================================
# 8. 生成策略对比 / Generation Strategy Comparison
# =============================================================================

print("\n" + "=" * 60)
print("生成策略对比 / Generation Strategy Comparison")
print("=" * 60)

def greedy_decode(model, start_tokens, max_new_tokens):
    """贪婪解码 / Greedy decoding"""
    model.eval()
    generated = start_tokens.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return generated

def top_k_sample(model, start_tokens, max_new_tokens, k=50, temperature=1.0):
    """Top-K 采样 / Top-K sampling"""
    model.eval()
    generated = start_tokens.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-K 过滤
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
            probs = F.softmax(top_k_logits, dim=-1)

            # 从 Top-K 中采样
            idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(1, idx)

            generated = torch.cat([generated, next_token], dim=1)

    return generated

# 测试不同策略
start = torch.tensor([[1, 5, 10]])

greedy_result = greedy_decode(decoder, start, max_new_tokens=5)
top_k_result = top_k_sample(decoder, start, max_new_tokens=5, k=10)

print(f"\n起始序列: {start[0].tolist()}")
print(f"贪婪解码: {greedy_result[0].tolist()}")
print(f"Top-K 采样 (k=10): {top_k_result[0].tolist()}")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 Transformer Decoder 的核心概念:
This example demonstrates core concepts of Transformer Decoder:

1. 因果掩码 / Causal Mask:
   - 确保每个位置只能看到之前的位置
   - 实现方式：上三角矩阵设为 -inf

2. 自回归生成 / Autoregressive Generation:
   - 逐 token 生成
   - 用已生成的内容预测下一个

3. 生成策略 / Generation Strategies:
   - Greedy: 选择概率最高的
   - Sampling: 按概率采样
   - Top-K: 从前 K 个中采样
   - Temperature: 控制随机性

4. GPT 架构:
   - 纯解码器结构
   - 自监督语言建模
   - 生成能力强
"""
