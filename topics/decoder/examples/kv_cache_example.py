"""
KV Cache 示例代码 / KV Cache Example Code
=========================================

本示例展示 KV Cache 的原理和实现。
This example demonstrates the principle and implementation of KV Cache.

KV Cache 是加速 Transformer 推理的关键技术。
KV Cache is a key technique for accelerating Transformer inference.

依赖安装 / Dependencies:
    pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# =============================================================================
# 1. 问题：为什么需要 KV Cache？
# =============================================================================

print("=" * 60)
print("为什么需要 KV Cache？ / Why KV Cache?")
print("=" * 60)

print("""
在自回归生成中，每次生成一个新 token 时:

无缓存 / Without Cache:
- 需要重新计算所有之前 token 的 K 和 V
- 计算量随序列长度线性增长
- 大量重复计算

有缓存 / With Cache:
- 缓存之前计算过的 K 和 V
- 只需计算新 token 的 K 和 V
- 计算量恒定

示例: 生成第 10 个 token 时
- 无缓存: 处理 10 个 token
- 有缓存: 只处理 1 个 token
""")

# =============================================================================
# 2. 简单的 Attention 层（无缓存）/ Simple Attention Layer (No Cache)
# =============================================================================

class SimpleAttention(nn.Module):
    """不带 KV Cache 的注意力层 / Attention without KV Cache"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """x: [batch, seq_len, d_model]"""
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(output)

# =============================================================================
# 3. 带 KV Cache 的 Attention 层 / Attention Layer with KV Cache
# =============================================================================

class CachedAttention(nn.Module):
    """带 KV Cache 的注意力层 / Attention with KV Cache"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, past_kv=None, use_cache=False):
        """
        参数 / Args:
            x: [batch, seq_len, d_model]
            past_kv: 之前缓存的 (K, V) tuple
            use_cache: 是否返回 KV Cache

        返回 / Returns:
            output: [batch, seq_len, d_model]
            present_kv: 当前的 (K, V) 用于后续缓存
        """
        batch_size, seq_len, _ = x.shape

        # 计算当前输入的 Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 如果有过去的 KV，拼接
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 应用因果掩码（如果有过去 KV，只对新部分掩码）
        if past_kv is not None:
            # 新 token 可以看到所有之前的 token，所以不需要掩码
            pass
        else:
            # 完整的因果掩码
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)

        # 返回缓存
        present_kv = (K, V) if use_cache else None

        return output, present_kv

# =============================================================================
# 4. 完整的 Decoder 层（带 KV Cache）/ Complete Decoder Layer with KV Cache
# =============================================================================

class CachedDecoderLayer(nn.Module):
    """带 KV Cache 的 Decoder 层"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = CachedAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ffn, d_model)
        )

    def forward(self, x, past_kv=None, use_cache=False):
        # Attention
        residual = x
        x = self.norm1(x)
        x, present_kv = self.attention(x, past_kv, use_cache)
        x = residual + x

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, present_kv

# =============================================================================
# 5. 性能对比 / Performance Comparison
# =============================================================================

print("\n" + "=" * 60)
print("性能对比 / Performance Comparison")
print("=" * 60)

# 创建模型
d_model = 256
num_heads = 8
num_layers = 4
vocab_size = 1000

class SimpleDecoder(nn.Module):
    """不带 KV Cache 的简单 Decoder"""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SimpleAttention(d_model, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

class CachedDecoder(nn.Module):
    """带 KV Cache 的 Decoder"""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            CachedAttention(d_model, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, past_kvs=None, use_cache=False):
        x = self.embed(x)

        present_kvs = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs else None
            x, present_kv = layer(x, past_kv, use_cache)
            present_kvs.append(present_kv)

        x = self.norm(x)
        return self.head(x), present_kvs if use_cache else None

simple_decoder = SimpleDecoder()
cached_decoder = CachedDecoder()

# 测试生成速度
def generate_without_cache(model, start_tokens, num_tokens):
    """无缓存的生成"""
    tokens = start_tokens.clone()
    for _ in range(num_tokens):
        logits = model(tokens)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens

def generate_with_cache(model, start_tokens, num_tokens):
    """带缓存的生成"""
    tokens = start_tokens.clone()

    # 第一次处理所有 token
    logits, past_kvs = model(tokens, use_cache=True)
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = torch.cat([tokens, next_token], dim=1)

    # 后续只处理新 token
    for _ in range(num_tokens - 1):
        logits, past_kvs = model(next_token, past_kvs, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokens

# 性能测试
start_tokens = torch.randint(0, vocab_size, (1, 10))
num_new_tokens = 20

print(f"\n生成 {num_new_tokens} 个新 token:")
print(f"起始序列长度: {start_tokens.shape[1]}")

# 无缓存
start_time = time.time()
result_no_cache = generate_without_cache(simple_decoder, start_tokens, num_new_tokens)
time_no_cache = time.time() - start_time

# 有缓存
start_time = time.time()
result_with_cache = generate_with_cache(cached_decoder, start_tokens, num_new_tokens)
time_with_cache = time.time() - start_time

print(f"\n无缓存耗时 / Without cache: {time_no_cache*1000:.2f} ms")
print(f"有缓存耗时 / With cache: {time_with_cache*1000:.2f} ms")
print(f"加速比 / Speedup: {time_no_cache/time_with_cache:.2f}x")

# =============================================================================
# 6. KV Cache 内存分析 / KV Cache Memory Analysis
# =============================================================================

print("\n" + "=" * 60)
print("KV Cache 内存分析 / KV Cache Memory Analysis")
print("=" * 60)

def compute_kv_cache_size(batch_size, num_layers, num_heads, head_dim, seq_len, dtype_bytes=2):
    """
    计算 KV Cache 的内存大小

    参数:
        batch_size: 批大小
        num_layers: 层数
        num_heads: 注意力头数
        head_dim: 每个头的维度
        seq_len: 序列长度
        dtype_bytes: 数据类型字节数 (FP16 = 2, FP32 = 4)

    返回:
        内存大小 (MB)
    """
    # 每层有 K 和 V 两个缓存
    # 形状: [batch, num_heads, seq_len, head_dim]
    cache_per_layer = 2 * batch_size * num_heads * seq_len * head_dim * dtype_bytes
    total_cache = cache_per_layer * num_layers
    return total_cache / (1024 * 1024)  # 转换为 MB

# 典型配置下的 KV Cache 大小
configs = [
    {"name": "GPT-2 Small", "layers": 12, "heads": 12, "head_dim": 64},
    {"name": "GPT-2 Medium", "layers": 24, "heads": 16, "head_dim": 64},
    {"name": "GPT-3", "layers": 96, "heads": 96, "head_dim": 128},
]

print("\nKV Cache 大小 (batch_size=1, dtype=FP16):")
print(f"{'模型':<15} {'序列长度':<10} {'KV Cache (MB)':<15}")
print("-" * 45)

for config in configs:
    for seq_len in [512, 1024, 2048, 4096]:
        size_mb = compute_kv_cache_size(
            batch_size=1,
            num_layers=config["layers"],
            num_heads=config["heads"],
            head_dim=config["head_dim"],
            seq_len=seq_len,
            dtype_bytes=2
        )
        print(f"{config['name']:<15} {seq_len:<10} {size_mb:<15.2f}")
    print()

# =============================================================================
# 7. KV Cache 实现细节 / KV Cache Implementation Details
# =============================================================================

print("\n" + "=" * 60)
print("KV Cache 实现细节 / Implementation Details")
print("=" * 60)

print("""
KV Cache 的关键点:

1. 缓存结构 / Cache Structure:
   - past_key_values: List[Tuple[Tensor, Tensor]]
   - 每层一个 (K, V) 元组
   - 形状: [batch, num_heads, past_seq_len, head_dim]

2. 增量更新 / Incremental Update:
   - 新 K/V 与缓存拼接
   - K = concat([past_k, new_k], dim=2)
   - V = concat([past_v, new_v], dim=2)

3. 注意事项 / Notes:
   - 需要管理缓存的创建和更新
   - 长序列时内存占用大
   - 可以结合连续批处理优化
""")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 KV Cache 的核心概念:
This example demonstrates core concepts of KV Cache:

1. 问题 / Problem:
   - 自回归生成需要重复计算
   - 序列越长，计算量越大

2. 解决方案 / Solution:
   - 缓存已计算的 K 和 V
   - 只计算新 token 的 K 和 V

3. 优势 / Advantages:
   - 显著加速推理
   - 计算量恒定（不随序列长度增长）

4. 内存代价 / Memory Cost:
   - O(L * B * H * D * S)
   - L: 层数, B: 批大小, H: 头数, D: 头维度, S: 序列长度

5. 实际应用 / Applications:
   - 所有主流 LLM 都使用 KV Cache
   - 是推理优化的基础
"""
