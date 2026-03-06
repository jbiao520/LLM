"""
GQA (Grouped Query Attention) 实现示例

演示 GQA 与 MHA、MQA 的区别。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """标准多头注意力 (MHA)"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # 投影
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # 合并头
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA)"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q 保持多头
        self.q_proj = nn.Linear(d_model, d_model)
        # K 和 V 只有单头
        self.k_proj = nn.Linear(d_model, self.head_dim)  # 单头
        self.v_proj = nn.Linear(d_model, self.head_dim)  # 单头
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Q: [batch, num_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V: [batch, 1, seq_len, head_dim] -> 扩展到 [batch, num_heads, seq_len, head_dim]
        k = self.k_proj(x).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        v = self.v_proj(x).unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # 合并头
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA)"""

    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.kv_head_dim = d_model // num_heads  # 保持相同的 head_dim

        # 每组包含的查询头数
        self.num_heads_per_group = num_heads // num_kv_heads

        # Q: 所有查询头
        self.q_proj = nn.Linear(d_model, d_model)
        # K, V: 只有 num_kv_heads 个头
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Q: [batch, num_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V: [batch, num_kv_heads, seq_len, head_dim]
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 扩展 K, V 到与 Q 相同的头数
        # 使用 repeat_interleave 来复制每个 KV 头到对应的查询头组
        k = k.repeat_interleave(self.num_heads_per_group, dim=1)
        v = v.repeat_interleave(self.num_heads_per_group, dim=1)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # 合并头
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


def compare_implementations():
    """比较不同注意力实现"""
    print("=" * 60)
    print("MHA vs MQA vs GQA 对比")
    print("=" * 60)

    d_model = 512
    num_heads = 8
    num_kv_heads = 2  # GQA: 8 个查询头，2 个 KV 头
    batch_size = 2
    seq_len = 16

    x = torch.randn(batch_size, seq_len, d_model)

    # MHA
    mha = MultiHeadAttention(d_model, num_heads)
    mha_params = sum(p.numel() for p in mha.parameters())

    # MQA
    mqa = MultiQueryAttention(d_model, num_heads)
    mqa_params = sum(p.numel() for p in mqa.parameters())

    # GQA
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
    gqa_params = sum(p.numel() for p in gqa.parameters())

    print(f"\n配置:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads (GQA): {num_kv_heads}")

    print(f"\n参数量:")
    print(f"  MHA: {mha_params:,}")
    print(f"  MQA: {mqa_params:,} ({mqa_params/mha_params:.1%} of MHA)")
    print(f"  GQA: {gqa_params:,} ({gqa_params/mha_params:.1%} of MHA)")

    print(f"\nKV Cache 大小 (推理时):")
    kv_per_token = 2 * d_model  # K + V
    print(f"  MHA: {num_heads * kv_per_token:,} 元素/token")
    print(f"  MQA: {1 * kv_per_token:,} 元素/token")
    print(f"  GQA: {num_kv_heads * kv_per_token:,} 元素/token")

    # 输出形状验证
    with torch.no_grad():
        mha_out = mha(x)
        mqa_out = mqa(x)
        gqa_out = gqa(x)

    print(f"\n输出形状验证:")
    print(f"  MHA: {mha_out.shape}")
    print(f"  MQA: {mqa_out.shape}")
    print(f"  GQA: {gqa_out.shape}")


def demonstrate_kv_cache():
    """演示 KV Cache 的使用"""
    print("\n" + "=" * 60)
    print("KV Cache 使用示例")
    print("=" * 60)

    print("""
在自回归生成中，可以使用 KV Cache 避免重复计算:

┌─────────────────────────────────────────────────────────────────┐
│                    无 KV Cache                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  生成第 1 个 token: 计算 Q₀, K₀, V₀, K₁, V₁, ..., Kₙ, Vₙ      │
│  生成第 2 个 token: 计算 Q₁, K₀, V₀, K₁, V₁, ..., Kₙ, Vₙ      │
│  ...                                                            │
│  生成第 n 个 token: 计算 Qₙ, K₀, V₀, K₁, V₁, ..., Kₙ, Vₙ      │
│                                                                 │
│  每次都要重新计算所有 K, V!                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    有 KV Cache                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  生成第 1 个 token: 计算 K₀, V₀, 存入 cache                     │
│  生成第 2 个 token: 计算 K₁, V₁, 追加到 cache                    │
│  ...                                                            │
│  生成第 n 个 token: 计算 Kₙ, Vₙ, 追加到 cache                    │
│                                                                 │
│  每次只计算新的 K, V，复用之前的!                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

GQA 的 KV Cache 优势:
- MHA: 存储 32 组 K, V
- GQA: 存储 8 组 K, V (节省 75%)
- MQA: 存储 1 组 K, V (节省 97%)
    """)


def show_llama_config():
    """展示 LLaMA 的 GQA 配置"""
    print("\n" + "=" * 60)
    print("LLaMA GQA 配置")
    print("=" * 60)

    configs = {
        "LLaMA 1 (7B)": {"num_heads": 32, "num_kv_heads": 32, "type": "MHA"},
        "LLaMA 2 (7B)": {"num_heads": 32, "num_kv_heads": 8, "type": "GQA"},
        "LLaMA 2 (13B)": {"num_heads": 40, "num_kv_heads": 8, "type": "GQA"},
        "LLaMA 2 (70B)": {"num_heads": 64, "num_kv_heads": 8, "type": "GQA"},
        "LLaMA 3 (8B)": {"num_heads": 32, "num_kv_heads": 8, "type": "GQA"},
        "LLaMA 3 (70B)": {"num_heads": 64, "num_kv_heads": 8, "type": "GQA"},
    }

    print(f"\n{'模型':<20} {'查询头':<10} {'KV头':<10} {'类型':<10} {'KV节省':<10}")
    print("-" * 60)

    for model, config in configs.items():
        kv_saving = 1 - config["num_kv_heads"] / config["num_heads"]
        print(f"{model:<20} {config['num_heads']:<10} {config['num_kv_heads']:<10} "
              f"{config['type']:<10} {kv_saving:.0%}")


if __name__ == "__main__":
    compare_implementations()
    demonstrate_kv_cache()
    show_llama_config()

    print("\n" + "=" * 60)
    print("关键要点:")
    print("=" * 60)
    print("""
1. MHA (Multi-Head Attention)
   - 每个头有独立的 Q, K, V
   - KV Cache 最大，但效果最好

2. MQA (Multi-Query Attention)
   - 所有头共享 K, V
   - KV Cache 最小，但效果可能下降

3. GQA (Grouped Query Attention)
   - 分组共享 K, V
   - 平衡了效率和效果
   - LLaMA 2/3 的标准选择

4. KV Cache 大小:
   - MHA: O(num_heads × seq_len)
   - MQA: O(seq_len)
   - GQA: O(num_kv_heads × seq_len)

5. 选择建议:
   - 新模型: 使用 GQA (8 组 KV 头)
   - 推理优化: GQA + Flash Attention
    """)
