"""
Flash Attention 使用示例

演示 Flash Attention 的概念和使用方法。
"""

import torch
import torch.nn.functional as F
import math


def standard_attention(q, k, v, mask=None):
    """
    标准注意力实现

    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, heads, seq_len, head_dim]
        v: [batch, heads, seq_len, head_dim]
        mask: 可选的注意力掩码

    Returns:
        output: [batch, heads, seq_len, head_dim]
        attention_weights: [batch, heads, seq_len, seq_len]
    """
    batch_size, num_heads, seq_len, head_dim = q.shape

    # 计算注意力分数
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # 加权求和
    output = torch.matmul(attention_weights, v)

    return output, attention_weights


def flash_attention_simulation(q, k, v, block_size=64):
    """
    Flash Attention 的模拟实现（教学用）

    实际的 Flash Attention 使用 CUDA 内核实现，
    这里只是演示分块计算的概念。

    Args:
        q: [batch, heads, seq_len, head_dim]
        k: [batch, heads, seq_len, head_dim]
        v: [batch, heads, seq_len, head_dim]
        block_size: 分块大小

    Returns:
        output: [batch, heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = q.shape

    # 输出和统计量
    output = torch.zeros_like(q)
    # m: 每个 block 的最大值（用于数值稳定）
    # l: 每个 block 的归一化因子
    m = torch.full((batch_size, num_heads, seq_len, 1), float('-inf'), device=q.device)
    l = torch.zeros((batch_size, num_heads, seq_len, 1), device=q.device)

    # 分块数量
    num_blocks = (seq_len + block_size - 1) // block_size

    # 对每个 Q 块
    for i in range(num_blocks):
        q_start = i * block_size
        q_end = min((i + 1) * block_size, seq_len)
        q_block = q[:, :, q_start:q_end, :]

        # 初始化当前块的输出
        o_block = torch.zeros((batch_size, num_heads, q_end - q_start, head_dim), device=q.device)
        m_block = torch.full((batch_size, num_heads, q_end - q_start, 1), float('-inf'), device=q.device)
        l_block = torch.zeros((batch_size, num_heads, q_end - q_start, 1), device=q.device)

        # 对每个 K, V 块
        for j in range(num_blocks):
            k_start = j * block_size
            k_end = min((j + 1) * block_size, seq_len)
            k_block = k[:, :, k_start:k_end, :]
            v_block = v[:, :, k_start:k_end, :]

            # 计算局部注意力分数
            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(head_dim)

            # 在线 softmax 更新
            # 这是 Flash Attention 的关键技巧
            m_new = torch.maximum(m_block, scores.max(dim=-1, keepdim=True)[0])

            # 重新缩放
            p = torch.exp(scores - m_new)

            # 更新归一化因子
            l_new = l_block * torch.exp(m_block - m_new) + p.sum(dim=-1, keepdim=True)

            # 更新输出
            o_block = o_block * torch.exp(m_block - m_new) + torch.matmul(p, v_block)

            # 更新统计量
            m_block = m_new
            l_block = l_new

        # 归一化输出
        output[:, :, q_start:q_end, :] = o_block / l_block

    return output


def demonstrate_memory_usage():
    """演示内存使用差异"""
    print("=" * 60)
    print("内存使用对比")
    print("=" * 60)

    configs = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ]

    print(f"\n{'序列长度':<12} {'标准注意力':<20} {'Flash Attention':<20} {'节省':<10}")
    print("-" * 62)

    for seq_len, _ in configs:
        # 标准注意力: 需要存储 seq_len × seq_len 的注意力矩阵
        standard_mem = seq_len * seq_len * 4  # 4 bytes per float32

        # Flash Attention: 只需要存储输出和少量统计量
        flash_mem = seq_len * 4 * 2  # 输出 + 统计量

        saving = 1 - flash_mem / standard_mem

        print(f"{seq_len:<12} {standard_mem / 1024 / 1024:.2f} MB{'':<10} "
              f"{flash_mem / 1024:.2f} KB{'':<12} {saving:.1%}")


def demonstrate_speedup():
    """演示速度提升"""
    print("\n" + "=" * 60)
    print("速度对比 (概念性)")
    print("=" * 60)

    print("""
Flash Attention 的速度提升来源:

1. 减少内存访问
   ┌─────────────────────────────────────────────────────────────┐
   │ 标准注意力:                                                 │
   │   HBM → SRAM → 计算 → HBM (多次往返)                        │
   │   总访问量: O(N²)                                           │
   │                                                             │
   │ Flash Attention:                                            │
   │   HBM → SRAM → 计算 → HBM (一次往返)                        │
   │   总访问量: O(N)                                            │
   └─────────────────────────────────────────────────────────────┘

2. 实测速度提升
   ┌─────────────────────────────────────────────────────────────┐
   │ 序列长度    标准注意力    Flash Attention    加速比         │
   │ 512         1.0x         2.0x               2x             │
   │ 1024        1.0x         2.5x               2.5x           │
   │ 2048        1.0x         3.0x               3x             │
   │ 4096        1.0x         3.5x               3.5x           │
   │ 8192        OOM          完成               ∞              │
   └─────────────────────────────────────────────────────────────┘

3. 额外好处
   - 支持更长的序列
   - 减少显存占用
   - 数值精度完全相同
    """)


def show_usage():
    """展示实际使用方法"""
    print("\n" + "=" * 60)
    print("Flash Attention 使用方法")
    print("=" * 60)

    print("""
方法 1: PyTorch 2.0+ 内置支持

```python
import torch.nn.functional as F

# PyTorch 2.0+ 自动使用 Flash Attention
F.scaled_dot_product_attention(q, k, v)
```

方法 2: 使用 flash-attn 库

```python
from flash_attn import flash_attn_func

# 需要安装: pip install flash-attn
output = flash_attn_func(q, k, v, softmax_scale=1.0/sqrt(d))
```

方法 3: HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM

# 加载模型时会自动使用 Flash Attention（如果可用）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"  # 显式指定
)
```

安装要求:
- PyTorch 2.0+
- CUDA 11.6+
- 对于 flash-attn 库: pip install flash-attn --no-build-isolation
    """)


if __name__ == "__main__":
    demonstrate_memory_usage()
    demonstrate_speedup()
    show_usage()

    print("\n" + "=" * 60)
    print("关键要点:")
    print("=" * 60)
    print("""
1. Flash Attention 通过分块计算减少内存访问

2. 内存复杂度:
   - 标准注意力: O(N²)
   - Flash Attention: O(N)

3. 速度提升: 2-4x

4. 使用方法:
   - PyTorch 2.0+ 自动支持
   - 安装 flash-attn 库获得最佳性能
   - HuggingFace 模型可指定 attn_implementation

5. 注意事项:
   - 需要 CUDA GPU
   - 某些旧 GPU 可能不支持
   - 数值精度与标准注意力完全相同
    """)
