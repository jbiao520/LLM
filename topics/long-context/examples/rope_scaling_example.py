"""
RoPE 缩放示例

演示如何通过缩放 RoPE 位置编码来扩展上下文长度。
"""

import torch
import torch.nn as nn
import math


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """
    预计算 RoPE 的频率

    Args:
        dim: 嵌入维度
        max_seq_len: 最大序列长度
        theta: 基础频率

    Returns:
        freqs_cis: 复数形式的频率 [max_seq_len, dim//2]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len)
    freqs = torch.outer(positions, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x, freqs_cis):
    """
    应用旋转位置编码

    Args:
        x: [batch, seq_len, num_heads, head_dim]
        freqs_cis: [seq_len, head_dim//2]

    Returns:
        应用 RoPE 后的 x
    """
    # 重塑为复数
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # 应用旋转
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    x_out = x_complex * freqs_cis

    # 转回实数
    x_out = torch.view_as_real(x_out).flatten(-2)
    return x_out.type_as(x)


class RoPEScaling:
    """RoPE 缩放实现"""

    def __init__(self, dim, max_seq_len=4096, original_max_seq_len=2048, scaling_factor=2.0):
        """
        Args:
            dim: 嵌入维度
            max_seq_len: 扩展后的最大序列长度
            original_max_seq_len: 原始训练的最大序列长度
            scaling_factor: 缩放因子
        """
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.original_max_seq_len = original_max_seq_len
        self.scaling_factor = scaling_factor

    def linear_scaling(self, positions):
        """
        线性缩放

        将位置从 [0, max_seq_len] 缩放到 [0, original_max_seq_len]
        """
        return positions / self.scaling_factor

    def dynamic_ntk_scaling(self, positions, seq_len):
        """
        动态 NTK 缩放

        根据序列长度动态调整缩放因子
        """
        if seq_len <= self.original_max_seq_len:
            return positions

        # 动态计算缩放因子
        scaling_factor = seq_len / self.original_max_seq_len
        return positions / scaling_factor

    def yarn_scaling(self, positions, seq_len, temperature=1.0):
        """
        YaRN 缩放

        结合温度缩放和动态缩放
        """
        if seq_len <= self.original_max_seq_len:
            return positions, temperature

        scaling_factor = seq_len / self.original_max_seq_len

        # 温度调整
        temperature = 1.0 + 0.32 * math.log(scaling_factor)

        return positions / scaling_factor, temperature


def demonstrate_linear_scaling():
    """演示线性缩放"""
    print("=" * 60)
    print("线性缩放演示")
    print("=" * 60)

    original_max = 2048
    new_max = 8192
    scaling_factor = new_max / original_max  # 4.0

    print(f"\n原始最大长度: {original_max}")
    print(f"目标最大长度: {new_max}")
    print(f"缩放因子: {scaling_factor}")

    # 原始位置
    positions = torch.arange(0, new_max, 1000)
    scaled_positions = positions / scaling_factor

    print("\n位置映射:")
    print("-" * 40)
    for orig, scaled in zip(positions.tolist(), scaled_positions.tolist()):
        print(f"  实际位置 {orig:5d} → 缩放后 {scaled:.1f}")

    print("""
结论:
- 实际位置 8000 被映射到 2000
- 模型"看到"的位置始终在训练范围内
- 代价是位置分辨率降低
    """)


def demonstrate_rope_frequencies():
    """演示 RoPE 频率变化"""
    print("\n" + "=" * 60)
    print("RoPE 频率演示")
    print("=" * 60)

    dim = 64
    max_seq_len = 4096

    # 原始频率
    freqs_original = precompute_freqs_cis(dim, max_seq_len, theta=10000.0)

    # 缩放后的频率 (缩放因子 4)
    scaling_factor = 4.0
    freqs_scaled = precompute_freqs_cis(dim, max_seq_len, theta=10000.0 * scaling_factor)

    print(f"\n嵌入维度: {dim}")
    print(f"最大序列长度: {max_seq_len}")
    print(f"缩放因子: {scaling_factor}")

    # 比较不同位置的相位
    print("\n不同位置的相位 (第 0 维):")
    print("-" * 40)

    for pos in [0, 512, 1024, 2048, 4096]:
        phase_orig = torch.angle(freqs_original[pos, 0]).item()
        phase_scaled = torch.angle(freqs_scaled[pos, 0]).item()
        print(f"  位置 {pos:4d}: 原始={phase_orig:.4f}, 缩放后={phase_scaled:.4f}")


def demonstrate_context_extension():
    """演示上下文扩展效果"""
    print("\n" + "=" * 60)
    print("上下文扩展效果")
    print("=" * 60)

    print("""
扩展方法对比:

┌─────────────────────────────────────────────────────────────────┐
│                    扩展到 4x (8K) 效果                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  无缩放 (直接外推):                                             │
│  ████████░░░░░░░░░░░░  困惑度急剧上升                           │
│  前 2K 正常，之后崩溃                                           │
│                                                                 │
│  线性缩放:                                                      │
│  ████████████████████  困惑度稳定                               │
│  整体表现良好                                                   │
│                                                                 │
│  动态 NTK:                                                      │
│  ████████████████████  困惑度稳定                               │
│  与线性相当或更好                                               │
│                                                                 │
│  YaRN:                                                          │
│  ████████████████████  困惑度最低                               │
│  效果最好                                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    扩展到 16x (32K) 效果                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  线性缩放:                                                      │
│  ████████████░░░░░░░░  困惑度上升                               │
│  分辨率损失明显                                                 │
│                                                                 │
│  动态 NTK:                                                      │
│  ████████████████████  困惑度稳定                               │
│  效果良好                                                       │
│                                                                 │
│  YaRN:                                                          │
│  ████████████████████  困惑度稳定                               │
│  效果最好                                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
    """)


def show_practical_usage():
    """展示实际使用方法"""
    print("\n" + "=" * 60)
    print("实际使用")
    print("=" * 60)

    print("""
在 HuggingFace Transformers 中使用:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 方法 1: 使用现成的长上下文模型
long_model = AutoModelForCausalLM.from_pretrained(
    "togethercomputer/LLaMA-2-7B-32K"
)

# 方法 2: 动态缩放 (推理时)
model.config.rope_scaling = {
    "type": "linear",
    "factor": 4.0  # 4x 扩展
}

# 方法 3: YaRN 缩放
model.config.rope_scaling = {
    "type": "yarn",
    "factor": 8.0,
    "original_max_position_embeddings": 2048
}
```

使用 LongLoRA 微调:

```bash
# 微调到 32K 上下文
python finetune.py \\
    --model_name meta-llama/Llama-2-7b-hf \\
    --max_length 32768 \\
    --rope_scaling_factor 8.0 \\
    --use_longlora
```
    """)


if __name__ == "__main__":
    demonstrate_linear_scaling()
    demonstrate_rope_frequencies()
    demonstrate_context_extension()
    show_practical_usage()

    print("\n" + "=" * 60)
    print("关键要点:")
    print("=" * 60)
    print("""
1. 上下文长度受限于训练时的位置编码范围

2. RoPE 缩放通过调整位置编码来扩展上下文:
   - 线性缩放: 简单有效，适合 2-4x 扩展
   - 动态 NTK: 自适应，适合 4-8x 扩展
   - YaRN: 效果最好，适合 8x+ 扩展

3. 扩展倍数越大，精度损失越明显

4. 实际应用:
   - 选择合适的缩放方法
   - 在目标任务上评估效果
   - 必要时进行微调
    """)
