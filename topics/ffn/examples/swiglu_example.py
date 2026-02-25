"""
SwiGLU 示例代码 / SwiGLU Example Code
=====================================

本示例展示 SwiGLU 激活函数及其在前馈网络中的应用。
This example demonstrates SwiGLU activation and its application in FFN.

SwiGLU 是 LLaMA 等现代 LLM 使用的 FFN 激活函数。
SwiGLU is the FFN activation used in modern LLMs like LLaMA.

依赖安装 / Dependencies:
    pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. GLU (Gated Linear Unit) 基础 / GLU Basics
# =============================================================================

print("=" * 60)
print("GLU (Gated Linear Unit) 基础 / GLU Basics")
print("=" * 60)

print("""
GLU 的数学定义:
GLU(x) = (xW) ⊙ σ(xV)

其中:
- W 和 V 是两个独立的线性变换
- σ 是 sigmoid 函数
- ⊙ 是逐元素乘法

门控机制允许模型选择性地传递信息。
""")

class GLU(nn.Module):
    """基础 GLU 实现"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff, bias=False)
        self.V = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.W(x) * torch.sigmoid(self.V(x))

# =============================================================================
# 2. SwiGLU 定义 / SwiGLU Definition
# =============================================================================

print("\n" + "=" * 60)
print("SwiGLU 定义 / SwiGLU Definition")
print("=" * 60)

print("""
SwiGLU 的数学定义:
SwiGLU(x) = Swish(xW) ⊙ (xV)
          = (xW ⊙ σ(βxW)) ⊙ (xV)  (当 β=1 时)

其中:
- Swish(x) = x ⊙ σ(x) = x * sigmoid(x)
- 也称为 SiLU (Sigmoid Linear Unit)

SwiGLU 的特点:
1. 结合了 Swish 的平滑性和 GLU 的门控机制
2. 在 LLM 中表现优于标准 FFN
3. 被 LLaMA、PaLM 等模型采用
""")

# =============================================================================
# 3. SwiGLU 实现 / SwiGLU Implementation
# =============================================================================

class SwiGLUFFN(nn.Module):
    """
    使用 SwiGLU 的 FFN 层 / FFN Layer with SwiGLU

    参数设置（为保持与标准 FFN 相同参数量）:
    d_ff = 2/3 * 4 * d_model ≈ 2.67 * d_model

    或简化为: d_ff = 8/3 * d_model
    """
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()

        # 如果未指定 d_ff，使用 8/3 * d_model 保持参数量相近
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)

        self.d_model = d_model
        self.d_ff = d_ff

        # 三个线性层（相比标准 FFN 多一个）
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # 主变换
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # 输出投影
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # 门控分支

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        SwiGLU(x) = Swish(xW1) ⊙ (xW3) @ W2
                  = (silu(xW1) * xW3) @ W2
        """
        # 门控机制: Swish(xW1) * xW3
        hidden = F.silu(self.w1(x)) * self.w3(x)
        # Dropout
        hidden = self.dropout(hidden)
        # 输出投影
        output = self.w2(hidden)
        return output

# =============================================================================
# 4. 测试 SwiGLU / Test SwiGLU
# =============================================================================

print("\n" + "=" * 60)
print("SwiGLU 测试 / SwiGLU Test")
print("=" * 60)

torch.manual_seed(42)

d_model = 768
d_ff_standard = 3072  # 标准 FFN 的 d_ff

# 创建模型
standard_ffn = nn.Sequential(
    nn.Linear(d_model, d_ff_standard),
    nn.GELU(),
    nn.Linear(d_ff_standard, d_model)
)

swiglu_ffn = SwiGLUFFN(d_model)  # 自动计算 d_ff

# 创建输入
x = torch.randn(2, 128, d_model)

# 前向传播
output_standard = standard_ffn(x)
output_swiglu = swiglu_ffn(x)

print(f"\n输入形状 / Input shape: {x.shape}")
print(f"标准 FFN 输出形状 / Standard FFN output shape: {output_standard.shape}")
print(f"SwiGLU FFN 输出形状 / SwiGLU FFN output shape: {output_swiglu.shape}")

# 参数量对比
params_standard = sum(p.numel() for p in standard_ffn.parameters())
params_swiglu = sum(p.numel() for p in swiglu_ffn.parameters())

print(f"\n参数量对比 / Parameter Comparison:")
print(f"  标准 FFN (d_ff=3072): {params_standard:,}")
print(f"  SwiGLU FFN (d_ff={swiglu_ffn.d_ff}): {params_swiglu:,}")
print(f"  比例 / Ratio: {params_swiglu/params_standard:.2f}x")

# =============================================================================
# 5. 不同 GLU 变体对比 / Compare Different GLU Variants
# =============================================================================

print("\n" + "=" * 60)
print("GLU 变体对比 / GLU Variants Comparison")
print("=" * 60)

class ReGLUFFN(nn.Module):
    """ReGLU: ReLU + GLU"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)) * self.w3(x))

class GeGLUFFN(nn.Module):
    """GeGLU: GELU + GLU"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)) * self.w3(x))

# 创建不同变体
d_ff_variants = int(8/3 * d_model)
variants = {
    'ReGLU': ReGLUFFN(d_model, d_ff_variants),
    'GeGLU': GeGLUFFN(d_model, d_ff_variants),
    'SwiGLU': SwiGLUFFN(d_model)
}

# 对比输出
print(f"\n各变体输出统计 (相同输入):")
for name, model in variants.items():
    with torch.no_grad():
        out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f"  {name}: mean={out.mean().item():.4f}, std={out.std().item():.4f}, params={params:,}")

# =============================================================================
# 6. 可视化门控机制 / Visualize Gating Mechanism
# =============================================================================

print("\n" + "=" * 60)
print("门控机制可视化 / Gating Mechanism Visualization")
print("=" * 60)

# 可视化 Swish 和 Sigmoid 函数
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

x_vis = torch.linspace(-4, 4, 100)

# 6.1 Sigmoid
ax1 = axes[0]
ax1.plot(x_vis.numpy(), torch.sigmoid(x_vis).numpy(), 'b-', linewidth=2, label='Sigmoid')
ax1.set_title('Sigmoid Function', fontsize=12)
ax1.set_xlabel('x')
ax1.set_ylabel('σ(x)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 6.2 Swish/SiLU
ax2 = axes[1]
ax2.plot(x_vis.numpy(), F.silu(x_vis).numpy(), 'r-', linewidth=2, label='Swish/SiLU')
ax2.set_title('Swish/SiLU Function', fontsize=12)
ax2.set_xlabel('x')
ax2.set_ylabel('x·σ(x)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 6.3 门控效果
ax3 = axes[2]
# 模拟门控: value * gate
value = torch.sin(x_vis * 2)  # 值函数
gate = torch.sigmoid(x_vis)   # 门控函数
gated = value * gate          # 门控后

ax3.plot(x_vis.numpy(), value.numpy(), 'g--', linewidth=1.5, label='Value', alpha=0.7)
ax3.plot(x_vis.numpy(), gate.numpy(), 'b--', linewidth=1.5, label='Gate', alpha=0.7)
ax3.plot(x_vis.numpy(), gated.numpy(), 'r-', linewidth=2, label='Gated Output')
ax3.set_title('Gating Effect', fontsize=12)
ax3.set_xlabel('x')
ax3.set_ylabel('Output')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.savefig('/tmp/swiglu_gating.png', dpi=150, bbox_inches='tight')
print(f"\n门控机制图像已保存至 / Image saved to: /tmp/swiglu_gating.png")

# =============================================================================
# 7. LLaMA 风格的 FFN / LLaMA-style FFN
# =============================================================================

print("\n" + "=" * 60)
print("LLaMA 风格 FFN / LLaMA-style FFN")
print("=" * 60)

class LLaMAFFN(nn.Module):
    """
    LLaMA 的 FFN 实现 / LLaMA's FFN Implementation

    特点:
    1. 使用 SwiGLU 激活
    2. 无偏置项
    3. d_ff = 2/3 * 4 * d_model (约化以保持参数量)
    """
    def __init__(self, dim, hidden_dim=None, multiple_of=256, dropout=0.0):
        super().__init__()

        # LLaMA 的 hidden_dim 计算方式
        if hidden_dim is None:
            hidden_dim = int(2 * 4 * dim / 3)
            # 向上取整到 multiple_of 的倍数
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# LLaMA-7B 的配置
llama_ffn = LLaMAFFN(dim=4096)  # LLaMA-7B 的 d_model

print(f"\nLLaMA-7B FFN 配置:")
print(f"  d_model: 4096")
print(f"  hidden_dim: {llama_ffn.w1.out_features}")
print(f"  参数量: {sum(p.numel() for p in llama_ffn.parameters()):,}")

# 测试
x_llama = torch.randn(1, 100, 4096)
out_llama = llama_ffn(x_llama)
print(f"  输入形状: {x_llama.shape}")
print(f"  输出形状: {out_llama.shape}")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 SwiGLU 的核心概念:
This example demonstrates core concepts of SwiGLU:

1. GLU 基础 / GLU Basics:
   GLU(x) = (xW) ⊙ σ(xV)
   门控机制选择性传递信息

2. SwiGLU 公式 / SwiGLU Formula:
   SwiGLU(x) = Swish(xW1) ⊙ (xW3) @ W2
             = (silu(xW1) * xW3) @ W2

3. 相比标准 FFN:
   - 多一个线性变换（3个 vs 2个）
   - 使用门控机制
   - 调整 d_ff 保持参数量相近

4. GLU 变体 / GLU Variants:
   - ReGLU: ReLU + GLU
   - GeGLU: GELU + GLU
   - SwiGLU: Swish + GLU (LLaMA 使用)

5. 实际应用:
   - LLaMA
   - PaLM
   - 其他现代 LLM
"""
