"""
GELU 激活函数示例 / GELU Activation Function Example
====================================================

本示例深入展示 GELU (Gaussian Error Linear Unit) 的原理和实现。
This example demonstrates the principle and implementation of GELU.

GELU 是 Transformer 和现代 LLM 中最常用的激活函数。
GELU is the most commonly used activation function in Transformers and modern LLMs.

依赖安装 / Dependencies:
    pip install torch matplotlib numpy scipy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# =============================================================================
# 1. GELU 数学定义 / Mathematical Definition of GELU
# =============================================================================

print("=" * 60)
print("GELU 数学定义 / GELU Mathematical Definition")
print("=" * 60)

print("""
GELU (Gaussian Error Linear Unit) 的数学定义:

    GELU(x) = x * P(X ≤ x) = x * Φ(x)

其中 Φ(x) 是标准正态分布的累积分布函数 (CDF):
    Φ(x) = (1/2) * [1 + erf(x / √2)]

erf 是误差函数 (Error Function)
""")

# =============================================================================
# 2. GELU 的不同实现方式 / Different Implementations of GELU
# =============================================================================

class GELUExact(nn.Module):
    """
    精确的 GELU 实现（使用 erf）
    Exact GELU implementation (using erf)
    """
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

class GELUTanh(nn.Module):
    """
    Tanh 近似的 GELU 实现（PyTorch 默认使用此近似）
    Tanh approximation of GELU (PyTorch uses this by default)
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))

class GELUSigmoid(nn.Module):
    """
    Sigmoid 近似的 GELU（也称为 SiLU 或 Swish）
    Sigmoid approximation of GELU (also known as SiLU or Swish)
    """
    def forward(self, x):
        return x * torch.sigmoid(x)

# 对比不同实现 / Compare different implementations
print("\n" + "=" * 60)
print("不同 GELU 实现对比 / Comparing Different GELU Implementations")
print("=" * 60)

x = torch.linspace(-4, 4, 1000)

gelu_exact = GELUExact()
gelu_tanh = GELUTanh()
gelu_sigmoid = GELUSigmoid()

y_exact = gelu_exact(x)
y_tanh = gelu_tanh(x)
y_sigmoid = gelu_sigmoid(x)
y_pytorch = F.gelu(x)  # PyTorch 内置

print(f"\n各实现在关键点的值 / Values at key points:")
print(f"{'x':>6} | {'Exact':>10} | {'Tanh Approx':>12} | {'Sigmoid':>10} | {'PyTorch':>10}")
print("-" * 60)

test_points = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
for val in test_points:
    x_test = torch.tensor(val)
    print(f"{val:>6.1f} | {gelu_exact(x_test).item():>10.6f} | "
          f"{gelu_tanh(x_test).item():>12.6f} | "
          f"{gelu_sigmoid(x_test).item():>10.6f} | "
          f"{F.gelU(x_test).item():>10.6f}")

# =============================================================================
# 3. GELU vs ReLU 对比 / GELU vs ReLU Comparison
# =============================================================================

print("\n" + "=" * 60)
print("GELU vs ReLU 对比 / GELU vs ReLU Comparison")
print("=" * 60)

def relu(x):
    return np.maximum(0, x)

def gelu_numpy(x):
    return x * norm.cdf(x)

x_np = np.linspace(-4, 4, 1000)
y_relu = relu(x_np)
y_gelu = gelu_numpy(x_np)

# 计算差异 / Compute differences
diff = np.abs(y_gelu - y_relu)
max_diff = np.max(diff)
max_diff_idx = np.argmax(diff)

print(f"\n最大差异 / Maximum difference: {max_diff:.6f}")
print(f"最大差异位置 / Location of max difference: x = {x_np[max_diff_idx]:.4f}")

# 关键区别 / Key differences
print("""
关键区别 / Key Differences:

1. GELU 在负区间有非零输出（曲线平滑过渡）
   GELU has non-zero output in negative region (smooth transition)

2. ReLU 在 x=0 处不可导，GELU 处处可导
   ReLU is not differentiable at x=0, GELU is differentiable everywhere

3. GELU 是非单调的（在负区间略有下降）
   GELU is non-monotonic (slightly decreases in negative region)

4. GELU 可以解释为随机正则化的期望
   GELU can be interpreted as the expectation of stochastic regularization
""")

# =============================================================================
# 4. 可视化 / Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 4.1 激活函数对比 / Activation function comparison
ax1 = axes[0, 0]
ax1.plot(x_np, y_relu, 'b-', linewidth=2, label='ReLU')
ax1.plot(x_np, y_gelu, 'r-', linewidth=2, label='GELU')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title('GELU vs ReLU', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 4.2 导数对比 / Derivative comparison
ax2 = axes[0, 1]
# ReLU 导数
relu_deriv = np.where(x_np > 0, 1, 0)
# GELU 导数: Φ(x) + x * φ(x)
gelu_deriv = norm.cdf(x_np) + x_np * norm.pdf(x_np)

ax2.plot(x_np, relu_deriv, 'b-', linewidth=2, label="ReLU'")
ax2.plot(x_np, gelu_deriv, 'r-', linewidth=2, label="GELU'")
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel("f'(x)", fontsize=12)
ax2.set_title('Derivatives: GELU vs ReLU', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 4.3 GELU 近似对比 / GELU approximations comparison
ax3 = axes[1, 0]
ax3.plot(x.numpy(), y_exact.numpy(), 'k-', linewidth=2, label='Exact (erf)')
ax3.plot(x.numpy(), y_tanh.numpy(), 'r--', linewidth=2, label='Tanh Approx')
ax3.plot(x.numpy(), y_sigmoid.numpy(), 'g:', linewidth=2, label='Sigmoid (SiLU)')
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('GELU(x)', fontsize=12)
ax3.set_title('GELU Approximations', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4.4 近似误差 / Approximation errors
ax4 = axes[1, 1]
error_tanh = torch.abs(y_exact - y_tanh).numpy()
error_sigmoid = torch.abs(y_exact - y_sigmoid).numpy()

ax4.plot(x.numpy(), error_tanh, 'r-', linewidth=2, label='Tanh Approx Error')
ax4.plot(x.numpy(), error_sigmoid, 'g-', linewidth=2, label='Sigmoid Approx Error')
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel('Absolute Error', fontsize=12)
ax4.set_title('Approximation Errors', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/gelu_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n图像已保存至 / Image saved to: /tmp/gelu_analysis.png")

# =============================================================================
# 5. 在 Transformer 中的应用 / Application in Transformer
# =============================================================================

print("\n" + "=" * 60)
print("Transformer 中的 GELU / GELU in Transformer")
print("=" * 60)

class TransformerFFN(nn.Module):
    """
    Transformer 的前馈网络（使用 GELU 激活）
    Transformer Feed-Forward Network (using GELU activation)
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2  (原始论文使用 ReLU)
        # FFN(x) = GELU(xW1 + b1)W2 + b2   (现代实现使用 GELU)
        x = self.linear1(x)
        x = F.gelu(x)  # 使用 GELU
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 测试 / Test
d_model, d_ff = 512, 2048
ffn = TransformerFFN(d_model, d_ff)

batch_size, seq_len = 4, 128
x = torch.randn(batch_size, seq_len, d_model)
output = ffn(x)

print(f"\nTransformer FFN with GELU:")
print(f"  输入形状 / Input shape: {x.shape}")
print(f"  输出形状 / Output shape: {output.shape}")
print(f"  参数量 / Parameters: {sum(p.numel() for p in ffn.parameters()):,}")

# =============================================================================
# 6. GELU 的概率解释 / Probabilistic Interpretation of GELU
# =============================================================================

print("\n" + "=" * 60)
print("GELU 的概率解释 / Probabilistic Interpretation of GELU")
print("=" * 60)

print("""
GELU 可以从两个角度理解:

1. **随机正则化视角** / Stochastic Regularization View:
   - 以概率 Φ(x) = P(X ≤ x) 保留输入 x
   - 以概率 1 - Φ(x) 将输入置零
   - GELU(x) = E[保留后的x] = x * Φ(x)

2. **确定性近似视角** / Deterministic Approximation View:
   - 类似于 Dropout，但输入相关的
   - 对大正数几乎总是保留 (Φ(x) ≈ 1)
   - 对大负数几乎总是丢弃 (Φ(x) ≈ 0)
   - 在 0 附近平滑过渡

这使得 GELU 具有"自适应门控"的特性。
""")

# 演示概率解释 / Demonstrate probabilistic interpretation
x_demo = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
prob_keep = torch.tensor([norm.cdf(val) for val in x_demo])
gelu_vals = x_demo * prob_keep

print("x 值 / Value | 保留概率 / Keep Prob | GELU(x)")
print("-" * 50)
for i, val in enumerate(x_demo):
    print(f"{val:>11.1f} | {prob_keep[i]:>18.4f} | {gelu_vals[i]:>7.4f}")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 GELU 的核心概念:
This example demonstrates core concepts of GELU:

1. 数学定义: GELU(x) = x * Φ(x)，其中 Φ 是标准正态 CDF
   Mathematical definition

2. 特点:
   - 处处可导
   - 非单调
   - 负区间有非零输出

3. 实现:
   - 精确实现使用 erf 函数
   - 常用 tanh 近似（PyTorch 默认）
   - sigmoid 近似等价于 SiLU/Swish

4. 为什么 Transformer 使用 GELU:
   - 平滑性带来更稳定的训练
   - 在 NLP 任务中表现更好
   - BERT、GPT、LLaMA 等都使用 GELU

5. 概率解释:
   - 可以理解为随机正则化的期望
   - 输入相关的"门控"机制
"""
