"""
激活函数对比示例 / Activation Functions Comparison Example
=========================================================

本示例展示各种激活函数的特性和对比。
This example demonstrates the properties and comparison of various activation functions.

依赖安装 / Dependencies:
    pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. 定义各种激活函数 / Define Various Activation Functions
# =============================================================================

def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU: max(alpha*x, x)"""
    return np.where(x > 0, x, alpha * x)

def sigmoid(x):
    """Sigmoid: 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh_func(x):
    """Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    return np.tanh(x)

def gelu(x):
    """GELU: x * Phi(x) (使用 tanh 近似)"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def swish(x, beta=1.0):
    """Swish: x * sigmoid(beta * x)"""
    return x * sigmoid(beta * x)

def elu(x, alpha=1.0):
    """ELU: x if x > 0 else alpha * (exp(x) - 1)"""
    return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))

def softplus(x):
    """Softplus: log(1 + exp(x))"""
    return np.log1p(np.exp(np.clip(x, -500, 500)))

# =============================================================================
# 2. 可视化激活函数 / Visualize Activation Functions
# =============================================================================

print("=" * 60)
print("激活函数可视化 / Activation Functions Visualization")
print("=" * 60)

# 创建输入范围 / Create input range
x = np.linspace(-5, 5, 1000)

# 计算各激活函数输出 / Compute outputs for each activation
activations = {
    'ReLU': relu(x),
    'Leaky ReLU': leaky_relu(x),
    'Sigmoid': sigmoid(x),
    'Tanh': tanh_func(x),
    'GELU': gelu(x),
    'Swish': swish(x),
    'ELU': elu(x),
    'Softplus': softplus(x),
}

# 绘制激活函数 / Plot activation functions
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, (name, y) in enumerate(activations.items()):
    ax = axes[idx]
    ax.plot(x, y, linewidth=2, label=name)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_title(name, fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.legend()

plt.tight_layout()
plt.savefig('/tmp/activation_functions.png', dpi=150, bbox_inches='tight')
print(f"\n激活函数图像已保存至 / Image saved to: /tmp/activation_functions.png")

# =============================================================================
# 3. 可视化导数 / Visualize Derivatives
# =============================================================================

print("\n" + "=" * 60)
print("激活函数导数 / Activation Function Derivatives")
print("=" * 60)

# 计算导数（使用数值微分）/ Compute derivatives (numerical)
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

derivatives = {
    'ReLU': numerical_derivative(relu, x),
    'Leaky ReLU': numerical_derivative(leaky_relu, x),
    'Sigmoid': numerical_derivative(sigmoid, x),
    'Tanh': numerical_derivative(tanh_func, x),
    'GELU': numerical_derivative(gelu, x),
    'Swish': numerical_derivative(swish, x),
}

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, (name, dy) in enumerate(derivatives.items()):
    ax = axes[idx]
    ax.plot(x, dy, linewidth=2, color='orange', label=f"{name}'")
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax.set_title(f"{name} Derivative", fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel("f'(x)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)
    ax.legend()

plt.tight_layout()
plt.savefig('/tmp/activation_derivatives.png', dpi=150, bbox_inches='tight')
print(f"导数图像已保存至 / Derivative image saved to: /tmp/activation_derivatives.png")

# =============================================================================
# 4. 梯度消失问题演示 / Vanishing Gradient Demonstration
# =============================================================================

print("\n" + "=" * 60)
print("梯度消失问题演示 / Vanishing Gradient Demonstration")
print("=" * 60)

# 模拟深层网络的梯度传播 / Simulate gradient propagation in deep networks
def simulate_gradient_flow(activation_fn, num_layers=20, initial_grad=1.0):
    """
    模拟梯度在深层网络中的传播
    Simulate gradient propagation through deep layers
    """
    gradients = [initial_grad]
    grad = initial_grad

    for _ in range(num_layers):
        # 假设每层的激活导数在 [-1, 1] 范围内的某个值
        # Sigmoid 和 Tanh 在饱和区的导数很小
        if activation_fn == 'sigmoid':
            # Sigmoid 最大导数为 0.25
            grad *= 0.25
        elif activation_fn == 'tanh':
            # Tanh 最大导数为 1，但在饱和区很小
            grad *= 0.5
        elif activation_fn == 'relu':
            # ReLU 导数为 0 或 1
            grad *= 1.0  # 假设不进入负区
        elif activation_fn == 'gelu':
            # GELU 导数在正区接近 1
            grad *= 0.95
        gradients.append(grad)

    return gradients

plt.figure(figsize=(10, 6))

for act_name in ['sigmoid', 'tanh', 'relu', 'gelu']:
    grads = simulate_gradient_flow(act_name)
    plt.plot(range(len(grads)), grads, linewidth=2, marker='o', label=act_name.upper())

plt.xlabel('Layer / 层', fontsize=12)
plt.ylabel('Gradient Magnitude / 梯度大小', fontsize=12)
plt.title('Gradient Flow Through Deep Layers / 深层网络中的梯度流', fontsize=14)
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/gradient_flow.png', dpi=150, bbox_inches='tight')
print(f"梯度流图像已保存至 / Gradient flow image saved to: /tmp/gradient_flow.png")

# =============================================================================
# 5. PyTorch 实现对比 / PyTorch Implementation Comparison
# =============================================================================

print("\n" + "=" * 60)
print("PyTorch 实现对比 / PyTorch Implementation Comparison")
print("=" * 60)

x_torch = torch.linspace(-5, 5, 1000)

# PyTorch 内置激活函数 / PyTorch built-in activation functions
pytorch_activations = {
    'ReLU': F.relu(x_torch),
    'Leaky ReLU': F.leaky_relu(x_torch, 0.01),
    'Sigmoid': torch.sigmoid(x_torch),
    'Tanh': torch.tanh(x_torch),
    'GELU': F.gelu(x_torch),
    'SiLU (Swish)': F.silu(x_torch),
    'ELU': F.elu(x_torch),
    'Softplus': F.softplus(x_torch),
}

print("\n各激活函数在 x=0 处的值 / Values at x=0:")
for name, y in pytorch_activations.items():
    idx = len(x_torch) // 2  # x=0 的索引
    print(f"  {name}: {y[idx].item():.4f}")

print("\n各激活函数在 x=2 处的值 / Values at x=2:")
for name, y in pytorch_activations.items():
    idx = int(0.7 * len(x_torch))  # x≈2 的索引
    print(f"  {name}: {y[idx].item():.4f}")

# =============================================================================
# 6. Softmax 详解 / Softmax Detailed Explanation
# =============================================================================

print("\n" + "=" * 60)
print("Softmax 详解 / Softmax Detailed Explanation")
print("=" * 60)

def softmax(x, temperature=1.0):
    """
    Softmax 函数（带温度参数）
    Softmax function with temperature parameter
    """
    x_scaled = x / temperature
    e_x = np.exp(x_scaled - np.max(x_scaled))  # 数值稳定性
    return e_x / e_x.sum()

# 演示温度参数效果 / Demonstrate temperature parameter effect
logits = np.array([2.0, 1.0, 0.1])

print("\n原始 logits / Original logits:", logits)
print("\n不同温度下的 Softmax 输出 / Softmax outputs with different temperatures:")

temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
for T in temperatures:
    probs = softmax(logits, T)
    print(f"  T={T:4.1f}: {probs}")

# 可视化温度效果 / Visualize temperature effect
fig, ax = plt.subplots(figsize=(10, 6))

x_bar = np.arange(len(logits))
width = 0.12

for i, T in enumerate(temperatures):
    probs = softmax(logits, T)
    ax.bar(x_bar + i * width, probs, width, label=f'T={T}')

ax.set_xlabel('Class / 类别', fontsize=12)
ax.set_ylabel('Probability / 概率', fontsize=12)
ax.set_title('Softmax Temperature Effect / Softmax 温度参数效果', fontsize=14)
ax.set_xticks(x_bar + width * 2.5)
ax.set_xticklabels(['Class 0 (logit=2.0)', 'Class 1 (logit=1.0)', 'Class 2 (logit=0.1)'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/tmp/softmax_temperature.png', dpi=150, bbox_inches='tight')
print(f"\nSoftmax 温度图像已保存至 / Softmax temperature image saved to: /tmp/softmax_temperature.png")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了各种激活函数的特性:
This example demonstrates the characteristics of various activation functions:

1. ReLU: 简单高效，但有 Dead ReLU 问题
   Simple and efficient, but has Dead ReLU problem

2. Leaky ReLU: 解决 Dead ReLU，允许小梯度流过负区
   Solves Dead ReLU, allows small gradient flow in negative region

3. Sigmoid: 输出 [0,1]，适合二分类，但有梯度消失问题
   Output [0,1], suitable for binary classification, but has vanishing gradient

4. Tanh: 输出 [-1,1]，零中心，但仍有梯度消失
   Output [-1,1], zero-centered, but still has vanishing gradient

5. GELU: 平滑、非单调，Transformer 首选
   Smooth, non-monotonic, preferred in Transformers

6. Swish: 自门控，在深层网络表现优秀
   Self-gated, performs well in deep networks

7. Softmax: 多分类输出，温度参数控制分布平滑度
   Multi-class output, temperature controls distribution smoothness
"""
