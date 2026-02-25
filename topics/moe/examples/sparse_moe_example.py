"""
稀疏 MoE 层示例代码 / Sparse MoE Layer Example Code
===================================================

本示例展示完整的稀疏混合专家模型（Sparse MoE）层的实现。
This example demonstrates a complete implementation of a Sparse MoE layer.

依赖安装 / Dependencies:
    pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. 专家网络 / Expert Network
# =============================================================================

class Expert(nn.Module):
    """
    单个专家网络（前馈网络）/ Single Expert Network (Feed-Forward)

    在 Transformer MoE 中，专家通常替换 FFN 层
    In Transformer MoE, experts typically replace the FFN layer
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # FFN with GELU activation
        return self.w2(self.dropout(F.gelu(self.w1(x))))

# =============================================================================
# 2. 完整的 Sparse MoE 层 / Complete Sparse MoE Layer
# =============================================================================

class SparseMoE(nn.Module):
    """
    稀疏混合专家层 / Sparse Mixture of Experts Layer

    特点 / Features:
    1. Top-K 路由 / Top-K routing
    2. 负载均衡损失 / Load balancing loss
    3. 专家容量限制 / Expert capacity limit
    """
    def __init__(self, d_model, d_ff, num_experts, top_k=2, dropout=0.1,
                 capacity_factor=1.25, aux_loss_weight=0.01):
        super().__init__()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight

        # 专家网络 / Expert networks
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])

        # 路由器 / Router
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        参数 / Args:
            x: [batch, seq_len, d_model]

        返回 / Returns:
            output: [batch, seq_len, d_model]
            aux_loss: 负载均衡辅助损失
        """
        batch_size, seq_len, d_model = x.shape
        total_tokens = batch_size * seq_len

        # 1. 计算路由分数 / Compute routing scores
        logits = self.gate(x)  # [batch, seq_len, num_experts]

        # 2. Top-K 选择 / Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # [batch, seq_len, top_k]

        # 3. 初始化输出 / Initialize output
        output = torch.zeros_like(x)

        # 4. 分发到专家 / Dispatch to experts
        # 为每个专家收集需要处理的 token
        for k in range(self.top_k):
            for expert_idx in range(self.num_experts):
                # 找出在这个位置选择该专家的所有 token
                mask = (top_k_indices[:, :, k] == expert_idx)  # [batch, seq_len]

                if mask.sum() == 0:
                    continue

                # 提取这些 token
                selected_tokens = x[mask]  # [num_selected, d_model]
                gate_weights = top_k_gates[:, :, k][mask]  # [num_selected]

                # 通过专家处理 / Process through expert
                expert_output = self.experts[expert_idx](selected_tokens)

                # 加权并累加到输出 / Weight and accumulate to output
                output[mask] += gate_weights.unsqueeze(-1) * expert_output

        # 5. 计算辅助损失 / Compute auxiliary loss
        aux_loss = self._compute_aux_loss(top_k_gates, top_k_indices)

        return output, aux_loss

    def _compute_aux_loss(self, gates, indices):
        """计算负载均衡损失 / Compute load balancing loss"""
        # f_i: 分配给专家 i 的 token 比例
        f = torch.zeros(self.num_experts, device=gates.device)
        for k in range(self.top_k):
            for idx in indices[:, :, k].flatten():
                f[idx] += 1
        f = f / (gates.shape[0] * gates.shape[1] * self.top_k)

        # P_i: 专家 i 的平均路由概率
        # 重建完整的 gates 矩阵
        full_gates = torch.zeros(gates.shape[0], gates.shape[1],
                                  self.num_experts, device=gates.device)
        for k in range(self.top_k):
            full_gates.scatter_(-1, indices[:, :, k:k+1], gates[:, :, k:k+1])
        P = full_gates.mean(dim=(0, 1))

        # 辅助损失
        aux_loss = self.num_experts * torch.sum(f * P)

        return self.aux_loss_weight * aux_loss

# =============================================================================
# 3. 测试 Sparse MoE 层 / Test Sparse MoE Layer
# =============================================================================

print("=" * 60)
print("Sparse MoE 层测试 / Sparse MoE Layer Test")
print("=" * 60)

torch.manual_seed(42)

d_model = 128
d_ff = 512
num_experts = 8
top_k = 2
batch_size, seq_len = 4, 16

# 创建 MoE 层 / Create MoE layer
moe = SparseMoE(d_model, d_ff, num_experts, top_k)

# 创建输入 / Create input
x = torch.randn(batch_size, seq_len, d_model)

# 前向传播 / Forward pass
output, aux_loss = moe(x)

print(f"\n配置 / Configuration:")
print(f"  d_model: {d_model}")
print(f"  d_ff: {d_ff}")
print(f"  num_experts: {num_experts}")
print(f"  top_k: {top_k}")

print(f"\n输入形状 / Input shape: {x.shape}")
print(f"输出形状 / Output shape: {output.shape}")
print(f"辅助损失 / Aux loss: {aux_loss.item():.6f}")

# 参数统计 / Parameter statistics
total_params = sum(p.numel() for p in moe.parameters())
expert_params = sum(p.numel() for p in moe.experts[0].parameters())
router_params = sum(p.numel() for p in moe.gate.parameters())

print(f"\n参数统计 / Parameter statistics:")
print(f"  总参数 / Total: {total_params:,}")
print(f"  每个 expert / Per expert: {expert_params:,}")
print(f"  路由器 / Router: {router_params:,}")
print(f"  等效稠密 FFN 参数 / Equivalent dense FFN: {d_model * d_ff * 2:,}")

# =============================================================================
# 4. MoE vs 稠密层对比 / MoE vs Dense Layer Comparison
# =============================================================================

print("\n" + "=" * 60)
print("MoE vs 稠密层对比 / MoE vs Dense Layer Comparison")
print("=" * 60)

class DenseFFN(nn.Module):
    """标准稠密 FFN 层 / Standard dense FFN layer"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.gelu(self.w1(x))))

# 对比参数量 / Compare parameters
dense_ffn = DenseFFN(d_model, d_ff)
moe_layer = SparseMoE(d_model, d_ff, num_experts, top_k)

dense_params = sum(p.numel() for p in dense_ffn.parameters())
moe_params = sum(p.numel() for p in moe_layer.parameters())
active_params = expert_params * top_k + router_params  # 每次激活的参数

print(f"\n参数量对比 / Parameter comparison:")
print(f"  稠密 FFN / Dense FFN: {dense_params:,}")
print(f"  MoE 总参数 / MoE total: {moe_params:,} ({moe_params/dense_params:.1f}x)")
print(f"  MoE 激活参数 / MoE active: {active_params:,} ({active_params/dense_params:.1f}x)")

# =============================================================================
# 5. 可视化专家使用情况 / Visualize Expert Usage
# =============================================================================

print("\n" + "=" * 60)
print("专家使用可视化 / Expert Usage Visualization")
print("=" * 60)

# 多次前向传播，统计专家使用 / Multiple forward passes, track expert usage
expert_usage = torch.zeros(num_experts)

for _ in range(10):
    x_test = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        _, _, logits = moe.gate(x_test), None, moe.gate(x_test)
        logits = moe.gate(x_test)
        _, indices = torch.topk(logits, top_k, dim=-1)

        for idx in indices.flatten():
            expert_usage[idx] += 1

# 可视化 / Visualize
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.bar(range(num_experts), expert_usage.numpy())
plt.xlabel('Expert ID')
plt.ylabel('Usage Count')
plt.title('Expert Usage Distribution')

# 绘制激活模式 / Plot activation pattern
plt.subplot(1, 2, 2)
x_vis = torch.randn(1, 20, d_model)
with torch.no_grad():
    logits_vis = moe.gate(x_vis)
    _, indices_vis = torch.topk(logits_vis, top_k, dim=-1)

# 创建热力图 / Create heatmap
activation_map = torch.zeros(20, num_experts)
for t in range(20):
    for k in range(top_k):
        activation_map[t, indices_vis[0, t, k]] = 1

plt.imshow(activation_map.numpy(), cmap='Blues', aspect='auto')
plt.xlabel('Expert ID')
plt.ylabel('Token Position')
plt.title('Expert Activation Pattern')

plt.tight_layout()
plt.savefig('/tmp/moe_expert_usage.png', dpi=150, bbox_inches='tight')
print(f"\n专家使用图像已保存至 / Expert usage image saved to: /tmp/moe_expert_usage.png")

# =============================================================================
# 6. 在 Transformer 中使用 MoE / Using MoE in Transformer
# =============================================================================

print("\n" + "=" * 60)
print("在 Transformer 中使用 MoE / Using MoE in Transformer")
print("=" * 60)

class TransformerBlockWithMoE(nn.Module):
    """
    使用 MoE 的 Transformer Block
    Transformer Block with MoE
    """
    def __init__(self, d_model, num_heads, d_ff, num_experts, top_k=2, dropout=0.1):
        super().__init__()

        # Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        # MoE FFN
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = SparseMoE(d_model, d_ff, num_experts, top_k, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention with Pre-LN
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = residual + self.dropout(x)

        # MoE FFN with Pre-LN
        residual = x
        x = self.norm2(x)
        x, aux_loss = self.moe(x)
        x = residual + self.dropout(x)

        return x, aux_loss

# 创建并测试 / Create and test
block = TransformerBlockWithMoE(
    d_model=256, num_heads=4, d_ff=1024, num_experts=8, top_k=2
)

x_test = torch.randn(2, 32, 256)
output, aux_loss = block(x_test)

print(f"\nTransformer Block with MoE:")
print(f"  输入形状 / Input shape: {x_test.shape}")
print(f"  输出形状 / Output shape: {output.shape}")
print(f"  辅助损失 / Aux loss: {aux_loss.item():.6f}")
print(f"  参数量 / Parameters: {sum(p.numel() for p in block.parameters()):,}")

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 Sparse MoE 层的完整实现:
This example demonstrates a complete Sparse MoE layer:

1. 组件 / Components:
   - Expert: 单个专家网络（通常是 FFN）
   - Router: Top-K 路由器
   - Load Balancing: 负载均衡机制

2. 优势 / Advantages:
   - 参数量大，但激活参数少
   - 可以增加模型容量而不增加计算量
   - 适合大规模模型

3. 训练考虑 / Training considerations:
   - 需要辅助损失保证负载均衡
   - 路由器需要单独的学习率设置
   - 可能需要容量限制防止过载

4. 实际应用 / Real-world applications:
   - Mixtral 8x7B
   - GPT-4 (据报道)
   - DeepSeek-MoE
   - Switch Transformer
"""
