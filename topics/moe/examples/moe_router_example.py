"""
MoE 路由器示例代码 / MoE Router Example Code
============================================

本示例展示混合专家模型（MoE）路由器的原理和实现。
This example demonstrates the principle and implementation of MoE router.

依赖安装 / Dependencies:
    pip install torch matplotlib numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. 基础路由器实现 / Basic Router Implementation
# =============================================================================

class MoERouter(nn.Module):
    """
    Top-K 路由器 / Top-K Router

    功能 / Functions:
    1. 计算每个 token 对每个专家的分数
    2. 选择 top-k 个专家
    3. 返回路由权重和索引
    """
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 路由器权重 / Router weights
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        参数 / Args:
            x: [batch, seq_len, d_model]

        返回 / Returns:
            gates: [batch, seq_len, top_k] - 路由权重
            indices: [batch, seq_len, top_k] - 选择的专家索引
        """
        # 计算路由分数 / Compute routing scores
        logits = self.gate(x)  # [batch, seq_len, num_experts]

        # Top-K 选择 / Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # Softmax 归一化（只对 top-k）/ Softmax normalization (only for top-k)
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        return top_k_gates, top_k_indices, logits

# =============================================================================
# 2. 测试路由器 / Test Router
# =============================================================================

print("=" * 60)
print("MoE 路由器测试 / MoE Router Test")
print("=" * 60)

torch.manual_seed(42)

d_model = 64
num_experts = 8
top_k = 2
batch_size, seq_len = 4, 10

# 创建路由器 / Create router
router = MoERouter(d_model, num_experts, top_k)

# 创建输入 / Create input
x = torch.randn(batch_size, seq_len, d_model)

# 路由 / Route
gates, indices, all_logits = router(x)

print(f"\n配置 / Configuration:")
print(f"  专家数量 / Num experts: {num_experts}")
print(f"  Top-K: {top_k}")

print(f"\n输入形状 / Input shape: {x.shape}")
print(f"路由权重形状 / Gates shape: {gates.shape}")
print(f"专家索引形状 / Indices shape: {indices.shape}")

print(f"\n第一个样本前 3 个 token 的路由 / Routing for first 3 tokens of first sample:")
for i in range(3):
    print(f"  Token {i}: 专家 {indices[0, i].tolist()} (权重 {gates[0, i].tolist()})")

# =============================================================================
# 3. 可视化路由分布 / Visualize Routing Distribution
# =============================================================================

print("\n" + "=" * 60)
print("路由分布可视化 / Routing Distribution Visualization")
print("=" * 60)

# 统计每个专家被选中的次数 / Count expert selections
expert_counts = torch.zeros(num_experts)
for b in range(batch_size):
    for s in range(seq_len):
        for idx in indices[b, s]:
            expert_counts[idx] += 1

# 可视化 / Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 3.1 专家选择频率 / Expert selection frequency
ax1 = axes[0]
ax1.bar(range(num_experts), expert_counts.numpy())
ax1.set_xlabel('Expert ID')
ax1.set_ylabel('Selection Count')
ax1.set_title('Expert Selection Distribution')
ax1.set_xticks(range(num_experts))

# 3.2 路由分数热力图 / Routing scores heatmap
ax2 = axes[1]
# 展示第一个样本的路由分数 / Show routing scores for first sample
ax2.imshow(all_logits[0].detach().numpy(), cmap='RdBu_r', aspect='auto')
ax2.set_xlabel('Expert ID')
ax2.set_ylabel('Token Position')
ax2.set_title('Routing Logits (First Sample)')
plt.colorbar(ax2.images[0], ax=ax2)

plt.tight_layout()
plt.savefig('/tmp/moe_router_distribution.png', dpi=150, bbox_inches='tight')
print(f"\n路由分布图像已保存至 / Distribution image saved to: /tmp/moe_router_distribution.png")

# =============================================================================
# 4. 负载均衡损失 / Load Balancing Loss
# =============================================================================

print("\n" + "=" * 60)
print("负载均衡损失 / Load Balancing Loss")
print("=" * 60)

def compute_load_balance_loss(gates, indices, num_experts):
    """
    计算负载均衡辅助损失 / Compute load balancing auxiliary loss

    L_aux = n * sum(f_i * P_i)
    其中:
    - f_i = 分配给专家 i 的 token 比例
    - P_i = 专家 i 的平均路由概率

    参数 / Args:
        gates: [batch, seq_len, top_k] - 路由权重
        indices: [batch, seq_len, top_k] - 专家索引
        num_experts: 专家总数

    返回 / Returns:
        loss: 负载均衡损失
    """
    batch_size, seq_len, top_k = gates.shape
    total_tokens = batch_size * seq_len

    # 计算 f_i: 每个 expert 实际处理的 token 比例
    # Compute f_i: actual proportion of tokens routed to each expert
    f = torch.zeros(num_experts, device=gates.device)
    for b in range(batch_size):
        for s in range(seq_len):
            for k in range(top_k):
                expert_idx = indices[b, s, k]
                f[expert_idx] += 1

    f = f / (total_tokens * top_k)  # 归一化

    # 计算 P_i: 每个 expert 的平均路由概率
    # Compute P_i: average routing probability for each expert
    # 这需要从完整的 logits 计算，这里用简化版本
    # 首先重建完整的 gates 矩阵
    full_gates = torch.zeros(batch_size, seq_len, num_experts, device=gates.device)
    for b in range(batch_size):
        for s in range(seq_len):
            for k in range(top_k):
                expert_idx = indices[b, s, k]
                full_gates[b, s, expert_idx] = gates[b, s, k]

    P = full_gates.mean(dim=(0, 1))  # [num_experts]

    # 负载均衡损失 / Load balance loss
    loss = num_experts * torch.sum(f * P)

    return loss, f, P

# 计算损失 / Compute loss
lb_loss, f, P = compute_load_balance_loss(gates, indices, num_experts)

print(f"\n负载均衡损失 / Load balance loss: {lb_loss.item():.4f}")
print(f"\n专家选择比例 f_i / Expert selection proportion f_i:")
for i, val in enumerate(f):
    print(f"  Expert {i}: {val.item():.4f}")

print(f"\n专家平均路由概率 P_i / Average routing probability P_i:")
for i, val in enumerate(P):
    print(f"  Expert {i}: {val.item():.4f}")

# 完美均衡时的损失 / Loss at perfect balance
perfect_loss = num_experts * (1/num_experts) * (1/num_experts)
print(f"\n完美均衡时的损失 / Perfect balance loss: {perfect_loss:.4f}")

# =============================================================================
# 5. 带噪声的路由 / Noisy Routing
# =============================================================================

print("\n" + "=" * 60)
print("带噪声的路由 / Noisy Routing")
print("=" * 60)

class NoisyMoERouter(nn.Module):
    """
    带噪声的 Top-K 路由器 / Noisy Top-K Router

    在路由分数上添加可学习的噪声，增加探索性
    Adds learnable noise to routing scores for exploration
    """
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.noise_weights = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        带噪声的路由 / Noisy routing

        H(x) = x @ W_g + noise * softplus(x @ W_noise)
        """
        # 基础路由分数 / Base routing scores
        logits = self.gate(x)

        if self.training:
            # 计算噪声 / Compute noise
            noise_logits = self.noise_weights(x)
            noise = torch.randn_like(logits) * F.softplus(noise_logits)
            logits = logits + noise

        # Top-K 选择 / Top-K selection
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)

        return top_k_gates, top_k_indices, logits

noisy_router = NoisyMoERouter(d_model, num_experts, top_k)
noisy_router.train()

gates_noisy, indices_noisy, _ = noisy_router(x)

print(f"\n带噪声路由结果 / Noisy routing results:")
print(f"第一个样本前 3 个 token 的路由 / Routing for first 3 tokens:")
for i in range(3):
    print(f"  Token {i}: 专家 {indices_noisy[0, i].tolist()} (权重 {gates_noisy[0, i].tolist()})")

# =============================================================================
# 6. 专家容量限制 / Expert Capacity Limit
# =============================================================================

print("\n" + "=" * 60)
print("专家容量限制 / Expert Capacity Limit")
print("=" * 60)

def routing_with_capacity(x, router, capacity_factor=1.25):
    """
    带容量限制的路由 / Routing with capacity limit

    每个 expert 最多处理 (total_tokens * top_k / num_experts) * capacity_factor 个 token
    """
    batch_size, seq_len, d_model = x.shape
    num_experts = router.num_experts
    top_k = router.top_k

    # 计算专家容量 / Compute expert capacity
    tokens_per_batch = batch_size * seq_len
    expert_capacity = int((tokens_per_batch * top_k / num_experts) * capacity_factor)

    print(f"每个专家最大容量 / Max capacity per expert: {expert_capacity} tokens")

    # 获取路由结果 / Get routing results
    gates, indices, logits = router(x)

    # 模拟容量限制 / Simulate capacity limit
    expert_load = torch.zeros(num_experts)
    overflow_count = 0

    for b in range(batch_size):
        for s in range(seq_len):
            for k in range(top_k):
                expert_idx = indices[b, s, k].item()
                if expert_load[expert_idx] < expert_capacity:
                    expert_load[expert_idx] += 1
                else:
                    overflow_count += 1
                    # 可以选择丢弃或路由到其他 expert

    print(f"\n专家负载 / Expert load:")
    for i, load in enumerate(expert_load):
        print(f"  Expert {i}: {int(load.item())} tokens (容量 {expert_capacity})")

    print(f"\n溢出 token 数 / Overflow tokens: {overflow_count}")

    return gates, indices

gates_cap, indices_cap = routing_with_capacity(x, router)

# =============================================================================
# 总结 / Summary
# =============================================================================
print("\n" + "=" * 60)
print("总结 / Summary")
print("=" * 60)
"""
本示例展示了 MoE 路由器的核心概念:
This example demonstrates core concepts of MoE router:

1. Top-K 路由 / Top-K routing:
   - 计算每个 token 对每个 expert 的分数
   - 选择分数最高的 K 个 expert
   - 对选中的 expert 权重进行 softmax 归一化

2. 负载均衡 / Load balancing:
   - 辅助损失鼓励均匀使用所有 expert
   - L_aux = n * sum(f_i * P_i)
   - 完美均衡时损失最小

3. 带噪声路由 / Noisy routing:
   - 在路由分数上添加噪声
   - 增加探索性，防止过早收敛

4. 容量限制 / Capacity limit:
   - 每个 expert 有最大处理 token 数
   - 防止单个 expert 过载
"""
