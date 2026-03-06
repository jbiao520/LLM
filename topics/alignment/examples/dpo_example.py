"""
DPO (Direct Preference Optimization) 示例

演示 DPO 算法的核心实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dpo_loss(policy_log_probs, reference_log_probs, beta=0.1):
    """
    DPO 损失函数

    Args:
        policy_log_probs: 策略模型的 log 概率 [batch_size, 2]
                         [:, 0] = y_w (好回答), [:, 1] = y_l (差回答)
        reference_log_probs: 参考模型的 log 概率 [batch_size, 2]
        beta: KL 惩罚系数

    Returns:
        loss: DPO 损失
    """
    # 计算 log 比率
    # log π_θ(y|x) - log π_ref(y|x)
    log_ratio_w = policy_log_probs[:, 0] - reference_log_probs[:, 0]
    log_ratio_l = policy_log_probs[:, 1] - reference_log_probs[:, 1]

    # DPO 损失
    # -log σ(β * (log_ratio_w - log_ratio_l))
    logits = beta * (log_ratio_w - log_ratio_l)
    loss = -F.logsigmoid(logits).mean()

    return loss


def compute_log_probs(model, input_ids, attention_mask=None):
    """
    计算模型对给定序列的 log 概率

    Args:
        model: 语言模型
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]

    Returns:
        log_probs: 每个序列的总 log 概率 [batch_size]
    """
    # 前向传播
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # 计算每个位置的 log 概率
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # 交叉熵
    log_probs = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction='none'
    )

    # 重塑并求和
    log_probs = log_probs.reshape(input_ids.size(0), -1)
    # 注意：这里得到的是负 log 概率（损失）
    return -log_probs.sum(dim=-1)


class DPOTrainer:
    """简化的 DPO 训练器"""

    def __init__(self, policy_model, reference_model, beta=0.1, lr=1e-6):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.beta = beta

        # 冻结参考模型
        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=lr)

    def train_step(self, batch):
        """
        单步训练

        Args:
            batch: 包含以下键的字典
                - prompt_ids: 提示 token IDs [batch_size, prompt_len]
                - chosen_ids: 好回答 token IDs [batch_size, chosen_len]
                - rejected_ids: 差回答 token IDs [batch_size, rejected_len]

        Returns:
            loss_dict: 包含损失和指标的字典
        """
        self.optimizer.zero_grad()

        # 构建完整序列
        chosen_input = torch.cat([batch['prompt_ids'], batch['chosen_ids']], dim=1)
        rejected_input = torch.cat([batch['prompt_ids'], batch['rejected_ids']], dim=1)

        # 计算策略模型的 log 概率
        policy_chosen_logp = compute_log_probs(self.policy_model, chosen_input)
        policy_rejected_logp = compute_log_probs(self.policy_model, rejected_input)
        policy_log_probs = torch.stack([policy_chosen_logp, policy_rejected_logp], dim=1)

        # 计算参考模型的 log 概率（不计算梯度）
        with torch.no_grad():
            ref_chosen_logp = compute_log_probs(self.reference_model, chosen_input)
            ref_rejected_logp = compute_log_probs(self.reference_model, rejected_input)
            reference_log_probs = torch.stack([ref_chosen_logp, ref_rejected_logp], dim=1)

        # 计算 DPO 损失
        loss = dpo_loss(policy_log_probs, reference_log_probs, self.beta)

        # 反向传播
        loss.backward()
        self.optimizer.step()

        # 计算指标
        with torch.no_grad():
            # 准确率：好回答的 log 比率是否高于差回答
            log_ratio_w = policy_log_probs[:, 0] - reference_log_probs[:, 0]
            log_ratio_l = policy_log_probs[:, 1] - reference_log_probs[:, 1]
            accuracy = (log_ratio_w > log_ratio_l).float().mean()

        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'log_ratio_w': log_ratio_w.mean().item(),
            'log_ratio_l': log_ratio_l.mean().item(),
        }


def demonstrate_dpo():
    """演示 DPO 的核心概念"""
    print("=" * 60)
    print("DPO 核心概念演示")
    print("=" * 60)

    # 模拟 log 概率
    # 假设 batch_size = 4
    batch_size = 4

    # 策略模型的 log 概率
    policy_log_probs = torch.tensor([
        [-5.0, -8.0],  # 样本 1: 好回答 -5, 差回答 -8
        [-6.0, -7.0],  # 样本 2: 好回答 -6, 差回答 -7
        [-4.5, -9.0],  # 样本 3: 好回答 -4.5, 差回答 -9
        [-7.0, -6.0],  # 样本 4: 好回答 -7, 差回答 -6 (不好!)
    ])

    # 参考模型的 log 概率
    reference_log_probs = torch.tensor([
        [-6.0, -6.0],  # 初始时两者相同
        [-6.0, -6.0],
        [-6.0, -6.0],
        [-6.0, -6.0],
    ])

    print("\n策略模型的 log 概率 (好回答, 差回答):")
    print(policy_log_probs)

    print("\n参考模型的 log 概率 (好回答, 差回答):")
    print(reference_log_probs)

    # 计算 log 比率
    log_ratio_w = policy_log_probs[:, 0] - reference_log_probs[:, 0]
    log_ratio_l = policy_log_probs[:, 1] - reference_log_probs[:, 1]

    print("\nLog 比率 (好回答):", log_ratio_w.tolist())
    print("Log 比率 (差回答):", log_ratio_l.tolist())

    # 计算 DPO 损失
    for beta in [0.1, 0.5, 1.0]:
        loss = dpo_loss(policy_log_probs, reference_log_probs, beta=beta)
        print(f"\nβ={beta} 时的 DPO 损失: {loss.item():.4f}")

    # 解释
    print("\n" + "-" * 60)
    print("解释:")
    print("-" * 60)
    print("""
样本 1-3: 好回答的 log 比率高于差回答 → 低损失
样本 4: 好回答的 log 比率低于差回答 → 高损失

DPO 的目标:
- 增大好回答的概率 (相对于参考模型)
- 降低差回答的概率 (相对于参考模型)
    """)


def demonstrate_training_dynamics():
    """演示训练动态"""
    print("\n" + "=" * 60)
    print("DPO 训练动态")
    print("=" * 60)

    print("""
DPO 训练过程中发生的变化:

初始状态 (训练前):
┌─────────────────────────────────────────────────────────┐
│ π_θ(y_w|x) ≈ π_ref(y_w|x)                               │
│ π_θ(y_l|x) ≈ π_ref(y_l|x)                               │
│                                                         │
│ log 比率 ≈ 0                                            │
└─────────────────────────────────────────────────────────┘

训练中:
┌─────────────────────────────────────────────────────────┐
│ π_θ(y_w|x) ↑ (增大好回答概率)                            │
│ π_θ(y_l|x) ↓ (降低差回答概率)                            │
│                                                         │
│ log π_θ(y_w|x) - log π_ref(y_w|x) → 正                 │
│ log π_θ(y_l|x) - log π_ref(y_l|x) → 负                 │
└─────────────────────────────────────────────────────────┘

训练后:
┌─────────────────────────────────────────────────────────┐
│ π_θ(y_w|x) > π_ref(y_w|x)                               │
│ π_θ(y_l|x) < π_ref(y_l|x)                               │
│                                                         │
│ 模型学会了偏好好回答胜过差回答                            │
└─────────────────────────────────────────────────────────┘

β 的作用:
- β 大: 更严格遵循偏好，但可能过拟合
- β 小: 更灵活，但可能偏离不够
    """)


def compare_with_rlhf():
    """与 RLHF 对比"""
    print("\n" + "=" * 60)
    print("DPO vs RLHF 对比")
    print("=" * 60)

    print("""
┌─────────────────────────────────────────────────────────────────┐
│                         RLHF                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  步骤:                                                          │
│  1. 训练奖励模型 (需要偏好数据)                                  │
│  2. 用 PPO 优化策略 (需要 RL 框架)                               │
│                                                                 │
│  复杂度: 高                                                      │
│  - 需要维护 4 个模型: 策略、参考、奖励、值                        │
│  - 需要调整 PPO 超参数                                          │
│  - 训练不稳定                                                    │
│                                                                 │
│  优点:                                                          │
│  - 理论成熟                                                      │
│  - 可以在线学习                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          DPO                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  步骤:                                                          │
│  1. 直接用偏好数据优化策略 (简单!)                               │
│                                                                 │
│  复杂度: 低                                                      │
│  - 只需要 2 个模型: 策略、参考                                    │
│  - 标准的监督学习                                                │
│  - 训练稳定                                                      │
│                                                                 │
│  优点:                                                          │
│  - 实现简单                                                      │
│  - 不需要奖励模型                                                │
│  - 效果相当或更好                                                │
│                                                                 │
│  缺点:                                                          │
│  - 只能离线学习                                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    demonstrate_dpo()
    demonstrate_training_dynamics()
    compare_with_rlhf()

    print("\n" + "=" * 60)
    print("关键要点:")
    print("=" * 60)
    print("""
1. DPO 直接用偏好数据优化模型，无需奖励模型

2. 核心公式:
   L = -log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                   - log π_θ(y_l|x) + log π_ref(y_l|x)))

3. β 控制偏离程度:
   - 大 β: 更严格遵循偏好
   - 小 β: 更灵活

4. DPO 比 RLHF 简单得多，但效果相当

5. 实际使用:
   - 准备偏好数据 (问题, 好回答, 差回答)
   - 用 DPO 损失微调模型
   - 监控准确率和 log 比率
    """)
