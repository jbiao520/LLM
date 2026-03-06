"""
奖励模型示例

演示如何训练和使用奖励模型进行回答评分。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRewardModel(nn.Module):
    """简单的奖励模型"""

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        # 输出标量奖励
        self.reward_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            rewards: [batch_size] 标量奖励
        """
        embeds = self.embedding(input_ids)
        hidden = self.transformer(embeds)

        # 使用最后一个 token 的隐藏状态
        last_hidden = hidden[:, -1, :]
        rewards = self.reward_head(last_hidden).squeeze(-1)

        return rewards


def bradley_terry_loss(chosen_rewards, rejected_rewards):
    """
    Bradley-Terry 损失函数

    P(chosen > rejected) = σ(r_chosen - r_rejected)

    Args:
        chosen_rewards: [batch_size] 好回答的奖励
        rejected_rewards: [batch_size] 差回答的奖励

    Returns:
        loss: 标量损失
    """
    # 计算 chosen 被选中的概率
    diff = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(diff).mean()

    return loss


def train_reward_model_step(model, optimizer, batch):
    """
    单步训练奖励模型

    Args:
        model: 奖励模型
        optimizer: 优化器
        batch: 包含 chosen_ids 和 rejected_ids

    Returns:
        loss: 损失值
        accuracy: 准确率
    """
    optimizer.zero_grad()

    # 计算奖励
    chosen_rewards = model(batch['chosen_ids'])
    rejected_rewards = model(batch['rejected_ids'])

    # 计算损失
    loss = bradley_terry_loss(chosen_rewards, rejected_rewards)

    # 反向传播
    loss.backward()
    optimizer.step()

    # 计算准确率
    with torch.no_grad():
        accuracy = (chosen_rewards > rejected_rewards).float().mean()

    return loss.item(), accuracy.item()


def demonstrate_reward_model():
    """演示奖励模型的概念"""
    print("=" * 60)
    print("奖励模型演示")
    print("=" * 60)

    # 模拟奖励分数
    print("\n模拟场景: 评估多个回答的质量")
    print("-" * 60)

    question = "如何学习编程？"

    answers = {
        "A": "从 Python 开始，它简单易学。推荐《Python编程从入门到实践》...",
        "B": "编程很难，不建议学。",
        "C": "随便找个教程看看就行。",
        "D": "学习编程的步骤：1. 选择语言 2. 找教程 3. 多练习 4. 做项目..."
    }

    # 模拟奖励模型输出
    rewards = {
        "A": 0.8,
        "B": -0.5,
        "C": 0.1,
        "D": 0.9
    }

    print(f"\n问题: {question}")
    print("\n回答及奖励分数:")
    for ans, text in answers.items():
        print(f"\n回答 {ans} (奖励: {rewards[ans]:.2f}):")
        print(f"  {text[:50]}...")

    # 排序
    sorted_answers = sorted(rewards.items(), key=lambda x: -x[1])
    print("\n\n排序结果:")
    for i, (ans, reward) in enumerate(sorted_answers):
        print(f"  {i+1}. 回答 {ans} (奖励: {reward:.2f})")


def demonstrate_bradley_terry():
    """演示 Bradley-Terry 模型"""
    print("\n" + "=" * 60)
    print("Bradley-Terry 模型")
    print("=" * 60)

    print("""
Bradley-Terry 模型用于从成对比较中学习评分:

公式:
P(A > B) = σ(r(A) - r(B))

其中:
- r(A), r(B) 是 A, B 的奖励分数
- σ 是 sigmoid 函数

示例:
┌─────────────────────────────────────────────────────────────┐
│ 回答 A 奖励: 0.8                                             │
│ 回答 B 奖励: 0.3                                             │
│                                                             │
│ P(A > B) = σ(0.8 - 0.3) = σ(0.5) ≈ 0.62                    │
│                                                             │
│ 含义: A 比 B 好的概率是 62%                                  │
└─────────────────────────────────────────────────────────────┘

训练目标:
最大化 P(chosen > rejected)

损失函数:
L = -log P(chosen > rejected)
  = -log σ(r_chosen - r_rejected)
    """)

    # 模拟不同奖励差对应的概率
    print("\n奖励差 vs 选择概率:")
    print("-" * 40)
    for diff in [-2, -1, -0.5, 0, 0.5, 1, 2]:
        prob = torch.sigmoid(torch.tensor(diff)).item()
        print(f"  差值 = {diff:+.1f} → P = {prob:.3f}")


def demonstrate_training():
    """演示训练过程"""
    print("\n" + "=" * 60)
    print("奖励模型训练过程")
    print("=" * 60)

    vocab_size = 1000
    embed_dim = 64
    hidden_dim = 128

    model = SimpleRewardModel(vocab_size, embed_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 模拟训练数据
    num_steps = 10
    batch_size = 4

    print(f"\n模型配置:")
    print(f"  词汇表大小: {vocab_size}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"  批次大小: {batch_size}")

    print(f"\n训练过程 (模拟):")
    print("-" * 40)

    for step in range(num_steps):
        # 模拟批次数据
        chosen_ids = torch.randint(0, vocab_size, (batch_size, 20))
        rejected_ids = torch.randint(0, vocab_size, (batch_size, 20))

        batch = {'chosen_ids': chosen_ids, 'rejected_ids': rejected_ids}
        loss, accuracy = train_reward_model_step(model, optimizer, batch)

        if step % 2 == 0:
            print(f"  Step {step}: Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")


def show_usage():
    """展示奖励模型的使用"""
    print("\n" + "=" * 60)
    print("奖励模型使用场景")
    print("=" * 60)

    print("""
1. 回答排序
   ┌─────────────────────────────────────────────────────────┐
   │ 问题: "什么是机器学习？"                                 │
   │                                                         │
   │ 生成多个回答 → 奖励模型打分 → 选择最高分                  │
   │                                                         │
   │ 回答 1: 0.8  ← 选择                                     │
   │ 回答 2: 0.5                                             │
   │ 回答 3: 0.3                                             │
   └─────────────────────────────────────────────────────────┘

2. RLHF 中的奖励信号
   ┌─────────────────────────────────────────────────────────┐
   │ 策略模型生成回答 → 奖励模型给出分数 → 用于 PPO 更新      │
   └─────────────────────────────────────────────────────────┘

3. 模型评估
   ┌─────────────────────────────────────────────────────────┐
   │ 比较不同模型在相同问题上的平均奖励                        │
   │                                                         │
   │ Model A: 平均奖励 0.72                                  │
   │ Model B: 平均奖励 0.68                                  │
   │ Model C: 平均奖励 0.75  ← 最好                          │
   └─────────────────────────────────────────────────────────┘

4. 过滤低质量输出
   ┌─────────────────────────────────────────────────────────┐
   │ 设定阈值，过滤奖励过低的回答                              │
   │                                                         │
   │ if reward < threshold:                                  │
   │     重新生成                                            │
   └─────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    demonstrate_reward_model()
    demonstrate_bradley_terry()
    demonstrate_training()
    show_usage()

    print("\n" + "=" * 60)
    print("关键要点:")
    print("=" * 60)
    print("""
1. 奖励模型学习给回答打分

2. 使用 Bradley-Terry 模型从成对比较中学习:
   P(A > B) = σ(r(A) - r(B))

3. 训练数据需要人类标注的偏好对

4. 奖励模型的用途:
   - 回答排序和选择
   - RLHF 中的奖励信号
   - 模型质量评估
   - 过滤低质量输出

5. 局限性:
   - 奖励模型可能学会"表面特征"而非真正质量
   - 需要持续更新以避免"奖励黑客"
    """)
