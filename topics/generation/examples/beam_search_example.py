"""
Beam Search 实现示例

演示标准 Beam Search 及其变体的实现。
"""

import torch
import torch.nn.functional as F
from collections import namedtuple
import heapq

# 用于存储 beam 候选的数据结构
BeamCandidate = namedtuple('BeamCandidate', ['score', 'tokens', 'finished'])


class BeamSearch:
    """标准 Beam Search 实现"""

    def __init__(self, beam_size, max_length, eos_token_id, length_penalty=0.0):
        """
        Args:
            beam_size: beam 宽度
            max_length: 最大生成长度
            eos_token_id: 结束 token ID
            length_penalty: 长度惩罚系数 (0 = 无惩罚, 1 = 完全归一化)
        """
        self.beam_size = beam_size
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.length_penalty = length_penalty

    def search(self, model, input_ids, attention_mask=None):
        """
        执行 beam search

        Args:
            model: 语言模型，需要支持 forward() 返回 logits
            input_ids: 输入 token IDs [1, seq_len]
            attention_mask: 注意力掩码

        Returns:
            best_tokens: 最佳序列的 token IDs
            best_score: 最佳序列的分数
        """
        # 初始化 beam
        # 每个 beam 元素: (累积分数, token 序列, 是否完成)
        beams = [(0.0, input_ids[0].tolist(), False)]

        # 已完成的候选
        completed = []

        for step in range(self.max_length):
            all_candidates = []

            for score, tokens, finished in beams:
                if finished:
                    # 已完成的 beam 直接保留
                    all_candidates.append((score, tokens, True))
                    continue

                # 准备输入
                current_input = torch.tensor([tokens])

                # 模型前向传播
                with torch.no_grad():
                    outputs = model(current_input)
                    logits = outputs.logits[:, -1, :]  # 取最后一个位置的 logits

                # 计算 log 概率
                log_probs = F.log_softmax(logits, dim=-1)

                # 获取 top-k 候选
                top_k_probs, top_k_ids = torch.topk(log_probs, self.beam_size)

                for i in range(self.beam_size):
                    token_id = top_k_ids[0, i].item()
                    token_prob = top_k_probs[0, i].item()

                    new_tokens = tokens + [token_id]
                    new_score = score + token_prob

                    # 检查是否结束
                    is_finished = (token_id == self.eos_token_id)

                    if is_finished:
                        # 应用长度惩罚
                        length = len(new_tokens) - len(input_ids[0])
                        normalized_score = new_score / ((5 + length) / 6) ** self.length_penalty
                        completed.append((normalized_score, new_tokens))
                    else:
                        all_candidates.append((new_score, new_tokens, False))

            # 选择 top-k beam
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:self.beam_size]

            # 如果所有 beam 都完成了，提前停止
            if all(finished for _, _, finished in beams):
                break

        # 如果没有完成的候选，将当前 beam 加入
        if not completed:
            for score, tokens, _ in beams:
                length = len(tokens) - len(input_ids[0])
                normalized_score = score / ((5 + length) / 6) ** self.length_penalty
                completed.append((normalized_score, tokens))

        # 返回最佳候选
        completed.sort(key=lambda x: x[0], reverse=True)
        best_score, best_tokens = completed[0]

        return best_tokens, best_score


class DiverseBeamSearch:
    """Diverse Beam Search 实现"""

    def __init__(self, beam_size, num_groups, diversity_penalty, max_length, eos_token_id):
        """
        Args:
            beam_size: 每组的 beam 宽度
            num_groups: 分组数量
            diversity_penalty: 多样性惩罚系数
            max_length: 最大生成长度
            eos_token_id: 结束 token ID
        """
        self.beam_size = beam_size
        self.num_groups = num_groups
        self.diversity_penalty = diversity_penalty
        self.max_length = max_length
        self.eos_token_id = eos_token_id

    def search(self, model, input_ids):
        """
        执行 diverse beam search

        将 beam 分成多组，组间施加多样性惩罚
        """
        # 初始化每个组的 beam
        group_beams = [
            [(0.0, input_ids[0].tolist(), False)]
            for _ in range(self.num_groups)
        ]

        completed = [[] for _ in range(self.num_groups)]

        for step in range(self.max_length):
            for group_idx in range(self.num_groups):
                all_candidates = []
                current_group = group_beams[group_idx]

                # 收集其他组已选择的 token（用于多样性惩罚）
                other_group_tokens = set()
                for other_idx in range(self.num_groups):
                    if other_idx != group_idx:
                        for _, tokens, _ in group_beams[other_idx]:
                            other_group_tokens.add(tokens[-1] if len(tokens) > len(input_ids[0]) else -1)

                for score, tokens, finished in current_group:
                    if finished:
                        all_candidates.append((score, tokens, True))
                        continue

                    current_input = torch.tensor([tokens])

                    with torch.no_grad():
                        outputs = model(current_input)
                        logits = outputs.logits[:, -1, :].clone()

                    # 应用多样性惩罚
                    for token_id in other_group_tokens:
                        if token_id >= 0:
                            logits[0, token_id] -= self.diversity_penalty

                    log_probs = F.log_softmax(logits, dim=-1)
                    top_k_probs, top_k_ids = torch.topk(log_probs, self.beam_size)

                    for i in range(self.beam_size):
                        token_id = top_k_ids[0, i].item()
                        token_prob = top_k_probs[0, i].item()

                        new_tokens = tokens + [token_id]
                        new_score = score + token_prob
                        is_finished = (token_id == self.eos_token_id)

                        if is_finished:
                            completed[group_idx].append((new_score, new_tokens))
                        else:
                            all_candidates.append((new_score, new_tokens, False))

                all_candidates.sort(key=lambda x: x[0], reverse=True)
                group_beams[group_idx] = all_candidates[:self.beam_size]

        # 合并所有组的完成候选
        all_completed = []
        for group in completed:
            all_completed.extend(group)

        if not all_completed:
            for group in group_beams:
                for score, tokens, _ in group:
                    all_completed.append((score, tokens))

        all_completed.sort(key=lambda x: x[0], reverse=True)
        return all_completed[0][1], all_completed[0][0]


def demonstrate_beam_search():
    """演示 Beam Search 的基本概念"""
    print("=" * 60)
    print("Beam Search 演示")
    print("=" * 60)

    # 模拟词汇表和模型
    vocab = ["<pad>", "<eos>", "I", "love", "coding", "Python", "Java", "and", "AI"]

    class FakeModel:
        """模拟语言模型"""
        def forward(self, input_ids):
            # 简单的模拟：根据上一个词预测下一个词
            batch_size = input_ids.shape[0]
            vocab_size = len(vocab)

            # 随机 logits
            logits = torch.randn(batch_size, 1, vocab_size)

            # 添加一些"合理"的偏好
            last_token = input_ids[0, -1].item()
            if last_token == 2:  # "I"
                logits[0, 0, 3] += 2  # "love"
            elif last_token == 3:  # "love"
                logits[0, 0, 4:6] += 1  # "coding" or "Python"
            elif last_token == 4:  # "coding"
                logits[0, 0, 5] += 1  # "Python"

            return type('Output', (), {'logits': logits})()

    # 创建模型实例
    model = FakeModel()
    model.eval = lambda: None

    print("\n词汇表:", vocab)
    print("Beam Size = 3, Max Length = 5")

    # 运行 beam search
    beam_search = BeamSearch(
        beam_size=3,
        max_length=5,
        eos_token_id=1,
        length_penalty=0.6
    )

    input_ids = torch.tensor([[2]])  # "I"

    print("\n开始生成...")
    print("-" * 40)

    # 手动模拟 beam search 过程
    print("\nStep 1: 从 'I' 开始")
    print("  Beam: ['I']")

    print("\nStep 2: 扩展")
    print("  Candidates:")
    print("    'I love' (score: -0.2)")
    print("    'I coding' (score: -0.8)")
    print("    'I Python' (score: -1.0)")

    print("\nStep 3: 继续扩展 top-3")
    print("  Beam: ['I love', 'I coding', 'I Python']")
    print("  Expanding 'I love':")
    print("    'I love coding' (score: -0.5)")
    print("    'I love Python' (score: -0.6)")

    print("\n最终结果:")
    print("  Best: 'I love coding Python <eos>' (score: -1.2)")


def visualize_beam_search():
    """可视化 Beam Search 过程"""
    print("\n" + "=" * 60)
    print("Beam Search 可视化")
    print("=" * 60)

    print("""
                    ┌─────────────────────────────────────┐
                    │         Beam Search 过程            │
                    └─────────────────────────────────────┘

    Step 0: 初始化
    ┌──────────────────┐
    │ Beam: ["The"]    │
    │ Score: 0.0       │
    └──────────────────┘

    Step 1: 扩展
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │ "The cat"        │   │ "The dog"        │   │ "The bird"       │
    │ Score: -0.3      │   │ Score: -0.4      │   │ Score: -0.8      │
    └──────────────────┘   └──────────────────┘   └──────────────────┘
            │                      │                      │
            ▼                      ▼                      ▼
    Step 2: 继续扩展 (Beam Width = 3)
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │ "The cat sat"    │   │ "The dog ran"    │   │ "The cat ran"    │
    │ Score: -0.6      │   │ Score: -0.7      │   │ Score: -0.8      │
    └──────────────────┘   └──────────────────┘   └──────────────────┘

    ... 继续直到 <eos> 或最大长度 ...

    最终选择: "The cat sat on the mat <eos>" (score: -2.1)
    """)


def compare_greedy_vs_beam():
    """比较贪婪搜索和 Beam Search"""
    print("\n" + "=" * 60)
    print("贪婪搜索 vs Beam Search")
    print("=" * 60)

    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    贪婪搜索 (Greedy)                            │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  每步选择概率最高的词，只保留一条路径                             │
    │                                                                 │
    │  优点:                                                          │
    │  - 速度快 (O(V × T))                                           │
    │  - 实现简单                                                     │
    │                                                                 │
    │  缺点:                                                          │
    │  - 可能陷入局部最优                                             │
    │  - 输出可能重复、无聊                                           │
    │  - 不适合需要全局优化的任务                                      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │                    Beam Search                                  │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  每步保留 k 条最有希望的路径                                     │
    │                                                                 │
    │  优点:                                                          │
    │  - 更可能找到全局最优解                                          │
    │  - 输出质量通常更高                                             │
    │  - 适合翻译、摘要等任务                                          │
    │                                                                 │
    │  缺点:                                                          │
    │  - 速度慢 (O(k × V × T))                                       │
    │  - 内存占用更大                                                 │
    │  - 仍可能产生重复                                               │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    示例对比:

    输入: "Translate to English: 猫"

    贪婪:
    "The cat" → "The cat is" → "The cat is a" → "The cat is a pet"
    (每步最优，但整体可能不是最佳翻译)

    Beam (k=3):
    ┌─ "The cat" ────→ "The cat" ────→ "The cat" ──→ "The cat"
    │
    ├─ "A cat" ──────→ "A cat" ──────→ "A cat" ────→ "A cat is"
    │
    └─ "Cat" ────────→ "Cat means" ──→ (淘汰)

    最终选择: "A cat is a pet" (整体分数最高)
    """)


def length_normalization_demo():
    """演示长度归一化的效果"""
    print("\n" + "=" * 60)
    print("长度归一化")
    print("=" * 60)

    # 模拟两个候选序列
    candidates = [
        ("The cat sat", -3.0, 4),      # 短序列
        ("The cat sat on the mat", -4.5, 7),  # 长序列
    ]

    print("\n候选序列:")
    for text, score, length in candidates:
        print(f"  '{text}' (累积分数: {score}, 长度: {length})")

    print("\n不使用归一化:")
    best = max(candidates, key=lambda x: x[1])
    print(f"  最佳: '{best[0]}' (分数: {best[1]})")

    print("\n使用长度归一化 (α=0.6):")
    alpha = 0.6
    normalized = [
        (text, score / ((5 + length) / 6) ** alpha, length)
        for text, score, length in candidates
    ]
    for text, norm_score, length in normalized:
        print(f"  '{text}' (归一化分数: {norm_score:.3f})")

    best_norm = max(normalized, key=lambda x: x[1])
    print(f"  最佳: '{best_norm[0]}' (归一化分数: {best_norm[1]:.3f})")


if __name__ == "__main__":
    demonstrate_beam_search()
    visualize_beam_search()
    compare_greedy_vs_beam()
    length_normalization_demo()

    print("\n" + "=" * 60)
    print("关键要点:")
    print("=" * 60)
    print("""
1. Beam Search 保留多条候选路径，比贪婪搜索更全面

2. Beam Width (k) 的选择:
   - k=1: 等同于贪婪
   - k=3-5: 常用设置
   - k 越大，质量越高但速度越慢

3. 长度归一化避免偏向短序列:
   - score_normalized = score / length^α
   - α 通常设为 0.6-1.0

4. Diverse Beam Search 通过分组增加多样性

5. 适用场景:
   - 机器翻译: Beam Search 效果好
   - 创意写作: 采样策略更合适
   - 代码生成: 可用 Beam Search 或低温度采样
    """)
