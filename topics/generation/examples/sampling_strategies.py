"""
文本生成采样策略示例

演示 Temperature、Top-K、Top-P 等采样策略的实现和效果。
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def softmax_with_temperature(logits, temperature):
    """
    带温度的 softmax

    Args:
        logits: 模型输出的 logits [batch_size, vocab_size]
        temperature: 温度参数，> 0

    Returns:
        概率分布
    """
    if temperature == 0:
        # 温度为 0 时退化为贪婪
        return F.one_hot(logits.argmax(dim=-1), num_classes=logits.size(-1)).float()

    scaled_logits = logits / temperature
    return F.softmax(scaled_logits, dim=-1)


def top_k_filtering(logits, top_k):
    """
    Top-K 过滤：将非 Top-K 的 logits 设为负无穷

    Args:
        logits: [batch_size, vocab_size]
        top_k: 保留的 token 数量

    Returns:
        过滤后的 logits
    """
    if top_k <= 0:
        return logits

    top_k = min(top_k, logits.size(-1))
    # 找到第 k 大的值
    values, _ = torch.topk(logits, top_k)
    min_value = values[:, -1].unsqueeze(-1)
    # 小于第 k 大的值设为负无穷
    return torch.where(logits < min_value, torch.full_like(logits, float('-inf')), logits)


def top_p_filtering(logits, top_p):
    """
    Top-P (Nucleus) 过滤：保留累计概率达到 top_p 的最小 token 集合

    Args:
        logits: [batch_size, vocab_size]
        top_p: 累计概率阈值 (0-1)

    Returns:
        过滤后的 logits
    """
    if top_p >= 1.0:
        return logits

    # 排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # 计算累计概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 找到需要移除的位置（累计概率超过 top_p 的）
    sorted_indices_to_remove = cumulative_probs > top_p

    # 保留第一个超过阈值的 token（右移一位）
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # 将需要移除的位置设为负无穷
    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # 恢复原始顺序
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def repetition_penalty(logits, prev_tokens, penalty=1.2):
    """
    重复惩罚：降低已出现 token 的概率

    Args:
        logits: [batch_size, vocab_size]
        prev_tokens: 之前生成的 token 列表
        penalty: 惩罚系数 (> 1)

    Returns:
        处理后的 logits
    """
    if len(prev_tokens) == 0:
        return logits

    for token in prev_tokens:
        if logits[0, token] > 0:
            logits[0, token] /= penalty
        else:
            logits[0, token] *= penalty

    return logits


def sample_token(logits, temperature=1.0, top_k=0, top_p=1.0, prev_tokens=None, rep_penalty=1.0):
    """
    综合采样函数

    Args:
        logits: [1, vocab_size]
        temperature: 温度
        top_k: Top-K 参数
        top_p: Top-P 参数
        prev_tokens: 之前生成的 token（用于重复惩罚）
        rep_penalty: 重复惩罚系数

    Returns:
        采样的 token id
    """
    # 1. 重复惩罚
    if prev_tokens and rep_penalty > 1.0:
        logits = repetition_penalty(logits.clone(), prev_tokens, rep_penalty)

    # 2. Top-K 过滤
    if top_k > 0:
        logits = top_k_filtering(logits, top_k)

    # 3. Top-P 过滤
    if top_p < 1.0:
        logits = top_p_filtering(logits, top_p)

    # 4. 温度缩放 + Softmax
    probs = softmax_with_temperature(logits, temperature)

    # 5. 采样
    if temperature == 0:
        return probs.argmax(dim=-1).item()

    return torch.multinomial(probs, num_samples=1).item()


def demonstrate_temperature():
    """演示温度对概率分布的影响"""
    print("=" * 60)
    print("Temperature 效果演示")
    print("=" * 60)

    # 模拟 logits（5 个 token）
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.3, 0.2]])

    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]

    print(f"\n原始 logits: {logits[0].tolist()}")
    print("\n不同温度下的概率分布:")
    print("-" * 60)

    for temp in temperatures:
        probs = softmax_with_temperature(logits, temp)
        print(f"T = {temp}: {probs[0].tolist()}")

    # 可视化
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(5)

        for temp in temperatures:
            probs = softmax_with_temperature(logits, temp)
            ax.bar([i + temperatures.index(temp) * 0.15 for i in x],
                   probs[0].tolist(), width=0.15, label=f'T={temp}')

        ax.set_xlabel('Token')
        ax.set_ylabel('Probability')
        ax.set_title('Effect of Temperature on Probability Distribution')
        ax.legend()
        ax.set_xticks([i + 0.3 for i in x])
        ax.set_xticklabels(['A', 'B', 'C', 'D', 'E'])
        plt.savefig('temperature_effect.png')
        print("\n图表已保存到 temperature_effect.png")
        plt.close()
    except Exception as e:
        print(f"\n无法生成图表: {e}")


def demonstrate_top_k_top_p():
    """演示 Top-K 和 Top-P 的效果"""
    print("\n" + "=" * 60)
    print("Top-K vs Top-P 演示")
    print("=" * 60)

    # 模拟一个更真实的 logits 分布
    logits = torch.tensor([[3.0, 2.5, 1.5, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.01]])

    print(f"\n原始 logits: {logits[0].tolist()}")

    probs = F.softmax(logits, dim=-1)
    print(f"原始概率: {probs[0].tolist()}")

    # Top-K
    print("\n--- Top-K 过滤 ---")
    for k in [3, 5]:
        filtered = top_k_filtering(logits.clone(), k)
        filtered_probs = F.softmax(filtered, dim=-1)
        print(f"Top-{k}: {filtered_probs[0].tolist()}")

    # Top-P
    print("\n--- Top-P 过滤 ---")
    for p in [0.8, 0.9, 0.95]:
        filtered = top_p_filtering(logits.clone(), p)
        filtered_probs = F.softmax(filtered, dim=-1)
        non_zero = [(i, f"{p:.3f}") for i, p in enumerate(filtered_probs[0].tolist()) if p > 0.001]
        print(f"Top-{p}: 保留 {len(non_zero)} 个 token: {non_zero}")


def sampling_comparison():
    """比较不同采样策略的效果"""
    print("\n" + "=" * 60)
    print("采样策略对比")
    print("=" * 60)

    # 模拟词汇表
    vocab = ["不错", "很好", "一般", "糟糕", "极好", "还行", "很差", "完美"]
    vocab_size = len(vocab)

    # 模拟 logits（对应 "今天天气" 后的预测）
    logits = torch.tensor([[2.0, 1.5, 1.0, 0.5, 0.8, 0.6, 0.3, 0.4]])

    print(f"\n词汇表: {vocab}")
    print(f"Logits: {logits[0].tolist()}")

    # 多次采样比较
    num_samples = 1000

    configs = [
        {"name": "贪婪 (T=0)", "temperature": 0, "top_k": 0, "top_p": 1.0},
        {"name": "T=0.5", "temperature": 0.5, "top_k": 0, "top_p": 1.0},
        {"name": "T=1.0", "temperature": 1.0, "top_k": 0, "top_p": 1.0},
        {"name": "Top-K=3", "temperature": 1.0, "top_k": 3, "top_p": 1.0},
        {"name": "Top-P=0.9", "temperature": 1.0, "top_k": 0, "top_p": 0.9},
    ]

    print("\n各策略采样结果 (1000 次):")
    print("-" * 60)

    for config in configs:
        counts = {word: 0 for word in vocab}
        for _ in range(num_samples):
            token_id = sample_token(
                logits.clone(),
                temperature=config["temperature"],
                top_k=config["top_k"],
                top_p=config["top_p"]
            )
            counts[vocab[token_id]] += 1

        # 排序并显示
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        top_3 = sorted_counts[:3]
        print(f"{config['name']}: {top_3}")


def repetition_penalty_demo():
    """演示重复惩罚的效果"""
    print("\n" + "=" * 60)
    print("重复惩罚演示")
    print("=" * 60)

    vocab = ["A", "B", "C", "D", "E"]
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])  # 均匀分布

    print(f"\n原始 logits (均匀分布): {logits[0].tolist()}")

    prev_tokens = [0, 0, 0]  # "A" 已出现 3 次

    for penalty in [1.0, 1.2, 1.5, 2.0]:
        penalized = repetition_penalty(logits.clone(), prev_tokens, penalty)
        probs = F.softmax(penalized, dim=-1)
        print(f"\n惩罚系数 = {penalty}:")
        print(f"  Logits: {penalized[0].tolist()}")
        print(f"  概率: {probs[0].tolist()}")
        print(f"  'A' 的概率: {probs[0, 0].item():.3f}")


def complete_generation_example():
    """完整的生成示例"""
    print("\n" + "=" * 60)
    print("完整文本生成示例")
    print("=" * 60)

    # 模拟一个小词汇表
    vocab = ["<pad>", "<eos>", "今天", "天气", "不错", "很好", "晴朗", "适合", "出门", "散步"]
    vocab_size = len(vocab)

    # 模拟一个简单的"模型"（实际中会用神经网络）
    # 这里用随机 logits 模拟，但给某些组合更高概率
    def fake_model(context):
        """模拟模型输出"""
        logits = torch.randn(1, vocab_size) * 0.5

        # 给一些"合理"的后续更高分
        if len(context) > 0:
            last = context[-1]
            if last == 2:  # "今天"
                logits[0, 3] += 2  # "天气"
            elif last == 3:  # "天气"
                logits[0, 4:7] += 1  # "不错", "很好", "晴朗"
            elif last in [4, 5, 6]:  # 形容词后
                logits[0, 7] += 1  # "适合"
            elif last == 7:  # "适合"
                logits[0, 8:10] += 1  # "出门", "散步"

        return logits

    # 生成函数
    def generate(max_length=10, temperature=0.8, top_p=0.9):
        generated = []
        context = []

        for _ in range(max_length):
            logits = fake_model(context)
            token = sample_token(
                logits,
                temperature=temperature,
                top_p=top_p,
                prev_tokens=context,
                rep_penalty=1.1
            )

            if token == 1:  # <eos>
                break

            generated.append(token)
            context.append(token)

        return [vocab[t] for t in generated]

    # 生成几次
    print("\n生成示例 (模拟模型):")
    for i in range(5):
        result = generate()
        print(f"  {i + 1}. {' '.join(result)}")


if __name__ == "__main__":
    demonstrate_temperature()
    demonstrate_top_k_top_p()
    sampling_comparison()
    repetition_penalty_demo()
    complete_generation_example()

    print("\n" + "=" * 60)
    print("关键要点:")
    print("=" * 60)
    print("""
1. Temperature 控制输出的确定性 vs 随机性
   - T → 0: 更确定，可能重复
   - T → ∞: 更随机，可能不连贯

2. Top-K 限制候选数量，简单但可能不够灵活

3. Top-P (Nucleus) 自适应选择候选，通常效果更好

4. 重复惩罚防止生成重复内容

5. 实际应用中通常组合使用:
   - Temperature = 0.7-1.0
   - Top-P = 0.9-0.95
   - Repetition Penalty = 1.1-1.2
    """)
