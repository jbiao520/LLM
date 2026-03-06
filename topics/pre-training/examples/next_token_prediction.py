"""
下一个词预测示例

演示预训练的核心任务：下一个词预测。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleLanguageModel(nn.Module):
    """简单的语言模型（用于教学）"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(512, embed_dim)  # 最大序列长度 512

        # 简单的 Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出层
        self.output = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len] token IDs

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape

        # Token embedding + Position embedding
        positions = torch.arange(seq_len, device=x.device)
        embeds = self.embedding(x) + self.pos_embedding(positions)

        # 创建因果掩码（只能看到之前的 token）
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )

        # Transformer
        # 这里用 memory=x 作为简化（实际应该是编码器输出）
        output = self.transformer(embeds, embeds, tgt_mask=causal_mask)

        # 输出 logits
        logits = self.output(output)

        return logits

    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        """生成新 token"""
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # 前向传播
            logits = self.forward(generated)

            # 取最后一个位置的 logits
            next_logits = logits[:, -1, :] / temperature

            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 添加到序列
            generated = torch.cat([generated, next_token], dim=1)

        return generated


def compute_loss(model, input_ids):
    """
    计算下一个词预测损失

    Args:
        model: 语言模型
        input_ids: [batch_size, seq_len] token IDs

    Returns:
        loss: 标量损失值
    """
    # 输入是前 n-1 个 token
    inputs = input_ids[:, :-1]
    # 目标是后 n-1 个 token
    targets = input_ids[:, 1:]

    # 前向传播
    logits = model(inputs)  # [batch, seq_len-1, vocab_size]

    # 计算交叉熵损失
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=0  # 忽略 padding token
    )

    return loss


def compute_perplexity(model, input_ids):
    """
    计算困惑度

    PPL = exp(average_loss)
    """
    loss = compute_loss(model, input_ids)
    perplexity = torch.exp(loss)
    return perplexity.item()


def demonstrate_next_token_prediction():
    """演示下一个词预测"""
    print("=" * 60)
    print("下一个词预测演示")
    print("=" * 60)

    # 创建一个小型词汇表
    vocab = ["<pad>", "<unk>", "今天", "天气", "不错", "很好", "晴朗", "适合", "出门"]
    vocab_size = len(vocab)
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}

    # 创建模型
    model = SimpleLanguageModel(vocab_size, embed_dim=64, hidden_dim=128)
    model.eval()

    # 示例输入
    sentence = "今天 天气"
    tokens = [word_to_id.get(w, 1) for w in sentence.split()]
    input_ids = torch.tensor([tokens])

    print(f"\n输入: '{sentence}'")
    print(f"Token IDs: {tokens}")

    # 前向传播
    with torch.no_grad():
        logits = model(input_ids)

    print(f"\n输出 logits 形状: {logits.shape}")

    # 查看最后一个位置的预测
    last_logits = logits[0, -1, :]
    probs = F.softmax(last_logits, dim=-1)

    print("\n下一个词预测概率:")
    for i, (word, prob) in enumerate(sorted(zip(vocab, probs.tolist()), key=lambda x: -x[1])[:5]):
        print(f"  {word}: {prob:.4f}")

    # 预测的词
    predicted_id = torch.argmax(last_logits).item()
    print(f"\n预测的下一个词: '{id_to_word[predicted_id]}'")


def demonstrate_training_step():
    """演示一个训练步骤"""
    print("\n" + "=" * 60)
    print("训练步骤演示")
    print("=" * 60)

    # 创建词汇和模型
    vocab = ["<pad>", "<unk>", "今天", "天气", "不错", "很好", "晴朗"]
    vocab_size = len(vocab)
    word_to_id = {w: i for i, w in enumerate(vocab)}

    model = SimpleLanguageModel(vocab_size, embed_dim=64, hidden_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练数据
    sentences = [
        "今天 天气 不错",
        "今天 天气 很好",
        "天气 晴朗",
    ]

    # 转换为 token IDs
    data = []
    for sent in sentences:
        tokens = [word_to_id.get(w, 1) for w in sent.split()]
        data.append(tokens)

    # 找到最大长度并填充
    max_len = max(len(t) for t in data)
    padded_data = [t + [0] * (max_len - len(t)) for t in data]
    input_ids = torch.tensor(padded_data)

    print(f"\n训练数据: {len(sentences)} 个样本")
    print(f"最大长度: {max_len}")

    # 训练几步
    print("\n训练过程:")
    for step in range(10):
        optimizer.zero_grad()
        loss = compute_loss(model, input_ids)
        loss.backward()
        optimizer.step()

        ppl = torch.exp(loss).item()
        print(f"  Step {step + 1}: Loss = {loss.item():.4f}, PPL = {ppl:.2f}")


def demonstrate_perplexity():
    """演示困惑度"""
    print("\n" + "=" * 60)
    print("困惑度演示")
    print("=" * 60)

    print("""
困惑度 (Perplexity) 衡量模型对文本的"惊讶程度"

公式: PPL = exp(平均交叉熵损失)

直观理解:
- PPL = 1: 模型总是完美预测 (不可能)
- PPL = 10: 模型平均在 10 个候选中犹豫
- PPL = 100: 模型非常不确定
- PPL = vocab_size: 模型完全随机猜测

典型值:
- GPT-2: ~20-30
- GPT-3: ~15-20
- LLaMA-7B: ~5-10 (在验证集上)
    """)

    # 模拟不同质量的模型
    vocab_size = 10000

    print("模拟不同困惑度:")
    for target_ppl in [2, 5, 10, 50, 100]:
        # 困惑度对应的平均损失
        loss = math.log(target_ppl)
        print(f"  PPL = {target_ppl:4d} → 平均损失 = {loss:.3f}")


def show_loss_computation():
    """展示损失计算的细节"""
    print("\n" + "=" * 60)
    print("损失计算详解")
    print("=" * 60)

    print("""
给定输入序列: "今天 天气 不错"

训练设置:
┌────────────────────────────────────────────────────────────┐
│  输入 (Input):  今天    天气                                 │
│  目标 (Target): 天气    不错                                 │
│                                                            │
│  模型预测每个位置下一个词的概率分布                           │
│  损失 = -log P(target | input)                             │
└────────────────────────────────────────────────────────────┘

位置 1:
  输入: "今天"
  目标: "天气"
  预测: P("天气" | "今天") = 0.3
  损失: -log(0.3) = 1.20

位置 2:
  输入: "今天 天气"
  目标: "不错"
  预测: P("不错" | "今天 天气") = 0.5
  损失: -log(0.5) = 0.69

总损失 = (1.20 + 0.69) / 2 = 0.945
困惑度 = exp(0.945) = 2.57
    """)


if __name__ == "__main__":
    demonstrate_next_token_prediction()
    demonstrate_training_step()
    demonstrate_perplexity()
    show_loss_computation()

    print("\n" + "=" * 60)
    print("关键要点:")
    print("=" * 60)
    print("""
1. 预训练目标：预测下一个词

2. 损失函数：交叉熵损失
   L = -Σ log P(x_t | x_{<t})

3. 困惑度 (PPL)：衡量模型质量
   PPL = exp(平均损失)
   PPL 越低越好

4. 训练循环：
   前向传播 → 计算损失 → 反向传播 → 更新参数

5. 预训练完成后，模型获得通用语言能力
    """)
