"""
BPE (Byte Pair Encoding) 从零实现示例

这个示例展示了 BPE 算法的核心原理：
1. 从字符级别开始
2. 统计相邻对频率
3. 迭代合并最高频对
"""

from collections import defaultdict
import re


class SimpleBPE:
    """简化的 BPE 实现，用于教学目的"""

    def __init__(self, num_merges=100):
        self.num_merges = num_merges
        self.merges = []  # 存储合并规则
        self.vocab = set()

    def train(self, corpus):
        """
        在语料库上训练 BPE

        Args:
            corpus: 文本列表，如 ["hello world", "hello there"]
        """
        # 第一步：将语料转换为单词频率字典
        # 每个单词拆分为字符，用 </w> 标记词尾
        word_freqs = defaultdict(int)
        for text in corpus:
            words = text.split()
            for word in words:
                # 将单词拆分为字符序列
                chars = ' '.join(list(word)) + ' </w>'
                word_freqs[chars] += 1

        print(f"初始词汇表大小: {len(self._get_vocab(word_freqs))}")
        print(f"初始词汇表样例: {list(self._get_vocab(word_freqs))[:10]}")

        # 迭代合并
        for i in range(self.num_merges):
            # 统计所有相邻对的频率
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                print(f"第 {i} 轮: 没有更多对可以合并")
                break

            # 找到最高频的对
            best_pair = max(pairs, key=pairs.get)
            freq = pairs[best_pair]

            if freq < 2:
                print(f"第 {i} 轮: 最高频对只出现 {freq} 次，停止合并")
                break

            # 合并该对
            word_freqs = self._merge_pair(word_freqs, best_pair)
            self.merges.append(best_pair)

            if i < 10 or i % 10 == 0:
                print(f"第 {i} 轮: 合并 {best_pair} (频率: {freq})")

        # 更新词汇表
        self.vocab = self._get_vocab(word_freqs)
        print(f"\n最终词汇表大小: {len(self.vocab)}")
        print(f"合并规则数量: {len(self.merges)}")

        return self

    def _get_vocab(self, word_freqs):
        """获取当前所有 token"""
        vocab = set()
        for word in word_freqs:
            tokens = word.split()
            vocab.update(tokens)
        return vocab

    def _get_pairs(self, word_freqs):
        """统计所有相邻对的频率"""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += freq
        return pairs

    def _merge_pair(self, word_freqs, pair):
        """合并指定的对"""
        new_word_freqs = defaultdict(int)
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] += freq

        return new_word_freqs

    def encode(self, text):
        """
        使用训练好的 BPE 编码文本

        Args:
            text: 要编码的文本

        Returns:
            token 列表
        """
        tokens = []
        words = text.split()

        for word in words:
            # 拆分为字符
            word_tokens = list(word) + ['</w>']

            # 按顺序应用合并规则
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == merge[0] and word_tokens[i + 1] == merge[1]:
                        word_tokens = word_tokens[:i] + [merge[0] + merge[1]] + word_tokens[i + 2:]
                    else:
                        i += 1

            tokens.extend(word_tokens)

        return tokens

    def decode(self, tokens):
        """
        将 tokens 解码为文本

        Args:
            tokens: token 列表

        Returns:
            解码后的文本
        """
        text = ''.join(tokens)
        # 移除 </w> 标记并恢复空格
        text = text.replace('</w>', ' ')
        return text.strip()


def demo_bpe_training():
    """演示 BPE 训练过程"""
    print("=" * 60)
    print("BPE 训练演示")
    print("=" * 60)

    # 使用一个小语料库
    corpus = [
        "the quick brown fox",
        "the quick brown dog",
        "the lazy dog",
        "the lazy fox",
        "a quick brown fox jumps",
        "the fox is quick",
        "the dog is lazy",
    ]

    print("\n语料库:")
    for sentence in corpus:
        print(f"  - {sentence}")

    print("\n" + "-" * 60)
    print("开始训练...")
    print("-" * 60 + "\n")

    bpe = SimpleBPE(num_merges=20)
    bpe.train(corpus)

    print("\n" + "-" * 60)
    print("合并规则:")
    print("-" * 60)
    for i, merge in enumerate(bpe.merges[:10]):
        print(f"  {i + 1}. {merge[0]} + {merge[1]} → {merge[0] + merge[1]}")
    if len(bpe.merges) > 10:
        print(f"  ... 共 {len(bpe.merges)} 条规则")


def demo_bpe_encoding():
    """演示 BPE 编码过程"""
    print("\n" + "=" * 60)
    print("BPE 编码演示")
    print("=" * 60)

    # 训练
    corpus = [
        "hello world",
        "hello there",
        "world hello",
        "hi there",
        "hello hello",
    ]

    bpe = SimpleBPE(num_merges=15)
    bpe.train(corpus)

    # 编码测试
    test_texts = [
        "hello world",
        "hello there world",
        "hi world",
    ]

    print("\n" + "-" * 60)
    print("编码测试:")
    print("-" * 60)

    for text in test_texts:
        tokens = bpe.encode(text)
        decoded = bpe.decode(tokens)
        print(f"\n原文: {text}")
        print(f"Tokens: {tokens}")
        print(f"解码: {decoded}")
        print(f"一致: {text == decoded}")


def show_pair_frequency():
    """展示相邻对频率统计"""
    print("\n" + "=" * 60)
    print("相邻对频率统计")
    print("=" * 60)

    word_freqs = defaultdict(int)
    word_freqs['h e l l o </w>'] = 5
    word_freqs['h e l p </w>'] = 3
    word_freqs['h e l l </w>'] = 2

    print("\n单词频率:")
    for word, freq in word_freqs.items():
        print(f"  {word}: {freq}")

    # 统计对
    pairs = defaultdict(int)
    for word, freq in word_freqs.items():
        tokens = word.split()
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += freq

    print("\n相邻对频率 (排序):")
    for pair, freq in sorted(pairs.items(), key=lambda x: -x[1]):
        print(f"  {pair}: {freq}")


if __name__ == "__main__":
    demo_bpe_training()
    demo_bpe_encoding()
    show_pair_frequency()

    print("\n" + "=" * 60)
    print("关键要点:")
    print("=" * 60)
    print("""
1. BPE 从字符级别开始，逐步合并高频相邻对
2. 合并顺序基于频率，最常见的对最先合并
3. 训练后得到固定的合并规则，用于编码新文本
4. 编码是确定性的：同样的文本总是得到同样的 tokens
5. </w> 标记用于区分词尾，便于解码时恢复空格
    """)
