"""
tiktoken 示例 - GPT 系列模型的分词器

tiktoken 是 OpenAI 开发的 BPE 分词器，用于 GPT-3.5、GPT-4 等模型。

安装: pip install tiktoken
"""

import tiktoken


def basic_usage():
    """tiktoken 基础用法"""
    print("=" * 60)
    print("tiktoken 基础用法")
    print("=" * 60)

    # 加载 GPT-4 的分词器
    enc = tiktoken.encoding_for_model("gpt-4")
    print(f"模型: gpt-4")
    print(f"词汇表大小: {enc.n_vocab}")
    print(f"最大 token 长度: {enc.max_token_value}")

    # 编码文本
    text = "Hello, 世界! This is a test."
    tokens = enc.encode(text)
    print(f"\n原文: {text}")
    print(f"Tokens: {tokens}")

    # 解码
    decoded = enc.decode(tokens)
    print(f"解码: {decoded}")

    # 查看每个 token 的文本
    print("\n逐 token 分析:")
    for token in tokens:
        token_text = enc.decode([token])
        print(f"  ID: {token:6d} → '{token_text}'")


def compare_models():
    """比较不同模型的分词器"""
    print("\n" + "=" * 60)
    print("不同模型的分词器对比")
    print("=" * 60)

    models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]
    text = "The quick brown fox jumps over the lazy dog. 快速的棕色狐狸跳过懒狗。"

    print(f"\n文本: {text}\n")

    for model in models:
        try:
            enc = tiktoken.encoding_for_model(model)
            tokens = enc.encode(text)
            print(f"{model}:")
            print(f"  Token 数量: {len(tokens)}")
            print(f"  前10个 tokens: {tokens[:10]}")
        except Exception as e:
            print(f"{model}: 不支持 ({e})")
        print()


def token_cost_estimation():
    """Token 成本估算"""
    print("=" * 60)
    print("Token 成本估算")
    print("=" * 60)

    enc = tiktoken.encoding_for_model("gpt-4")

    # 不同类型的文本
    texts = {
        "英文短文": "The quick brown fox jumps over the lazy dog.",
        "中文短文": "那只敏捷的棕色狐狸跳过了懒狗。",
        "中英混合": "Hello 世界! This is a test 测试。",
        "代码": """
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""",
        "长文本": "This is a sample text. " * 100,
    }

    # GPT-4 价格 (2024 年参考)
    price_per_1k_input = 0.03  # $0.03 / 1K tokens (input)
    price_per_1k_output = 0.06  # $0.06 / 1K tokens (output)

    print(f"\n价格参考: ${price_per_1k_input}/1K input tokens, ${price_per_1k_output}/1K output tokens")
    print("-" * 60)

    for name, text in texts.items():
        tokens = enc.encode(text)
        char_count = len(text)
        token_count = len(tokens)
        ratio = char_count / token_count if token_count > 0 else 0
        cost = (token_count / 1000) * price_per_1k_input

        print(f"\n{name}:")
        print(f"  字符数: {char_count}")
        print(f"  Token 数: {token_count}")
        print(f"  字符/Token 比率: {ratio:.2f}")
        print(f"  估算成本: ${cost:.4f}")


def token_manipulation():
    """Token 操作技巧"""
    print("\n" + "=" * 60)
    print("Token 操作技巧")
    print("=" * 60)

    enc = tiktoken.encoding_for_model("gpt-4")

    # 1. 计算精确的 token 数量
    text = "这是一个用于测试的文本。"
    token_count = len(enc.encode(text))
    print(f"1. Token 计数: '{text}' → {token_count} tokens")

    # 2. 截断到指定 token 数量
    long_text = "This is a very long text. " * 50
    max_tokens = 20
    tokens = enc.encode(long_text)
    truncated_tokens = tokens[:max_tokens]
    truncated_text = enc.decode(truncated_tokens)
    print(f"\n2. 截断到 {max_tokens} tokens:")
    print(f"   原文长度: {len(enc.encode(long_text))} tokens")
    print(f"   截断后: {len(truncated_tokens)} tokens")
    print(f"   截断文本: '{truncated_text}...'")

    # 3. 检查是否包含特殊 token
    print(f"\n3. 特殊 tokens:")
    print(f"   <|endoftext|>: {enc.encode('<|endoftext|>', allowed_special='all')}")
    print(f"   <|fim_prefix|>: {enc.encode('<|fim_prefix|>', allowed_special='all')}")

    # 4. 处理边界情况
    print(f"\n4. 边界情况:")
    edge_cases = [
        "",  # 空字符串
        " ",  # 单空格
        "   ",  # 多空格
        "\n",  # 换行
        "\t",  # 制表符
        "a",  # 单字符
        "🎉",  # emoji
        "🎉🎉🎉",  # 多个 emoji
    ]
    for case in edge_cases:
        tokens = enc.encode(case)
        print(f"   '{repr(case)}' → {tokens}")


def analyze_tokenization_artifacts():
    """分析分词器的人为产物"""
    print("\n" + "=" * 60)
    print("分词器人为产物分析")
    print("=" * 60)

    enc = tiktoken.encoding_for_model("gpt-4")

    # 1. 数字分词问题
    print("1. 数字分词:")
    numbers = ["123", "1234", "12345", "123456", "1234567890"]
    for num in numbers:
        tokens = enc.encode(num)
        token_texts = [enc.decode([t]) for t in tokens]
        print(f"   {num} → {tokens} → {token_texts}")

    # 2. 空格影响
    print("\n2. 空格的影响:")
    pairs = [
        ("hello", " hello"),
        ("world", " world"),
        ("the", " the"),
    ]
    for no_space, with_space in pairs:
        tokens_no = enc.encode(no_space)
        tokens_with = enc.encode(with_space)
        print(f"   '{no_space}' → {tokens_no}")
        print(f"   '{with_space}' → {tokens_with}")
        print()

    # 3. 大小写影响
    print("3. 大小写影响:")
    cases = [("hello", "Hello", "HELLO"), ("the", "The", "THE")]
    for group in cases:
        for word in group:
            tokens = enc.encode(word)
            print(f"   '{word}' → {tokens}")
        print()

    # 4. 中英混合
    print("4. 中英混合:")
    mixed = [
        "Hello世界",
        "Hello 世界",
        "Hello，世界",
        "Hello, 世界",
    ]
    for text in mixed:
        tokens = enc.encode(text)
        print(f"   '{text}' → {len(tokens)} tokens: {tokens}")


def batch_encoding():
    """批量编码"""
    print("\n" + "=" * 60)
    print("批量编码")
    print("=" * 60)

    enc = tiktoken.encoding_for_model("gpt-4")

    texts = [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]

    # 批量编码
    all_tokens = [enc.encode(text) for text in texts]
    print("批量编码结果:")
    for text, tokens in zip(texts, all_tokens):
        print(f"  '{text}' → {tokens}")

    # 计算总 token 数
    total_tokens = sum(len(tokens) for tokens in all_tokens)
    print(f"\n总 token 数: {total_tokens}")


def available_encodings():
    """查看可用的编码"""
    print("\n" + "=" * 60)
    print("可用的编码")
    print("=" * 60)

    encodings = tiktoken.list_encoding_names()
    print(f"可用编码: {encodings}")

    for name in encodings:
        try:
            enc = tiktoken.get_encoding(name)
            print(f"\n{name}:")
            print(f"  词汇表大小: {enc.n_vocab}")
        except Exception as e:
            print(f"\n{name}: 加载失败 ({e})")


if __name__ == "__main__":
    basic_usage()
    compare_models()
    token_cost_estimation()
    token_manipulation()
    analyze_tokenization_artifacts()
    batch_encoding()
    available_encodings()

    print("\n" + "=" * 60)
    print("关键要点:")
    print("=" * 60)
    print("""
1. tiktoken 是 GPT 系列的官方分词器
2. 不同模型可能使用不同的编码（cl100k_base, o200k_base 等）
3. Token 数量影响 API 成本，优化 prompt 可以省钱
4. 空格、大小写、数字都可能影响 tokenization
5. 使用 encode() 编码，decode() 解码
6. 使用 allowed_special='all' 处理特殊 token
    """)
