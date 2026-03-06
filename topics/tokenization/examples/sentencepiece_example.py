"""
SentencePiece 示例 - LLaMA 等开源模型使用的分词器

SentencePiece 是一个语言无关的分词器，支持 BPE 和 Unigram 算法。
被 LLaMA、Mistral 等模型使用。

安装: pip install sentencepiece
"""

import sentencepiece as spm
import os
import tempfile


def train_sentencepiece_model():
    """训练一个简单的 SentencePiece 模型"""
    print("=" * 60)
    print("训练 SentencePiece 模型")
    print("=" * 60)

    # 创建临时训练文件
    corpus = """
    The quick brown fox jumps over the lazy dog.
    The lazy dog sleeps all day long.
    A quick brown dog runs in the park.
    The fox and the dog are good friends.
    Hello world, this is a test.
    Machine learning is fascinating.
    Natural language processing is a subset of AI.
    Transformers have revolutionized NLP.
    """ * 10  # 重复以增加语料

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(corpus)
        temp_file = f.name

    model_prefix = tempfile.mktemp()

    try:
        # 训练模型
        # 参数说明:
        # - input: 输入文件
        # - model_prefix: 输出模型前缀
        # - vocab_size: 词汇表大小
        # - model_type: 'bpe' 或 'unigram'
        # - character_coverage: 字符覆盖率 (1.0 = 100%)
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=100,
            model_type='bpe',
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )

        print(f"模型训练完成!")
        print(f"模型文件: {model_prefix}.model")
        print(f"词汇表文件: {model_prefix}.vocab")

        # 加载并测试模型
        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")

        print(f"\n词汇表大小: {sp.get_piece_size()}")
        print(f"特殊 tokens:")
        print(f"  PAD ID: {sp.pad_id()} ({sp.id_to_piece(sp.pad_id())})")
        print(f"  UNK ID: {sp.unk_id()} ({sp.id_to_piece(sp.unk_id())})")
        print(f"  BOS ID: {sp.bos_id()} ({sp.id_to_piece(sp.bos_id())})")
        print(f"  EOS ID: {sp.eos_id()} ({sp.id_to_piece(sp.eos_id())})")

        # 展示部分词汇
        print(f"\n前20个 tokens:")
        for i in range(20):
            print(f"  {i}: '{sp.id_to_piece(i)}'")

        return sp, model_prefix

    finally:
        os.unlink(temp_file)


def basic_operations(sp):
    """基础操作"""
    print("\n" + "=" * 60)
    print("SentencePiece 基础操作")
    print("=" * 60)

    text = "Hello world, this is a test."

    # 编码为 pieces (字符串)
    pieces = sp.encode_as_pieces(text)
    print(f"原文: {text}")
    print(f"Pieces: {pieces}")

    # 编码为 IDs (整数)
    ids = sp.encode_as_ids(text)
    print(f"IDs: {ids}")

    # 解码
    decoded_from_pieces = sp.decode_pieces(pieces)
    decoded_from_ids = sp.decode_ids(ids)
    print(f"从 pieces 解码: {decoded_from_pieces}")
    print(f"从 IDs 解码: {decoded_from_ids}")

    # 单个 token 操作
    print(f"\n单个 token 操作:")
    print(f"  '▁Hello' → ID: {sp.piece_to_id('▁Hello')}")
    print(f"  ID 5 → Piece: '{sp.id_to_piece(5)}'")


def compare_bpe_unigram():
    """比较 BPE 和 Unigram 模型"""
    print("\n" + "=" * 60)
    print("BPE vs Unigram 对比")
    print("=" * 60)

    corpus = """
    The cat sat on the mat.
    The dog sat on the log.
    The rat sat on the bat.
    The cat and the dog are friends.
    """ * 20

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(corpus)
        temp_file = f.name

    results = {}

    for model_type in ['bpe', 'unigram']:
        model_prefix = tempfile.mktemp()

        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=50,
            model_type=model_type,
            character_coverage=1.0,
        )

        sp = spm.SentencePieceProcessor()
        sp.load(f"{model_prefix}.model")

        test_text = "The cat sat on the mat."
        pieces = sp.encode_as_pieces(test_text)

        results[model_type] = {
            'pieces': pieces,
            'count': len(pieces),
            'vocab_size': sp.get_piece_size(),
        }

        # 清理
        os.unlink(f"{model_prefix}.model")
        os.unlink(f"{model_prefix}.vocab")

    os.unlink(temp_file)

    print(f"\n测试文本: '{test_text}'")
    print(f"\nBPE:")
    print(f"  Pieces: {results['bpe']['pieces']}")
    print(f"  Token 数: {results['bpe']['count']}")

    print(f"\nUnigram:")
    print(f"  Pieces: {results['unigram']['pieces']}")
    print(f"  Token 数: {results['unigram']['count']}")


def sentencepiece_features():
    """SentencePiece 特性演示"""
    print("\n" + "=" * 60)
    print("SentencePiece 特性")
    print("=" * 60)

    corpus = "Hello world. 你好世界. こんにちは世界."

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(corpus * 50)
        temp_file = f.name

    model_prefix = tempfile.mktemp()

    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=100,
        model_type='bpe',
        character_coverage=0.9995,  # 高覆盖率支持多语言
    )

    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    # 1. 空格处理
    print("1. 空格处理 (使用 ▁ 符号):")
    texts = ["Hello world", "Hello  world", "Hello   world"]
    for text in texts:
        pieces = sp.encode_as_pieces(text)
        print(f"   '{text}' → {pieces}")

    # 2. 多语言支持
    print("\n2. 多语言支持:")
    texts = [
        "Hello world",
        "你好世界",
        "Hello 世界",
        "こんにちは",
    ]
    for text in texts:
        pieces = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)
        print(f"   '{text}' → {len(pieces)} tokens")
        print(f"      Pieces: {pieces}")

    # 3. 批处理
    print("\n3. 批处理:")
    batch_texts = ["Hello", "World", "Test"]
    batch_ids = sp.encode(batch_texts)
    for text, ids in zip(batch_texts, batch_ids):
        print(f"   '{text}' → {ids}")

    # 4. 设置阈值
    print("\n4. 设置最小/最大 token 长度:")
    sp.set_encode_extra_options("bos:eos")  # 添加 BOS 和 EOS
    text = "Hello world"
    ids = sp.encode(text)
    print(f"   带 BOS/EOS: '{text}' → {ids}")
    print(f"   BOS token: '{sp.id_to_piece(sp.bos_id())}'")
    print(f"   EOS token: '{sp.id_to_piece(sp.eos_id())}'")

    # 清理
    os.unlink(temp_file)
    os.unlink(f"{model_prefix}.model")
    os.unlink(f"{model_prefix}.vocab")


def load_pretrained_tokenizer():
    """加载预训练的 tokenizer (如 LLaMA)"""
    print("\n" + "=" * 60)
    print("加载预训练 Tokenizer 示例")
    print("=" * 60)

    print("""
# 加载 LLaMA tokenizer
sp = spm.SentencePieceProcessor()
sp.load("llama/tokenizer.model")

# 编码
text = "Hello, world!"
pieces = sp.encode_as_pieces(text)
ids = sp.encode_as_ids(text)

# 解码
decoded = sp.decode_ids(ids)

# 查看词汇表
print(f"Vocab size: {sp.get_piece_size()}")
for i in range(10):
    print(f"  {i}: {sp.id_to_piece(i)}")
    """)

    print("\n注意: 需要下载 LLaMA 的 tokenizer.model 文件")


def sampling_tokenization():
    """采样式分词 (仅 Unigram)"""
    print("\n" + "=" * 60)
    print("采样式分词 (Subword Regularization)")
    print("=" * 60)

    corpus = "The quick brown fox jumps over the lazy dog." * 30

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(corpus)
        temp_file = f.name

    model_prefix = tempfile.mktemp()

    # 使用 Unigram 模型（支持采样）
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=100,
        model_type='unigram',
        character_coverage=1.0,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    text = "the quick brown fox"

    print(f"文本: '{text}'")
    print(f"\n确定性编码 (nbest_size=1):")
    for _ in range(3):
        pieces = sp.encode(text, out_type=str, nbest_size=1)
        print(f"  {pieces}")

    print(f"\n采样编码 (nbest_size=-1, alpha=0.1):")
    for _ in range(3):
        pieces = sp.encode(text, out_type=str, nbest_size=-1, alpha=0.1)
        print(f"  {pieces}")

    # 清理
    os.unlink(temp_file)
    os.unlink(f"{model_prefix}.model")
    os.unlink(f"{model_prefix}.vocab")


if __name__ == "__main__":
    sp, model_prefix = train_sentencepiece_model()
    basic_operations(sp)
    compare_bpe_unigram()
    sentencepiece_features()
    load_pretrained_tokenizer()
    sampling_tokenization()

    # 清理
    if os.path.exists(f"{model_prefix}.model"):
        os.unlink(f"{model_prefix}.model")
    if os.path.exists(f"{model_prefix}.vocab"):
        os.unlink(f"{model_prefix}.vocab")

    print("\n" + "=" * 60)
    print("关键要点:")
    print("=" * 60)
    print("""
1. SentencePiece 是语言无关的分词器，不依赖预分词
2. 使用 ▁ (下划线) 表示空格/词首
3. 支持 BPE 和 Unigram 两种算法
4. Unigram 支持采样式分词 (Subword Regularization)
5. character_coverage 参数控制字符覆盖率
6. 被许多开源模型使用 (LLaMA, Mistral, T5 等)
    """)
