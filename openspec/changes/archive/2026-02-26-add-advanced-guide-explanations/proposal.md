## Why

当前各主题的 `advanced-guide.md` 含有较多数学公式，但对公式含义、变量和直觉解释不足，数学基础薄弱的读者理解门槛高。补充详细中文解释可显著降低理解成本，让读者更快把公式与概念建立联系。

## What Changes

- 为每个 topic 下的 `advanced-guide.md` 中出现的数学公式补充详细中文解释，包括变量含义、公式直觉/作用以及关键推导思路。
- 统一公式解释的呈现方式，使读者能快速定位“公式 → 含义 → 直觉”的对应关系。

## Capabilities

### New Capabilities
- `<none>`

### Modified Capabilities
- `activation-topic`: 深入版文档中的公式提供详细中文解释，面向数学基础薄弱读者
- `attention-topic`: 深入版文档中的公式提供详细中文解释，面向数学基础薄弱读者
- `decoder-topic`: 深入版文档中的公式提供详细中文解释，面向数学基础薄弱读者
- `embedding-topic`: 深入版文档中的公式提供详细中文解释，面向数学基础薄弱读者
- `encoder-topic`: 深入版文档中的公式提供详细中文解释，面向数学基础薄弱读者
- `ffn-topic`: 深入版文档中的公式提供详细中文解释，面向数学基础薄弱读者
- `fine-tuning-full-topic`: 深入版文档中的公式提供详细中文解释，面向数学基础薄弱读者
- `fine-tuning-lora-topic`: 深入版文档中的公式提供详细中文解释，面向数学基础薄弱读者
- `inference-acceleration-topic`: 深入版文档中的公式提供详细中文解释，面向数学基础薄弱读者
- `inference-quantization-topic`: 深入版文档中的公式提供详细中文解释，面向数学基础薄弱读者
- `moe-topic`: 深入版文档中的公式提供详细中文解释，面向数学基础薄弱读者
- `normalization-topic`: 深入版文档中的公式提供详细中文解释，面向数学基础薄弱读者

## Impact

- 文档内容：`topics/**/advanced-guide.md`
- 需求规范：对应 topic 的 openspec 规格文件（新增“公式解释”要求）
