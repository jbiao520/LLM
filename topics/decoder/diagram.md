# Decoder 流程图解

> 通过可视化图表理解 Transformer Decoder 的工作流程

## Decoder 整体架构

```mermaid
flowchart TB
    A["输入 Embedding"] --> B["位置编码"]
    B --> C["Masked Self-Attention"]
    C --> D["Add & Norm"]
    D --> E["Cross-Attention<br/>(可选)"]
    E --> F["Add & Norm"]
    F --> G["Feed Forward"]
    G --> H["Add & Norm"]
    H --> I["输出"]

    style I fill:#c8e6c9
```

## 因果掩码机制

```mermaid
flowchart TB
    subgraph 掩码前
        A1["位置1"] --> B1["位置2"] --> C1["位置3"] --> D1["位置4"]
    end

    subgraph 掩码后
        A2["位置1: ✓ ✓ ✗ ✗"]
        B2["位置2: ✓ ✓ ✓ ✗"]
        C2["位置3: ✓ ✓ ✓ ✓"]
    end

    D["只能看到当前位置和之前的内容"]

    style A2 fill:#c8e6c9
    style B2 fill:#c8e6c9
    style C2 fill:#c8e6c9
```

## 自回归生成流程

```mermaid
flowchart LR
    A["输入: <s>"] --> B["生成: 你"]
    B --> C["输入: <s> 你"]
    C --> D["生成: 好"]
    D --> E["输入: <s> 你 好"]
    E --> F["生成: !"]

    G["逐步生成，每次一个 token"]

    style F fill:#c8e6c9
```

## KV Cache 工作原理

```mermaid
flowchart TB
    subgraph 第1步
        A1["输入: 你"] --> B1["计算 K1, V1<br/>缓存"]
    end

    subgraph 第2步
        A2["输入: 好"] --> B2["复用 K1,V1<br/>只计算 K2,V2"]
    end

    subgraph 第3步
        A3["输入: !"] --> B3["复用 K1-K2,V1-V2<br/>只计算 K3,V3"]
    end

    style B3 fill:#c8e6c9
```

## GPT Decoder 结构

```mermaid
flowchart TB
    A["Token IDs"] --> B["Token Embedding"]
    A --> C["Position Embedding"]

    B --> D["相加"]
    C --> D

    D --> E["Transformer Block x N"]
    E --> F["Layer Norm"]
    F --> G["线性层 → 词表"]

    G --> H["Softmax → 概率"]
    H --> I["采样下一个 token"]

    style I fill:#c8e6c9
```

## Cross-Attention (Encoder-Decoder)

```mermaid
flowchart TB
    subgraph Decoder
        A["Decoder 输入"] --> B["Masked Self-Attention"]
        B --> C["Cross-Attention"]
    end

    subgraph Encoder
        E["Encoder 输出<br/>K, V"]
    end

    C --> D["Q 来自 Decoder<br/>K,V 来自 Encoder"]

    D --> F["融合编码器信息"]

    style F fill:#c8e6c9
```

## 图解说明

### 关键特性

| 特性 | 说明 |
|------|------|
| 因果掩码 | 只能看到之前的内容 |
| 自回归 | 逐步生成 |
| KV Cache | 缓存避免重复计算 |

### 典型配置

| 模型 | 层数 | 隐藏维度 | 注意力头 |
|------|------|----------|----------|
| GPT-2 Small | 12 | 768 | 12 |
| GPT-2 Medium | 24 | 1024 | 16 |
| GPT-3 | 96 | 12288 | 96 |

### 应用场景

- 文本生成
- 代码补全
- 对话系统
- 翻译 (Encoder-Decoder)
