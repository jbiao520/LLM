# Encoder 流程图解

> 通过可视化图表理解 Transformer Encoder 的工作流程

## Encoder 整体架构

```mermaid
flowchart TB
    A["输入 Embedding"] --> B["位置编码"]
    B --> C["Multi-Head Attention"]
    C --> D["Add & Norm"]
    D --> E["Feed Forward"]
    E --> F["Add & Norm"]
    F --> G["输出"]

    subgraph 堆叠N次
        C
        D
        E
        F
    end

    style G fill:#c8e6c9
```

## 单层 Encoder 详细流程

```mermaid
flowchart TB
    A["输入 x"] --> B["Layer Norm"]
    B --> C["Multi-Head Self-Attention"]
    C --> D["残差连接<br/>x + Attention(x)"]

    D --> E["Layer Norm"]
    E --> F["FFN"]
    F --> G["残差连接<br/>x + FFN(x)"]

    G --> H["输出"]

    style H fill:#c8e6c9
```

## BERT Encoder 结构

```mermaid
flowchart TB
    A["[CLS] Token 1 Token 2 ... Token N [SEP]"] --> B["Token Embedding"]
    A --> C["Segment Embedding"]
    A --> D["Position Embedding"]

    B --> E["相加"]
    C --> E
    D --> E

    E --> F["Encoder x 12"]

    F --> G["[CLS] 输出<br/>句子表示"]
    F --> H["Token 输出<br/>用于分类/NER"]

    style G fill:#c8e6c9
    style H fill:#c8e6c9
```

## Self-Attention 在 Encoder 中

```mermaid
flowchart LR
    subgraph 双向注意力
        A["Token 1"] --> B["可以看到"]
        C["Token 2"] --> B
        D["Token 3"] --> B
        E["Token 4"] --> B
    end

    F["每个 token 可以<br/>看到所有其他 token"]

    style B fill:#c8e6c9
```

## Encoder 堆叠

```mermaid
flowchart TB
    A["输入"] --> L1["Layer 1"]
    L1 --> L2["Layer 2"]
    L2 --> L3["Layer 3"]
    L3 --> L4["..."]
    L4 --> L5["Layer N"]
    L5 --> B["输出"]

    C["每层提取不同级别的特征"]

    style B fill:#c8e6c9
```

## BERT 预训练任务

```mermaid
flowchart TB
    subgraph MLM
        A["The [MASK] sat on mat"] --> B["预测: cat"]
    end

    subgraph NSP
        C["句子 A: 我爱编程"] --> D{"是下一句?"}
        E["句子 B: Python很好"] --> D
    end

    style B fill:#c8e6c9
    style D fill:#c8e6c9
```

## 图解说明

### 关键特性

| 特性 | 说明 |
|------|------|
| 双向注意力 | 可以看到所有位置 |
| 并行处理 | 所有 token 同时处理 |
| 位置编码 | 注入位置信息 |

### 典型配置

| 模型 | 层数 | 隐藏维度 | 注意力头 |
|------|------|----------|----------|
| BERT-Base | 12 | 768 | 12 |
| BERT-Large | 24 | 1024 | 16 |

### 应用场景

- 文本分类
- 命名实体识别
- 问答系统
- 语义相似度
