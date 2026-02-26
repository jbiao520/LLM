# Embedding 流程图解

> 通过可���化图表理解 Embedding 的工作流程

## Token Embedding 流程

```mermaid
flowchart LR
    A["输入文本<br/>'你好世界'"] --> B["分词<br/>[你, 好, 世, 界]"]
    B --> C["Token ID<br/>[101, 234, 567, 890]"]
    C --> D["查表<br/>Embedding Matrix"]
    D --> E["向量序列<br/>[v1, v2, v3, v4]"]

    style A fill:#e1f5fe
    style E fill:#c8e6c9
```

## Embedding 矩阵查表

```mermaid
flowchart TB
    subgraph 查表过程
        A["Token ID: 101"] --> B{"查表<br/>W[101]"}
        B --> C["向量<br/>[0.1, 0.2, ..., 0.768]"]
    end

    subgraph Embedding矩阵
        D["W[0]: [0.0, 0.0, ...]"]
        E["W[1]: [0.1, 0.2, ...]"]
        F["..."]
        G["W[101]: [0.1, 0.2, ..., 0.768]"]
        H["..."]
        I["W[V-1]: [0.9, 0.8, ...]"]
    end

    A -.-> G

    style C fill:#c8e6c9
```

## 完整 Embedding 流程

```mermaid
flowchart TB
    A["输入: '我爱AI'"] --> B["分词"]
    B --> C["Token IDs<br/>[50, 123, 789]"]

    C --> D["Token Embedding<br/>查表得到向量"]
    C --> E["Position Embedding<br/>位置编码"]

    D --> F["相加"]
    E --> F

    F --> G["最终 Embedding<br/>[3, 768]"]

    style G fill:#c8e6c9
```

## Word2Vec 训练流程

```mermaid
flowchart LR
    subgraph Skip-gram
        A["中心词<br/>'cat'"] --> B["预测上下文"]
        B --> C["'the'"]
        B --> D["'sat'"]
        B --> E["'on'"]
    end

    subgraph 训练目标
        F["最大化<br/>P(context|center)"]
    end

    A -.-> F

    style A fill:#bbdefb
```

## 位置编码流程

```mermaid
flowchart TB
    A["位置 0"] --> B["PE(0) = sin/cos<br/>[0.0, 1.0, 0.0, ...]"]
    C["位置 1"] --> D["PE(1) = sin/cos<br/>[0.84, 0.54, ...]"]
    E["位置 2"] --> F["PE(2) = sin/cos<br/>[0.91, -0.41, ...]"]

    B --> G["位置向量序列"]
    D --> G
    F --> G

    style G fill:#c8e6c9
```

## RoPE 旋转位置编码

```mermaid
flowchart LR
    A["向量 x"] --> B["旋转角度 θ<br/>由位置决定"]
    B --> C["x' = x * e^(iθ)"]

    subgraph 特点
        D["相对位置感知"]
        E["外推能力强"]
    end

    C --> D
    C --> E

    style C fill:#c8e6c9
```

## 图解说明

### 关键概念

| 概念 | 说明 |
|------|------|
| Token Embedding | 将词 ID 映射为向量 |
| Position Embedding | 编码位置信息 |
| Embedding Matrix | 学习的查找表 |
| 维度 | 通常 768-4096 |

### 流程要点

1. **分词**：文本 → Token IDs
2. **查表**：ID → 向量
3. **位置编码**：加入位置信息
4. **输出**：送入 Transformer
