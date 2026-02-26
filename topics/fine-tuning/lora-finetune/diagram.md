# LoRA 微调流程图解

> 通过可视化图表理解 LoRA 微调的完整工作流程

## LoRA 核心原理图

```mermaid
flowchart TB
    subgraph 原始模型
        W["权重矩阵 W<br/>(冻结不训练)"]
    end

    subgraph LoRA适配器
        A["矩阵 A<br/>(r × k)<br/>初始化为0"]
        B["矩阵 B<br/>(d × r)<br/>随机初始化"]
    end

    subgraph 输出
        SUM["W + α/r × BA"]
    end

    Input["输入 x"] --> W
    Input --> A
    A --> B
    W --> SUM
    B --> SUM
    SUM --> Output["输出 h"]

    style W fill:#f9f,stroke:#333
    style A fill:#9f9,stroke:#333
    style B fill:#9f9,stroke:#333
```

## LoRA 微调完整流程

```mermaid
flowchart LR
    subgraph 准备阶段
        A[加载预训练模型] --> B[冻结所有参数]
        B --> C[添加LoRA适配器]
    end

    subgraph 训练阶段
        C --> D[前向传播]
        D --> E[计算损失]
        E --> F[反向传播<br/>只更新LoRA参数]
        F --> G{损失收敛?}
        G -->|否| D
        G -->|是| H[保存LoRA权重]
    end

    subgraph 部署阶段
        H --> I[合并LoRA到基础模型<br/>或单独加载]
        I --> J[部署推理]
    end

    style A fill:#e1f5fe
    style H fill:#c8e6c9
    style J fill:#fff9c4
```

## LoRA vs 全参数微调对比

```mermaid
flowchart TB
    subgraph 全参数微调
        FT1["所有参数参与训练"] --> FT2["需要大量显存"]
        FT2 --> FT3["保存完整模型副本"]
        FT3 --> FT4["每个任务一个<br/>完整模型文件"]
    end

    subgraph LoRA微调
        L1["只训练1%参数"] --> L2["显存需求大幅降低"]
        L2 --> L3["只保存小适配器"]
        L3 --> L4["多个任务共享<br/>同一基础模型"]
    end

    style FT1 fill:#ffcdd2
    style FT4 fill:#ffcdd2
    style L1 fill:#c8e6c9
    style L4 fill:#c8e6c9
```

## QLoRA 量化流程

```mermaid
flowchart LR
    subgraph 加载阶段
        A[16-bit 原始模型] --> B[量化为 4-bit NF4]
        B --> C[双重量化<br/>压缩量化常数]
    end

    subgraph 训练阶段
        C --> D[分页加载到GPU]
        D --> E[按需反量化计算]
        E --> F[只训练LoRA参数<br/>保持4-bit存储]
    end

    subgraph 推理阶段
        F --> G[合并LoRA权重]
        G --> H[可选: 保持4-bit<br/>或转回16-bit]
    end

    style A fill:#bbdefb
    style B fill:#e1bee7
    style F fill:#c8e6c9
```

## LoRA 参数选择决策树

```mermaid
flowchart TD
    Start["选择 LoRA Rank (r)"] --> Q1{"任务复杂度?"}

    Q1 -->|"简单<br/>(风格/分类)"| R1["r = 1-4"]
    Q1 -->|"中等<br/>(指令微调)"| R2["r = 8-16"]
    Q1 -->|"复杂<br/>(新知识)"| R3["r = 32-64"]

    R1 --> Alpha["设置 α = 2r 或 16"]
    R2 --> Alpha
    R3 --> Alpha

    Alpha --> Modules{"目标模块?"}
    Modules -->|"最小干预"| M1["q_proj, v_proj"]
    Modules -->|"标准配置"| M2["q,k,v,o_proj"]
    Modules -->|"最大能力"| M3["所有线性层"]

    M1 --> Done["开始训练"]
    M2 --> Done
    M3 --> Done

    style Start fill:#e1f5fe
    style Done fill:#c8e6c9
```

## 图解说明

### 核心原理图
- **蓝色框**：原始冻结权重，不参与训练
- **绿色框**：LoRA 可训练参数，只有原模型的 0.1%-1%
- **数学公式**：$W_{new} = W + \frac{\alpha}{r} BA$

### 关键概念

| 概念 | 说明 | 推荐值 |
|------|------|--------|
| Rank (r) | 低秩维度，决定表达能力 | 8-16 |
| Alpha (α) | 缩放因子，控制影响强度 | 16-32 |
| Target Modules | 应用 LoRA 的层 | q_proj, v_proj |

### 流程要点

1. **准备**：加载模型 → 冻结参数 → 注入 LoRA
2. **训练**：只更新 $A$ 和 $B$ 矩阵，其他不变
3. **部署**：可合并到原模型，或单独加载适配器
