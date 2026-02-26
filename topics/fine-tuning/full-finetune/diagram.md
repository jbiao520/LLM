# 全参数微调流程图解

> 通过可视化图表理解全参数微调的完整工作流程

## 全参数微调总体流程

```mermaid
flowchart TB
    subgraph 准备阶段
        A[预训练模型<br/>如 LLaMA, GPT] --> B[准备任务数据<br/>标注/指令数据]
        B --> C[数据预处理<br/>分词/格式化]
    end

    subgraph 训练阶段
        C --> D[初始化优化器<br/>AdamW]
        D --> E[前向传播]
        E --> F[计算损失]
        F --> G[反向传播<br/>计算所有参数梯度]
        G --> H[优化器更新<br/>所有参数改变]
        H --> I{收敛?}
        I -->|否| E
        I -->|是| J[保存模型]
    end

    subgraph 部署阶段
        J --> K[评估验证]
        K --> L[模型部署]
    end

    style A fill:#e1f5fe
    style J fill:#c8e6c9
    style L fill:#fff9c4
```

## 全参数微调 vs LoRA 对比

```mermaid
flowchart LR
    subgraph 全参数微调
        direction TB
        FT1["更新所有参数<br/>W = W - η∇L"] --> FT2["模型完全改变"]
        FT2 --> FT3["保存完整模型<br/>~70GB"]
    end

    subgraph LoRA微调
        direction TB
        L1["冻结原参数<br/>只训练 A, B"] --> L2["模型部分改变<br/>W_new = W + BA"]
        L2 --> L3["保存适配器<br/>~200MB"]
    end

    style FT3 fill:#ffcdd2
    style L3 fill:#c8e6c9
```

## 显存使用分析

```mermaid
flowchart TB
    subgraph 7B模型全参数微调显存
        M1["模型权重 FP16<br/>14 GB"]
        M2["梯度<br/>14 GB"]
        M3["优化器状态<br/>28 GB"]
        M4["激活值<br/>~4 GB"]
        M1 --> SUM["总计 ~60 GB"]
        M2 --> SUM
        M3 --> SUM
        M4 --> SUM
    end

    style M3 fill:#ffcdd2
    style SUM fill:#fff9c4
```

## 训练优化技术

```mermaid
flowchart LR
    A[显存不足?] --> B{选择优化方案}

    B -->|单卡| C[梯度累积]
    B -->|计算够| D[梯度检查点]
    B -->|多卡| E[分布式训练]

    C --> F["有效批次 = batch × steps<br/>用时间换空间"]
    D --> G["重计算激活值<br/>节省30-50%显存"]
    E --> H["FSDP/DeepSpeed<br/>模型分片到多卡"]

    style F fill:#c8e6c9
    style G fill:#c8e6c9
    style H fill:#c8e6c9
```

## 灾难性遗忘问题

```mermaid
flowchart TB
    subgraph 问题
        A[预训练模型] --> B[全参数微调]
        B --> C["新任务能力强 ✓"]
        B --> D["原任务能力下降 ✗"]
    end

    subgraph 解决方案
        E[降低学习率] --> F["缓慢适应<br/>保留原知识"]
        G[混合数据训练] --> H["新旧数据混合<br/>持续回顾"]
        I[早停策略] --> J["监控验证集<br/>及时停止"]
    end

    style D fill:#ffcdd2
    style F fill:#c8e6c9
    style H fill:#c8e6c9
    style J fill:#c8e6c9
```

## 学习率调度

```mermaid
flowchart LR
    subgraph 调度过程
        A[0] -->|预热| B[warmup_steps]
        B -->|保持/上升| C[max_lr]
        C -->|余弦衰减| D[0]

        A -.-> E["学习率 = 0"]
        C -.-> F["学习率 = 2e-5"]
        D -.-> G["学习率 → 0"]
    end

    style C fill:#fff9c4
```

## 决策树：选择微调方法

```mermaid
flowchart TD
    Start["需要微调大模型"] --> Q1{"任务与预训练<br/>差异大吗?"}

    Q1 -->|"是<br/>(如医学、法律)"| Q2{"有充足<br/>计算资源?"}
    Q1 -->|"否<br/>(通用任务)"| LoRA["使用 LoRA/QLoRA<br/>推荐方案"]

    Q2 -->|"是"| Full["全参数微调<br/>追求最佳效果"]
    Q2 -->|"否"| Q3{"数据量 > 10K?"}

    Q3 -->|"是"| QLoRA["QLoRA<br/>高性价比"]
    Q3 -->|"否"| LoRA

    style Full fill:#bbdefb
    style LoRA fill:#c8e6c9
    style QLoRA fill:#c8e6c9

    Full --> Done["开始训练"]
    LoRA --> Done
    QLoRA --> Done
```

## 图解说明

### 关键概念

| 概念 | 说明 | 推荐值 |
|------|------|--------|
| 学习率 | 参数更新步长 | 1e-5 ~ 5e-5 |
| 批次大小 | 一次训练的样本数 | 尽可能大 |
| 梯度累积 | 模拟大批次 | 4-16 步 |
| 预热步数 | 学习率预热 | 总步数的 3-10% |

### 显存优化技巧

1. **梯度检查点**：省 30-50% 显存，慢 20%
2. **混合精度**：FP16/BF16，省 50% 模型显存
3. **梯度累积**：用时间换空间
4. **DeepSpeed ZeRO**：多卡分片，适合超大模型

### 何时选择全参数微调

- ✅ 任务与预训练差异大
- ✅ 有充足的计算资源
- ✅ 追求极致效果
- ✅ 数据量充足（>10K 样本）
