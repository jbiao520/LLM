## Context

需要在 LLM 学习项目中添加 Embedding 相关的学习内容。Embedding 是理解大语言模型的基础，包括：
- **Word Embedding（词嵌入）**：将词语转换为向量表示
- **Position Embedding（位置嵌入）**：为序列中的位置生成向量表示

目标受众分为两类：
1. 无基础人群：需要通俗易懂的科普讲解
2. 有基础人群：需要深入的技术细节和数学原理

## Goals / Non-Goals

**Goals:**
- 创建 `topics/embedding/` 目录结构
- 为每个子主题提供两种版本的讲解文件
- 内容清晰、结构完整

**Non-Goals:**
- 不包含可执行代码示例（后续单独添加）
- 不涉及其他 embedding 类型（如 sentence embedding、image embedding）

## Decisions

### 1. 目录结构设计

```
topics/embedding/
├── README.md                          # 模块概览
├── word-embedding/
│   ├── beginner-guide.md              # 科普版本
│   ├── advanced-guide.md              # 深入版本
│   └── examples/                      # 代码示例
└── position-embedding/
    ├── beginner-guide.md              # 科普版本
    ├── advanced-guide.md              # 深入版本
    └── examples/                      # 代码示例
```

**理由**：扁平的子目录结构便于查找，文件命名清晰表明内容类型。

### 2. 内容风格定义

**科普版本 (beginner-guide.md)**：
- 使用类比和图解
- 避免数学公式
- 循序渐进的概念引入
- 适合零基础读者

**深入版本 (advanced-guide.md)**：
- 包含数学推导
- 引用原始论文
- 代码实现细节
- 适合有机器学习基础的读者

## Risks / Trade-offs

| 风险 | 缓解措施 |
|------|----------|
| 内容可能过于冗长 | 保持聚焦核心概念，适度控制篇幅 |
| 两个版本内容可能重复 | 明确区分目标受众，避免简单复制 |
