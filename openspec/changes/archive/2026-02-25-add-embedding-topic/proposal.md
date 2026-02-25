## Why

需要在项目中添加 Embedding（嵌入）相关的学习内容，这是理解大语言模型的基��知识。目前 `topics/` 目录下缺少这部分内容。Embedding 是连接自然语言和神经网络的核心概念，包括词嵌入和位置嵌入两个重要主题。

## What Changes

- 在 `topics/` 下新增 `embedding/` 目录
- 创建 `word-embedding/` 子目录，包含两个版本的讲解文件
- 创建 `position-embedding/` 子目录，包含两个版本的讲解文件
- 每个子目录包含 `examples/` 代码示例目录

## Capabilities

### New Capabilities

- `embedding-topic`: Embedding 学习模块，包含词嵌入和位置嵌入两个子主题，每个主题提供科普版和深入版两种讲解风格

### Modified Capabilities

（无现有能力需要修改）

## Impact

- 新增 `topics/embedding/` 目录及其子目录
- 新增 4 个 Markdown 讲解文件
- 新增 `topics/embedding/README.md` 概览文件
