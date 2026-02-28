## ADDED Requirements

### Requirement: PDF 构建脚本可执行

系统 SHALL 提供一个 Node.js 脚本 `scripts/build-pdf.mjs`，可通过 `node scripts/build-pdf.mjs` 命令执行。

#### Scenario: 成功执行构建脚本
- **WHEN** ���户运行 `node scripts/build-pdf.mjs`
- **THEN** 脚本在 `output/LLM学习指南.pdf` 生成 PDF 文件

#### Scenario: 缺少依赖时提示安装
- **WHEN** 用户运行脚本但缺少必要依赖
- **THEN** 脚本输出明确的错误信息，提示运行 `npm install`

---

### Requirement: 按配置文件组织章节顺序

系统 SHALL 读取 `scripts/pdf-config.yaml` 配置文件，按其中定义的顺序组织 PDF 章节。

#### Scenario: 读取章节配置
- **WHEN** 脚本执行时
- **THEN** 从 pdf-config.yaml 读取章节列表和输出路径

#### Scenario: 配置文件缺失时报错
- **WHEN** pdf-config.yaml 文件不存在
- **THEN** 脚本输出错误信息并退出

---

### Requirement: 渲染 Mermaid 图表

系统 SHALL 将 Markdown 中的 Mermaid 代码块渲染为可视化图表，并配置足够的文本大小限制以支持复杂的中文图表。

#### Scenario: 渲染 flowchart
- **WHEN** MD 文件包含 ` ```mermaid ... ``` ` 代码块
- **THEN** PDF 中显示渲染后的流程图，而非原始代码

#### Scenario: 渲染大文本图表
- **WHEN** Mermaid 图表节点包含较长的中文文本
- **THEN** 图表正常渲染，不抛出 "Maximum text size in diagram exceeded" 错误

---

### Requirement: 渲染 LaTeX 数学公式

系统 SHALL 将 LaTeX 数学公式渲染为数学符号。

#### Scenario: 渲染行间公式
- **WHEN** MD 文件包含 `$$...$$` 公式
- **THEN** PDF 中显示渲染后的数学公式

---

### Requirement: 代码语法高亮

系统 SHALL 对代码块进行语法高亮显示。

#### Scenario: Python 代码高亮
- **WHEN** MD 文件包含 ` ```python ... ``` ` 代码块
- **THEN** PDF 中显示语法高亮的 Python 代码

---

### Requirement: 代码示例作为章节附录

系统 SHALL 将每个章节的 `examples/*.py` 文件作为附录放在该章末尾。

#### Scenario: 章节包含代码附录
- **WHEN** 章节配置中包含 examples 目录
- **THEN** PDF 在该章末尾添加"附录：代码示例"部分，包含所有 Python 文件

---

### Requirement: 目录只到章级别

系统 SHALL 生成只包含章节级别（不包含节和小节）的目录。

#### Scenario: 生成章节目录
- **WHEN** PDF 生成时
- **THEN** 目录页只显示第1章、第2章等章级标题，不显示子标题

---

### Requirement: 页脚只显示页码

系统 SHALL 在每页页脚居中显示页码，不显示其他信息。

#### Scenario: 页脚格式
- **WHEN** PDF 页面生成时
- **THEN** 页脚居中显示页码（如 "— 5 —"）

---

### Requirement: 电子书风格排版

系统 SHALL 使用电子书风格的 CSS 样式，包括：
- 合适的字体和行高
- 舒适的页边距
- 清晰的标题层级

#### Scenario: 中文电子书风格
- **WHEN** PDF 生成时
- **THEN** 使用适合中文阅读的字体、行高（1.8）和页边距（25mm）

---

### Requirement: 支持中文

系统 SHALL 正确渲染中文字符，不出现乱码或缺失。

#### Scenario: 中文内容渲染
- **WHEN** MD 文件包含中文内容
- **THEN** PDF 中正确显示所有中文字符
