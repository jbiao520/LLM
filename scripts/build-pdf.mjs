#!/usr/bin/env node
/**
 * LLM 学习指南 PDF 构建脚本
 * 使用 Puppeteer 将 Markdown 文件转换为 PDF 电子书
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import yaml from 'js-yaml';
import MarkdownIt from 'markdown-it';
import hljs from 'highlight.js';
import katex from 'katex';
import puppeteer from 'puppeteer';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');

// 配置文件路径
const CONFIG_PATH = path.join(ROOT_DIR, 'scripts', 'pdf-config.yaml');
const CSS_PATH = path.join(ROOT_DIR, 'scripts', 'templates', 'ebook.css');

/**
 * 读取配置文件
 */
function loadConfig() {
  if (!fs.existsSync(CONFIG_PATH)) {
    console.error('错误: 配置文件 scripts/pdf-config.yaml 不存在');
    process.exit(1);
  }
  const content = fs.readFileSync(CONFIG_PATH, 'utf-8');
  return yaml.load(content);
}

/**
 * 读取文件内容
 * @param {string} filePath - 相对于项目根目录的路径或绝对路径
 */
function readFile(filePath) {
  const fullPath = path.isAbsolute(filePath) ? filePath : path.join(ROOT_DIR, filePath);
  if (!fs.existsSync(fullPath)) {
    console.warn(`警告: 文件不存在 ${filePath}`);
    return '';
  }
  return fs.readFileSync(fullPath, 'utf-8');
}

/**
 * 读取目录中的所有 Python 文件
 */
function readPythonFiles(dirPath) {
  const fullPath = path.join(ROOT_DIR, dirPath);
  if (!fs.existsSync(fullPath)) {
    return [];
  }
  const files = fs.readdirSync(fullPath)
    .filter(f => f.endsWith('.py'))
    .sort();
  return files.map(f => ({
    name: f,
    content: fs.readFileSync(path.join(fullPath, f), 'utf-8')
  }));
}

/**
 * 创建 Markdown 解析器
 */
function createMarkdownParser() {
  return MarkdownIt({
    html: true,
    linkify: true,
    typographer: true,
    highlight: function (str, lang) {
      if (lang && hljs.getLanguage(lang)) {
        try {
          return `<pre class="hljs"><code>${hljs.highlight(str, { language: lang, ignoreIllegals: true }).value}</code></pre>`;
        } catch (e) {
          console.warn(`代码高亮失败: ${lang}`);
        }
      }
      return `<pre class="hljs"><code>${escapeHtml(str)}</code></pre>`;
    }
  });
}

/**
 * HTML 转义
 */
function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * 渲染 LaTeX 公式
 */
function renderLatex(content) {
  // 渲染行间公式 $$...$$
  content = content.replace(/\$\$([\s\S]+?)\$\$/g, (match, formula) => {
    try {
      return katex.renderToString(formula.trim(), {
        displayMode: true,
        throwOnError: false,
        output: 'html'
      });
    } catch (e) {
      console.warn(`LaTeX 渲染失败: ${formula.substring(0, 50)}...`);
      return `<div class="latex-error">${escapeHtml(formula)}</div>`;
    }
  });

  // 渲染行内公式 $...$
  content = content.replace(/\$([^\$\n]+?)\$/g, (match, formula) => {
    try {
      return katex.renderToString(formula.trim(), {
        displayMode: false,
        throwOnError: false,
        output: 'html'
      });
    } catch (e) {
      return match;
    }
  });

  return content;
}

// 存储 mermaid 图表的临时变量
const mermaidPlaceholders = [];

/**
 * 处理 Mermaid ��表
 * 将 mermaid 代码块替换为占位符，Markdown 解析后再恢复
 */
function processMermaid(content) {
  mermaidPlaceholders.length = 0;
  return content.replace(/```mermaid\n([\s\S]*?)```/g, (match, code) => {
    const index = mermaidPlaceholders.length;
    mermaidPlaceholders.push(code.trim());
    return `MERMAID_PLACEHOLDER_${index}`;
  });
}

/**
 * 恢复 Mermaid 图表
 * 将占位符替换为实际的 mermaid div
 */
function restoreMermaid(html) {
  mermaidPlaceholders.forEach((code, index) => {
    // 将代码内容进行 HTML 转义，防止解析问题
    const escapedCode = code
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    html = html.replace(`MERMAID_PLACEHOLDER_${index}`, `<div class="mermaid">${escapedCode}</div>`);
  });
  return html;
}

/**
 * 生成代码附录 HTML
 */
function generateCodeAppendix(examples) {
  if (!examples || examples.length === 0) {
    return '';
  }

  let html = '<div class="code-appendix page-break">';
  html += '<h2>附录：代码示例</h2>';

  for (const file of examples) {
    html += `<div class="code-file">`;
    html += `<h3>${file.name}</h3>`;
    const highlighted = hljs.highlight(file.content, { language: 'python', ignoreIllegals: true }).value;
    html += `<pre class="hljs"><code>${highlighted}</code></pre>`;
    html += '</div>';
  }

  html += '</div>';
  return html;
}

/**
 * 处理章节内容
 */
function processChapter(chapter, md) {
  let html = '';

  // 章节标题
  html += `<div class="chapter">`;
  html += `<h1>${chapter.title}</h1>`;

  // 处理主内容
  for (const source of (chapter.sources || [])) {
    let content = readFile(source);
    if (content) {
      // 处理 Mermaid 图表 (替换为占位符)
      content = processMermaid(content);
      // 渲染 LaTeX 公式
      content = renderLatex(content);
      // 转换 Markdown 到 HTML
      let rendered = md.render(content);
      // 恢复 Mermaid 图表
      rendered = restoreMermaid(rendered);
      html += rendered;
    }
  }

  // 处理子章节
  if (chapter.subtopics) {
    for (const subtopic of chapter.subtopics) {
      html += `<h2 class="subtopic-title">${subtopic.title}</h2>`;
      for (const source of (subtopic.sources || [])) {
        let content = readFile(source);
        if (content) {
          content = processMermaid(content);
          content = renderLatex(content);
          let rendered = md.render(content);
          rendered = restoreMermaid(rendered);
          html += rendered;
        }
      }
      // 添加子章节代码附录
      if (subtopic.examples) {
        const examples = readPythonFiles(subtopic.examples);
        html += generateCodeAppendix(examples);
      }
    }
  }

  // 添加章节代码附录
  if (chapter.examples && !chapter.subtopics) {
    const examples = readPythonFiles(chapter.examples);
    html += generateCodeAppendix(examples);
  }

  html += '</div>';
  return html;
}

/**
 * 生成目录 HTML
 */
function generateToc(chapters) {
  let html = '<div class="toc">';
  html += '<h1>目录</h1>';
  html += '<ul>';

  for (const chapter of chapters) {
    html += `<li class="toc-chapter">${chapter.title}</li>`;
    if (chapter.subtopics) {
      html += '<ul>';
      for (const subtopic of chapter.subtopics) {
        html += `<li>${subtopic.title}</li>`;
      }
      html += '</ul>';
    }
  }

  html += '</ul>';
  html += '</div>';
  return html;
}

/**
 * 生成完整的 HTML 文档
 */
function generateHtml(config, content) {
  const css = readFile(CSS_PATH);

  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${config.title}</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <style>
    ${css}
  </style>
</head>
<body>
  ${content}
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>
    mermaid.initialize({
      startOnLoad: true,
      theme: 'default',
      maxTextSize: 100000,
      themeVariables: {
        fontFamily: 'Noto Sans SC, PingFang SC, Microsoft YaHei, sans-serif'
      }
    });
  </script>
</body>
</html>`;
}

/**
 * 生成 PDF
 */
async function generatePdf(html, outputPath) {
  console.log('正在启动浏览器...');
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  try {
    const page = await browser.newPage();

    // 设置内容
    await page.setContent(html, {
      waitUntil: 'networkidle0'
    });

    // 等待 Mermaid 渲染完成
    console.log('正在渲染 Mermaid 图表...');
    await page.waitForFunction(() => {
      const mermaidDivs = document.querySelectorAll('.mermaid');
      if (mermaidDivs.length === 0) return true;
      return Array.from(mermaidDivs).every(div => div.getAttribute('data-processed'));
    }, { timeout: 30000 }).catch(() => {
      console.warn('警告: 部分 Mermaid 图表可能未完全渲染');
    });

    // 额外等待确保渲染完成
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 生成 PDF
    console.log('正在生成 PDF...');
    const fullOutputPath = path.join(ROOT_DIR, outputPath);
    const outputDir = path.dirname(fullOutputPath);

    // 确保输出目录存在
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    await page.pdf({
      path: fullOutputPath,
      format: 'A4',
      margin: {
        top: '25mm',
        bottom: '25mm',
        left: '25mm',
        right: '25mm'
      },
      printBackground: true
    });

    console.log(`PDF 生成成功: ${fullOutputPath}`);
    return fullOutputPath;

  } finally {
    await browser.close();
  }
}

/**
 * 检查依赖是否已安装
 */
function checkDependencies() {
  const deps = ['puppeteer', 'markdown-it', 'katex', 'highlight.js', 'js-yaml'];
  const missing = [];

  for (const dep of deps) {
    try {
      // 检查 node_modules 中是否存在
      const depPath = path.join(ROOT_DIR, 'node_modules', dep);
      if (!fs.existsSync(depPath)) {
        missing.push(dep);
      }
    } catch (e) {
      missing.push(dep);
    }
  }

  return missing;
}

/**
 * 主函数
 */
async function main() {
  console.log('=== LLM 学习指南 PDF 构建脚本 ===\n');

  // 检查依赖
  console.log('检查依赖...');
  const missingDeps = checkDependencies();
  if (missingDeps.length > 0) {
    console.error('错误: 缺少必要依赖:', missingDeps.join(', '));
    console.error('请运行 npm install');
    process.exit(1);
  }
  console.log('依赖检查通过\n');

  // 加载配置
  console.log('加载配置文件...');
  const config = loadConfig();
  console.log(`书名: ${config.title}`);
  console.log(`输出: ${config.output}\n`);

  // 创建 Markdown 解析器
  const md = createMarkdownParser();

  // 处理所有章节
  console.log('处理章节内容...');
  let allContent = '';

  // 生成目录
  allContent += generateToc(config.chapters);

  // 处理每个章节
  for (let i = 0; i < config.chapters.length; i++) {
    const chapter = config.chapters[i];
    console.log(`  [${i + 1}/${config.chapters.length}] ${chapter.title}`);
    allContent += processChapter(chapter, md);
  }

  // 生成完整 HTML
  console.log('\n生成 HTML...');
  const html = generateHtml(config, allContent);

  // 保存中间 HTML (调试用)
  const htmlPath = path.join(ROOT_DIR, 'output', 'debug.html');
  const outputDir = path.dirname(htmlPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  fs.writeFileSync(htmlPath, html, 'utf-8');
  console.log(`HTML 已保存: ${htmlPath}`);

  // 生成 PDF
  await generatePdf(html, config.output);

  console.log('\n=== 构建完成 ===');
}

main().catch(err => {
  console.error('构建失败:', err);
  process.exit(1);
});
