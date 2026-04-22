# arxiv-paper-reader

精读 arXiv 论文，生成本地 HTML 阅读笔记，支持持续交互补充。

## 触发条件

用户发送 arXiv 链接时自动触发：
```
精读 https://arxiv.org/abs/2504.13181
```

## 工作流程

### Phase 1: 自动处理（无需用户输入）

1. **下载 PDF** — 从 arXiv 下载论文 PDF
2. **提取内容** — 用 PyMuPDF 提取全文文本 + 主要图片（> 50KB）
3. **LLM 分析** — 总结论文：Background、Method、Results、Takeaways
4. **交互询问** — 问用户：
   > "论文已分析完毕。请告诉我放在 `paper_reading/` 的哪个子目录？例如 `vision_encoder/perception-encoder/`"

### Phase 2: 生成本地笔记（用户回答后）

5. **创建目录** — `mkdir -p paper_reading/<用户指定的子目录>`
6. **保存文件** — PDF、提取的图片、提取的文本
7. **生成 HTML 阅读笔记** — 包含以下板块：
   - 基本信息（标题、作者、arXiv、年份）
   - Background & Motivation
   - 核心贡献 / 主要发现
   - 方法详解
   - 实验结果
   - 重要 Takeaways
   - 个人思考（预留空白区域，供后续填充）
   - **Q & A 区域**（后续交互中逐步填充）
8. **告知用户** — 输出 HTML 路径，提示可以继续补充

### Phase 3: 持续交互补充

用户可以在同一 session 中继续提问、补充见解：
```
用户: 补充一点：这个工作和 DINOv2 的关系是什么？
    ↓
我: [分析回答] 已更新到 HTML 的 Q & A 区域
```

**每次用户补充后，读取当前 HTML → 在 Q & A 区域追加新条目 → 写回文件。**

---

## HTML 笔记模板结构

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="paper-title" content="{论文标题}">
  <meta name="arxiv-id" content="{arxiv编号}">
  <meta name="last-updated" content="{ISO时间}">
  <title>{论文标题} — Reading Notes</title>
  <style>
    /* 暗色/亮色自适应的简洁样式 */
    :root {
      --bg: #f8f9fa;
      --card: #ffffff;
      --text: #1a1a2e;
      --text-secondary: #555;
      --accent: #0066cc;
      --accent-light: #e6f0fa;
      --border: #e0e0e0;
      --success: #2e7d32;
      --warning: #ed6c02;
      --danger: #d32f2f;
      --purple: #7c4dff;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f1115;
        --card: #1a1d24;
        --text: #e0e2e8;
        --text-secondary: #a0a4b0;
        --accent: #4da3ff;
        --accent-light: #1a2d4a;
        --border: #2a2d36;
      }
    }
    /* ... 具体样式省略，参考实际生成的 reading_notes.html ... */
  </style>
</head>
<body>
  <!-- 论文信息头 -->
  <!-- Background & Motivation -->
  <!-- 核心发现 -->
  <!-- 方法详解 -->
  <!-- 实验结果 -->
  <!-- Takeaways -->
  <!-- 个人思考 -->
  
  <!-- Q & A 区域 -->
  <div class="card" id="qa">
    <h2>Q & A</h2>
    <div class="qa-item">
      <div class="qa-q">用户: 补充的问题</div>
      <div class="qa-a">回答内容...</div>
      <div class="qa-time">2025-04-21 14:30</div>
    </div>
  </div>
</body>
</html>
```

---

## 关键实现细节

### 1. Session 内跟踪当前论文

在 Phase 1 完成后，**记录当前正在处理的论文信息**（如 HTML 文件路径），以便 Phase 3 快速定位：

```javascript
// 伪代码：在分析完成后记录
const currentPaper = {
  arxivId: "2504.13181",
  htmlPath: "/Users/.../paper_reading/vision_encoder/perception-encoder/reading_notes.html",
  pdfPath: "/Users/.../paper_reading/vision_encoder/perception-encoder/paper.pdf",
  createdAt: "2025-04-21T10:00:00Z"
};
```

> 实际实现：通过读取当前目录列表或用户上下文来推断。如果用户在同一 session 中继续讨论，默认使用最近一次生成的 HTML。

### 2. Q & A 更新机制

```
1. 读取现有 HTML 文件
2. 找到 <div class="card" id="qa"> 区域
3. 在最后一个 .qa-item 后追加新的 Q&A block
4. 更新 <meta name="last-updated">
5. 写回文件
```

Q&A HTML 格式：
```html
<div class="qa-item">
  <div class="qa-q">🙋 用户: {问题}</div>
  <div class="qa-a">🤖 {回答}</div>
  <div class="qa-time">{ISO时间}</div>
</div>
```

### 3. 用户指定子目录的交互

如果用户不确定目录结构，提供建议：
> "建议放在 `paper_reading/vision_encoder/perception-encoder/`，如果涉及多模态可以放 `mllm/`，涉及视频可以放 `video_understanding/`。请告诉我你的偏好，或让我帮你选一个？"

如果用户只给一个类别名（如 `vision_encoder`），自动追加论文标题的 kebab-case 版本作为子目录：
> "好的，放在 `paper_reading/vision_encoder/perception-encoder/`"

---

## 目录结构示例

```
paper_reading/
├── vision_encoder/
│   └── perception-encoder/
│       ├── reading_notes.html      # HTML 阅读笔记
│       ├── paper.pdf               # 原始 PDF
│       ├── paper_text.txt          # 提取的全文文本
│       └── images/                 # 提取的图表
│           ├── page2_img1.jpeg
│           └── ...
├── mllm/
│   └── some-mllm-paper/
│       ├── reading_notes.html
│       └── ...
└── ...
```

---

## 使用示例

### 完整流程

```
用户: 精读 https://arxiv.org/abs/2504.13181
    ↓
我: 📥 下载论文 PDF
    📄 提取 33 张主要图表 + 全文文本
    🧠 LLM 分析论文内容
    
    论文已分析完毕！请告诉我放在 paper_reading/ 的哪个子目录？
    例如：vision_encoder/perception-encoder/
    
用户: vision_encoder/perception-encoder/
    ↓
我: ✅ 已保存到 paper_reading/vision_encoder/perception-encoder/
    📄 reading_notes.html 已生成
    
    你可以继续问我问题，我会把回答更新到 HTML 的 Q&A 区域。
    
用户: 这个和 SigLIP2 有什么区别？
    ↓
我: [详细回答]
    💾 已更新到 reading_notes.html 的 Q&A 区域
```

### 后续补充

```
用户: 刚刚那篇 Perception Encoder 的 Q&A 里，补充一下视频数据引擎的细节
    ↓
我: [读取 reading_notes.html]
    [在 Q&A 区域追加]
    💾 已更新
```

---

## 依赖

- **PyMuPDF**: PDF 解析和图片提取
- **LLM**: 论文内容分析
- **文件系统**: 本地目录创建和 HTML 文件读写

## 与其他 skill 的关系

```
paper-research (主题调研) → 发现感兴趣的论文
    ↓
arxiv-paper-reader (精读单篇) → 生成本地 HTML 笔记
    ↓
后续交互 → 逐步完善 Q&A
```
