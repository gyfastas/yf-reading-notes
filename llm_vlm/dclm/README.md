# DCLM: DataComp for Language Models

**论文**: DCLM: In search of the next generation of training sets for language models  
**arXiv**: https://arxiv.org/abs/2406.11794  
**本地文件**: `paper.pdf`

## 核心贡献

DCLM 是一个用于语言模型训练数据集控制的测试平台，提供：
- **240T tokens** 标准化语料库（从 Common Crawl 提取）
- 基于 OpenLM 框架的有效预训练配方
- **53 个下游评估任务**

## 关键结果

| 模型 | 数据 | MMLU 5-shot | 计算效率 |
|------|------|-------------|----------|
| DCLM-Baseline 7B | 2.6T tokens | **64%** | 基准 |
| MAP-Neo 7B | - | 57.4% | 更低 |
| Llama 3 8B | - | 66% | **6.6×** 更多计算 |

## 核心发现

### 1. Model-based Filtering 是关键
对比多种筛选方法后，**fastText bigram classifier** 配合精心选择的正负样本表现最佳。

测试的方法包括：
- PageRank score filtering
- Semantic Deduplication (SemDedup)
- BGE 嵌入线性分类器
- AskLLM (提示 LM 判断)
- Perplexity filtering (CCNet 风格)
- Top-k average logits

### 2. Fuzzy Deduplication

#### MinHash
- 局部敏感哈希 (LSH) 技术
- 基于 Jaccard 相似度
- n-gram: 5 tokens
- 目标相似度: 0.8
- 哈希排列: 1,395 (93 buckets × 15 hashes)

#### Bloom Filter (BFF)
- 空间高效的概率性数据结构
- 用于大规模去重 (>10TB)
- 无假阴性，有假阳性
- DCLM-Baseline 使用此方法

### 3. 数据管理流程
```
Common Crawl → 文本提取 (Resiliparse/Trafilatura) → 启发式清洗 → 去重 → Model-based filtering
```

## 在 GLM-5 中的应用

GLM-5 在其数据管线中使用了 DCLM 的改进策略：
- 基于 sentence embeddings 的 DCLM 分类器
- 识别并聚合超出标准分类器的高质量数据
- World Knowledge 分类器（针对 Wikipedia 优化）

## 论文关键章节

- **Sec 4.3**: Deduplication (MinHash vs Bloom Filter)
- **Sec 4.4**: Model-based quality filtering
- **Appendix L**: 额外去重分析

---

*Added: 2026-04-11*
