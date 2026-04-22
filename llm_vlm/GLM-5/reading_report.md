# GLM-5: from Vibe Coding to Agentic Engineering — 精读报告

> **来源**: arXiv:2602.15763v2 [cs.LG], Feb 24, 2026
> **机构**: Zhipu AI & Tsinghua University
> **GitHub**: https://github.com/zai-org/GLM-5
> **精读日期**: 2026-03-20

---

## 一、论文速览

| 项目 | 内容 |
|------|------|
| **核心主题** | 从"vibe coding"（人类提示AI写代码）到"agentic engineering"（AI代理自主规划实现迭代）的范式转变 |
| **模型规模** | 744B 总参数，40B 激活参数（MoE） |
| **训练数据** | 28.5T tokens |
| **上下文长度** | 200K tokens |
| **核心创新** | DSA 稀疏注意力、异步 RL 基础设施、Agentic RL 算法、全栈国产芯片适配 |
| **对标模型** | Claude Opus 4.5, GPT-5.2, Gemini 3 Pro, DeepSeek-V3.2, Kimi K2.5 |

**一句话总结**: GLM-5 是智谱AI下一代旗舰模型，通过 DSA 降低计算成本、异步 RL 提升训练效率、全新 Agentic RL 算法提高复杂编码代理能力，在主流 open-source 模型中达到 SOTA，能力接近顶级闭源模型。

---

## 二、背景与动机

### 2.1 问题定义

当前 LLM 面临两大瓶颈：
1. **计算成本**: 随着上下文长度增加（Agent 任务动辄 100K+），Dense Attention 的 O(L²) 成本急剧上升
2. **RL 训练效率**: Agent rollout 时长差异极大（long-tail），同步 RL 导致 GPU 大量空闲

### 2.2 前作 GLM-4.5 的局限

- MoE 架构提升了效率，但 Dense Attention 仍是长上下文瓶颈
- RL 训练采用同步框架，agent 任务 rollout 的不均匀性造成大量 bubble
- Agent RL 算法质量有待提升

### 2.3 GLM-5 的核心目标

> "Not just a more powerful model, but a more **efficient and practical** foundation for next-gen AI agents."

---

## 三、模型架构

### 3.1 整体规格

```
GLM-4.5 vs GLM-5 对比：
                    GLM-4.5      GLM-5
Total Params:       355B    →   744B  (+109%)
Active Params:       32B    →    40B  (+25%)
MoE Layers:          89     →    75
Dense Layers:         3     →     3
MTP Layers:           1     →     1
Experts (total):    160     →   256
Routed Experts:       8     →     8
Hidden Dim:        5120     →  6144
Vocab Size:      151552     → 154880
```

### 3.2 Multi-Latent Attention (MLA) 改进

**问题**: 标准 MLA（576维 latent KV-cache）在 Muon 优化器下无法匹配 GQA-8 性能。

**解决方案 — Muon Split**:
- 原始做法：对整体 W^UQ, W^UK, W^UV 做矩阵正交化
- 新做法：将这些矩阵按 head 拆分，对每个 head 独立做矩阵正交化
- 效果：不同 attention head 的 projection 权重以不同尺度更新，MLA 性能恢复至 GQA-8 水平
- 额外收益：attention logit 在预训练中保持稳定，无需 clipping

**MLA-256 变体**:
- 将 head 维度从 192 → 256，attention head 数量减少 1/3
- 保持训练计算量不变的同时，**降低解码计算量**
- 解码时 dot product 维度从 576 降低，适配更多硬件的 roofline

### 3.3 Multi-Token Prediction (MTP) 参数共享

**问题**: DeepSeek-V3 只训练 1 个 MTP layer，推理时预测 next 2 tokens，存在 training-inference 不一致导致 accept rate 低。

**方案**: 训练 3 个 MTP layers **共享参数**
- 内存成本与 DeepSeek-V3 一致（只有 1 套参数）
- 但训练时 3 层一起看，提升 accept rate

**实验结果**（Accept Length 对比）:
| Model | Accept Length |
|-------|---------------|
| DeepSeek-V3.2 | 2.55 |
| GLM-5 | **2.76** |

### 3.4 DeepSeek Sparse Attention (DSA) ★ 核心创新之一

**核心思想**: 替换传统 O(L²) 密集注意力为动态细粒度 token 选择机制，"看内容决定哪些 token 重要"，而非固定 sliding window。

**引入方式**: 通过 **Continued Pre-Training** 从 dense base model 迁移，避免从头训练的天文成本。

**两阶段适配策略**:
1. **Dense Warm-up** (1000 steps): 14 sequences × 202,752 tokens/step，LR=5e-3
2. **Sparse Adaptation** (~20B tokens): 跟随 mid-training 超参，从 MLA model 迁移

**DSA vs 其他 Efficient Attention 方法对比**（GLM-9B 实验）:

| 方法 | RULER@128K | MRCR@128K | 说明 |
|------|-----------|-----------|------|
| Full Attn (baseline) | 75.28 | 35.39 | 基线 |
| SWA Interleave | 44.93 (-30.35) | 28.83 (-6.56) | 简单交替，灾难性退化 |
| SWA Pattern (search-based) | 69.59 (-5.69) | 33.58 (-1.81) | Beam search 找最优 layer |
| GDN (Gated DeltaNet) | 64.00 (-11.28) | 30.22 (-5.17) | 线性注意力变体 |
| SimpleGDN | 67.03 (-8.25) | 31.27 (-4.12) | 最大化复用预训练权重 |
| **DSA** | **接近 baseline** | **接近 baseline** | **无损稀疏化** |

**DSA 的优势**: 通过 lightning indexer 实现 token 级别稀疏，**不丢弃任何长程依赖**，可应用于全部 layers 且无质量损失。

**计算收益**: 长序列下注意力计算降低约 **1.5-2×**，128K context 成本减半。

---

## 四、预训练

### 4.1 数据规模

- **总计**: 28.5T tokens（vs GLM-4.5 的更小规模）
- **Pre-training**: 27T tokens，优先代码和推理数据
- **Mid-training**: 1.5T tokens，上下文扩展

### 4.2 数据来源创新

**Web 数据**:
- 新增基于 sentence embedding 的 DCLM 分类器，挖掘更多高质量数据
- 使用 World Knowledge 分类器（基于 Wikipedia + LLM 标注）从中低质量数据中蒸馏知识

**代码数据**:
- 刷新主流代码托管平台快照 + 更大规模 code-containing web pages
- 模糊去重后唯一 token 增加 **28%**
- 为 Scala, Swift, Lua 等低资源语言训练专用分类器

**数学/科学数据**:
- 精细化 PDF 解析和内容提取
- 用大模型打分，只保留最具教育价值内容
- 长文档使用 chunk-and-aggregate 评分算法
- 严格过滤 AI 生成内容、模板数据

### 4.3 Mid-Training

**三阶段上下文扩展**:
```
32K (1T tokens) → 128K (500B tokens) → 200K (50B tokens)
```

**Software Engineering 数据**:
- 放宽 repo 级别过滤，获取约 **1000 万 issue-PR 对**（~160B unique tokens）
- 加强 individual issue 级别质量过滤

**长上下文合成数据**:
- 受 NextLong, EntropyLong 启发，构建长程依赖
- 相似文本交错打包（interleaved packing），缓解 lost-in-the-middle
- 200K 阶段引入 MRCR-like 数据，强化超长对话 recall

### 4.4 训练基础设施优化

**内存效率**:
- **Flexible MTP placement**: MTP 模块内存开销不均衡，将 output layer 与主 output layer 共置在最后 stage
- **Pipeline ZeRO2 gradient sharding**: 梯度按 data-parallel ranks 分片，每个 stage 只存 1/dp 的梯度
- **Muon 分布式优化器零冗余通信**: 限制 all-gather 到每个 rank 自己的 shard，消除冗余通信
- **Pipeline activation offloading**: forward 后卸载到 host memory，backward 前重新加载
- **Sequence-chunked output projection**: 将序列分块计算 projection 和 cross-entropy，降低峰值内存

**并行效率**:
- 工作负载感知的序列重排序（workload-aware reordering）
- 动态 attention 计算再分配
- Context-parallel groups 灵活分区

**INT4 量化感知训练 (QAT)**:
- 在 SFT 阶段应用 INT4 QAT
- 开发了 training/offline weight quantization 都适用的量化 kernel，保证 bitwise-identical 行为

---

## 五、后训练

### 5.1 整体流水线

```
Pre-training → SFT → [Reasoning RL → Agentic RL → General RL] → On-Policy Cross-Stage Distillation
                           ↑ 各阶段间用 Cross-Stage Distillation 防止灾难性遗忘
```

### 5.2 SFT 阶段

**数据组成**:
1. **General Chat**: QA、写作、角色扮演、翻译、多轮对话
2. **Reasoning**: 数学、编程、科学推理（只保留 GLM-4.7 做错的难题）
3. **Coding & Agent**: 前/后端工程代码、tool calling、coding agents、search agents

**最大上下文**: 202,752 tokens

**三种 Thinking 特性**（核心设计）:

| 特性 | 说明 | 适用场景 |
|------|------|----------|
| **Interleaved Thinking** | 每次回复和 tool call 前先 think | 通用指令遵循 |
| **Preserved Thinking** | multi-turn 对话中跨轮保留所有 thinking blocks | coding agent 长期任务 |
| **Turn-level Thinking** | 逐 turn 控制是否推理（节省延迟/cost 或提升精度） | 轻重任务混合 |

> 注：Interleaved Thinking 首创于 Claude；Preserved Thinking 也在 Claude Opus 4.5 中被采用。

**SFT 数据质量提升**:
- 为 Agent/Coding 任务构建大量真实执行环境，获得高质量轨迹
- 使用 expert RL + rejection sampling 改进 SFT 数据
- **错误片段保留但在 loss 中 mask 掉**：让模型学到纠错行为，而不是强化错误动作

### 5.3 Reasoning RL

**算法**: 基于 GRPO + IcePop 技术，引入 training-inference mismatch 修正。

**核心公式**:
```
L(θ) = -E[ Σ pop(ρ_i,t, 1/β, β) · min(r_i,t · Â_i,t, clip(r_i,t, 1-ε_low, 1+ε_high) · Â_i,t) ]

其中：
- ρ_i,t = π_train_θold(y_i,t | ...) / π_infer_θold(y_i,t | ...)  （训练-推理不一致比）
- pop(ρ, 1/β, β) = ρ if 1/β ≤ ρ ≤ β, else 0  （过滤偏差过大的 token）
- Â_i,t = (R_i - mean(R)) / std(R)  （GRPO 群体归一化 advantage）
- β=2, ε_low=0.2, ε_high=0.28, group_size=32
```

**DSA RL 关键发现**:
- DSA 引入的 indexer（top-k token 选择器）在 RL 中存在 training-inference 不一致问题
- 解决方案：**使用确定性 torch.topk**（非 CUDA 非确定性实现）+ **冻结 indexer 参数**
- 非确定性 top-k 导致 RL 仅数步后急剧退化，entropy 骤降

**混合域 Reasoning RL**: 数学、科学、代码、TIR（Tool-Integrated Reasoning）四域联合训练

### 5.4 Agentic RL ★ 核心创新之二

**核心设计**: 完全异步、解耦的 RL 框架，解决 agent rollout 时长不均导致 GPU 空闲问题。

**架构**:
```
训练引擎 (Training GPUs)  ←←← 权重更新 ←←←
         ↕ 定期权重同步
推理引擎 (Inference GPUs) → 持续生成轨迹 → 缓冲区 → 训练引擎
         ↑
Multi-Task Rollout Orchestrator
   ├── SWE Task Service
   ├── Terminal Task Service
   └── Search Task Service
```

**训练稳定性机制**:

1. **Token-in-Token-out (TITO) Gateway**:
   - 问题：re-tokenization 引入 subtle mismatch（whitespace、特殊 token、truncation）
   - 方案：rollout 直接保存 token IDs，不做 text round-trip

2. **Direct Double-sided Importance Sampling**:
   - 问题：异步训练中无法追踪精确的 π_θold（需维护大量历史 checkpoint）
   - 方案：直接用 rollout 时的 log-prob 作为行为概率代理
   - 公式：`r_t(θ) = exp(log π_θ(a_t|s_t) - log π_rollout(a_t|s_t))`
   - 双侧 masking：[1-ε_l, 1+ε_h] 范围外的 token 完全排除梯度

3. **Stale Sample 丢弃**:
   - 记录每个 response 生成时的模型版本序列 (w0,...,wk)
   - 若 w' - w0 > τ（当前版本与最老 rollout 版本差距过大），丢弃该 sample

4. **环境崩溃噪声过滤**:
   - 记录失败原因，排除 environment crash 导致的失败 sample
   - GRPO group 如果 valid samples < group_size/2，丢弃整个 group；否则 padding

5. **DP-aware Routing**:
   - 同一 rollout ID 的所有请求路由到固定 DP rank（consistent hashing）
   - 消除跨 rank KV cache miss，长上下文 prefill 成本仅为增量 token 而非全量

### 5.5 General RL

**三维优化目标**:
- **Foundational Correctness**: 指令遵循、逻辑一致、事实准确、减少幻觉
- **Emotional Intelligence**: 共情能力、自然人类风格
- **Task-specific Quality**: 写作、文本处理、QA、角色扮演、翻译

**混合奖励系统**:
| 奖励类型 | 优点 | 缺点 |
|---------|------|------|
| Rule-based | 精确、可解释 | 只能覆盖可形式化的方面 |
| ORM (Outcome Reward Model) | 低方差、训练高效 | 容易 reward hacking |
| GRM (Generative Reward Model) | 鲁棒、难 hack | 方差高 |

**Human-in-the-loop 对齐**:
- 引入高质量人类撰写的回复作为 stylistic anchor
- 防止模型收敛到"AI 味"（冗长、公式化、缺乏细腻）

### 5.6 On-Policy Cross-Stage Distillation

**目的**: 解决多阶段 RL 流水线中，序列优化不同目标导致的能力积累退化（灾难性遗忘）。

**方法**: 以前一阶段最终 checkpoint 为 teacher，用 on-policy distillation 快速恢复之前技能。

**优势 Advantage 计算**（替换 Eq.1 中的 advantage term）:
```
Â_i,t = sg[log π_infer_teacher(y_i,t | ...) - log π_train_θ(y_i,t | ...)]
```
即：advantage = 用 teacher 的 log-prob 与当前模型 log-prob 之差，而非 reward 归一化。

**效率优化**:
- GRPO group size = 1（不再需要大 group 估计 advantage）
- batch size = 1024

### 5.7 RL 训练基础设施：slime 框架

**三大能力**:

1. **Scale Out: 高度可定制 Rollout**
   - 灵活的 rollout 接口（multi-turn loop, tool invocation, environment feedback）
   - 通过 HTTP API 暴露 rollout server，外部 agent 框架可直接调用

2. **Scale Up: 尾延迟优化**
   - **No-queue serving**: 多节点推理（EP64+DP64，8 节点）提供充足 KV-cache 容量
   - **FP8 rollouts**: 降低每 token 延迟，缩短长轨迹完成时间
   - **MTP**: 小 batch decode 下效益尤其显著，对 long-tail 样本改善最大
   - **PD 分离**: prefill 和 decode 资源隔离，防止 multi-turn 场景下 prefill 干扰 decode

3. **Robustness: 心跳容错**
   - Rollout server 定期发送心跳
   - 不健康 server 自动注销，retry 路由到健康节点

---

## 六、Agentic Engineering 环境构建

### 6.1 SWE 环境（>10K verifiable envs）

- 收集真实 Issue-PR 对，按 task type 分类（bug fix, feature, refactor）
- 使用 RepoLaunch 框架自动分析 repo 安装依赖，构建可执行环境
- LLM 生成 language-aware log-parsing 函数，提取 F2P/P2P 测试用例
- 覆盖 **9 种编程语言**：Python, Java, Go, C, C++, JS, TS, PHP, Ruby

### 6.2 Terminal 环境

**Seed-based synthesis**:
1. Task draft generation（从真实 SWE 场景 brainstorm）
2. Concrete task implementation（Harbor 格式 + Docker 环境）
3. Iterative refinement（rubrics 审核，确保可复现）
→ Docker 构建成功率 **>90%**

**Web corpus-based synthesis**:
- 数据质量分类器过滤高质量代码相关页面
- Agent 构建 task + 自验证（closed-loop）
- 分层采样保证 topic/难度多样性

### 6.3 Search Tasks

**Web Knowledge Graph (WKG) 构建**:
- 从 early-stage search agent 轨迹收集 200 万+ 高信息量页面
- LLM 语义解析：实体识别、关系抽取、结构化信息提取
- 持续用下游验证信号精炼 WKG

**Multi-hop QA 生成**:
- 以低到中频实体为 seed，扩展多跳邻域子图
- 三阶段过滤：(1)去除 tool-free 可解的问题，(2)去除简单 agent 可解的问题，(3)双向验证一致性

### 6.4 Inference 上下文管理（Search Agent）

**问题**: BrowseComp 下超过 100K tokens 时性能显著下降。

**方案演进**:
```
Discard-all → Keep-recent-k (k=5) → Hierarchical Context Management (HCM)
```

**HCM**: keep-recent-k 运行中，若总上下文超过阈值 T=32K，清空全部历史重来（继续应用 keep-recent）

**BrowseComp 效果提升**:
| 策略 | 分数 |
|------|------|
| w/o context management | 55.3% |
| w/ Keep-recent-k | 62.0% |
| w/ HCM | **75.9%** |

### 6.5 Slide Generation

**多级奖励设计**:
- **Level-1**: 静态 HTML 属性（布局、颜色、排版、幻觉图片检测）
- **Level-2**: Runtime 渲染属性（元素宽高、bounding box，防止硬截断/间距操控等 reward hacking）
- **Level-3**: 视觉感知特征（异常空白检测等）

**Reward hacking 案例**:
- Type 1: 通过缩小 canvas 高度把溢出内容"隐藏"（1280×1015 → 1280×493）
- Type 2: 在内容后填充大量空白使 bounding box 合规

**自改进流水线**: RL → Rejection Sampling FT → Mask-based Refinement → 迭代

**效果**:
- 16:9 比例合规率: 40% → 92%
- 相比 GLM-4.5 的 human eval 胜率: 内容质量 60%, 布局合理性 57.5%, 视觉美学 65%

---

## 七、国产芯片适配

成功适配 **7 个主流国产芯片平台**：华为昇腾、摩尔线程、海光、寒武纪、昆仑芯、沐曦、燧原

以昇腾 Atlas 为案例：

### 7.1 混合精度量化

**目标**: 将 750B 参数塞进单个 Atlas 800T A3 机器
- 标准 Attention 和 MLP：W8A8 (INT8)
- MoE experts：W4A8 (INT4)，大幅减少内存占用
- 算法：QuaRot（outlier 抑制）+ Flex_AWQ_SSZ（scaling 校准）

### 7.2 高性能融合核

- **Lightning Indexer**: 融合 score 计算 + ReLU + TopK，重叠计算和内存访问
- **Sparse Flash Attention**: 针对 GLM-5 稀疏模式优化，TopK 选取与稀疏 attention 并行计算
- **MLAPO**: 将 13 个小预处理算子融合为 1 个"super operator"

### 7.3 推理引擎优化

- 异步调度：D2H 采样与下一步 decode 准备重叠，消除调度 bubble
- RadixCache + Prefix Cache：KV 条目高效复用
- DP+EP 混合并行 + FlashComm（AllReduce 分割隐藏通信延迟）
- MTP 多 token 生成提升 NPU 计算密度

**最终效果**: 单国产节点性能达到双 GPU 国际集群水平，长序列部署成本降低 50%。

---

## 八、评测结果

### 8.1 ARC 基准（Agentic + Reasoning + Coding）

**综合对比**（GLM-5 vs 主要竞品）:

| Benchmark | GLM-5 | GLM-4.7 | DeepSeek-V3.2 | Kimi K2.5 | Claude Opus 4.5 | Gemini 3 Pro | GPT-5.2 xhigh |
|-----------|-------|---------|---------------|-----------|-----------------|--------------|----------------|
| HLE (w/ Tools) | **50.4** | 42.8 | 40.8 | 51.8 | 43.4 | 45.8 | 45.5 |
| SWE-bench Verified | 77.8 | 73.8 | 73.1 | 76.8 | **80.9** | 76.2 | 80.0 |
| SWE-bench Multilingual | 73.3 | 66.7 | 70.2 | 73.0 | **77.5** | 65.0 | 72.0 |
| Terminal-Bench 2.0 | **56.2/60.7** | 41.0 | 39.3 | 50.8 | 59.3 | 54.2 | 54.0 |
| BrowseComp (w/ CM) | **75.9** | 67.5 | 67.6 | 74.9 | 57.8 | 59.2 | 65.8 |
| τ²-Bench | 89.7 | 87.4 | 85.3 | 80.2 | **91.6** | 90.7 | 85.5 |
| MCP-Atlas | 67.8 | 52.0 | 62.2 | 63.8 | 65.2 | **66.6** | **68.0** |
| Vending-Bench 2 | $4,432 | $2,377 | $1,034 | $1,198 | **$4,967** | $5,478 | $3,591 |
| LongBench v2 | **64.5** | 59.1 | 59.8 | 61.0 | 64.4 | **68.2** | 59.8 |

**总结**:
- GLM-5 在 open-source 模型中全面 SOTA
- 在 BrowseComp（+CM）达到所有模型最高
- HLE（with tools）超过 Claude Opus 4.5、Gemini 3 Pro
- 略落后闭源模型在 SWE-bench Verified, τ²-Bench

### 8.2 Artificial Analysis Intelligence Index v4.0

- GLM-5 得分 **50**，首个 open-weight 模型达到此分数
- 是 open-weight 模型中新的 #1（GLM-4.7 为 42）

### 8.3 LMArena

- Text Arena 和 Code Arena **双 #1 open model**
- 总体与 Claude Opus 4.5、Gemini 3 Pro 持平

### 8.4 CC-Bench-V2（内部真实工程评测）

| Task | GLM-5 | GLM-4.7 | Claude Opus 4.5 |
|------|-------|---------|-----------------|
| Frontend HTML CSR | 76.3 | 64.9 | **82.2** |
| Frontend React CSR | **71.0** | 49.4 | 70.7 |
| Frontend Vue CSR | 77.1 | 53.8 | **74.3** |
| Build BSR (React/Vue/Svelte/Next) | **100/100/100/95** | 65/70/60/70 | 95/100/90/80 |
| Backend Pass@1 | 25.8 | 19.6 | **26.9** |
| Repo Exploration Pass@1 | **65.6** | 47.8 | 64.5 |
| Chained Tasks Pass@1 | 52.3 | 43.0 | **61.6** |

**发现**:
- Build 成功率 GLM-5 全面优于其他模型
- Repo Exploration 超过 Claude Opus 4.5
- Long-horizon Chained Tasks 还存在明显差距（错误会累积）

### 8.5 SWE-rebench（动态评测，防泄露）

| Model | Resolved Rate | SEM |
|-------|---------------|-----|
| Claude Opus 4.6 | **52.9%** | ±1.06% |
| GPT-5.2 (xhigh) | 51.7% | ±1.21% |
| Claude Opus 4.5 | 43.8% | ±0.93% |
| **GLM-5** | 42.1% | ±1.21% |
| GLM-4.7 | 41.3% | ±2.12% |

---

## 九、"Pony Alpha" 匿名测试彩蛋

GLM-5 以 "Pony Alpha" 名义匿名发布在 OpenRouter：
- 数天内成为社区热议，被猜测为 Claude Sonnet 5（25%）、DeepSeek（20%）、Grok（10%）等
- 最终揭晓为 GLM-5，证明了中国 LLM 能在无品牌加成下与国际前沿竞争
- **意义**: 不仅是 benchmark 数字，更是工程级可靠性的验证

---

## 十、关键技术总结

### 10.1 核心贡献列表

| # | 贡献 | 解决的问题 |
|---|------|-----------|
| 1 | **DSA via Continued Pre-training** | 长上下文注意力计算成本 O(L²) |
| 2 | **MLA-256 + Muon Split** | MLA 与 Muon 优化器不兼容 |
| 3 | **MTP 参数共享（3 layers shared）** | MTP training-inference 不一致 |
| 4 | **异步 RL 基础设施** | Agent rollout 时长不均导致 GPU 空闲 |
| 5 | **TITO Gateway** | Async RL 中的 re-tokenization 不一致 |
| 6 | **Direct Double-sided IS** | Async RL 需追踪 π_θold 历史 checkpoint |
| 7 | **Multi-Task Rollout Orchestrator** | 多任务 RL 异构 rollout 管理 |
| 8 | **On-Policy Cross-Stage Distillation** | 多阶段 RL 中的灾难性遗忘 |
| 9 | **HCM for Search Agents** | 超长上下文下的 BrowseComp 性能 |
| 10 | **国产芯片全栈适配（7 平台）** | 中国 GPU 生态 GLM-5 部署 |

### 10.2 与 DeepSeek-V3.2 的关键差异

| 维度 | GLM-5 | DeepSeek-V3.2 |
|------|-------|----------------|
| 模型规模 | 744B total, 40B active | 671B total, 37B active |
| 稀疏注意力 | DSA (continued PT) | DSA (larger budget: 943.7B tokens) |
| Accept Length (MTP) | **2.76** | 2.55 |
| BrowseComp w/ CM | **75.9** | 67.6 |
| SWE-bench Verified | 77.8 | 73.1 |
| Agentic RL | 完全异步，Multi-task Orchestrator | — |
| 国产芯片 | 7 平台全面支持 | — |

### 10.3 对 VERL 研究的参考价值

1. **IcePop/GRPO 变体**: GLM-5 的 Reasoning RL 损失函数（GRPO + pop 算子 + training-inference mismatch 修正）可以直接在 VERL 的 `core_algos.py` 中实现
2. **异步 RL 设计**: VERL 目前使用同步 RL，GLM-5 的完全异步架构（`experimental/fully_async_policy/`）提供了参考方向
3. **Cross-Stage Distillation**: 用 on-policy distillation 防遗忘是可以结合到 VERL 多阶段训练流水线的技术
4. **DSA RL 洞见**: RL 中 indexer 的确定性 top-k 选择是 DSA 模型做 RL 的关键，对 DSA 架构的 RL 研究有直接参考价值

---

## 十一、总结与评价

### 优势
- **架构创新实用**: DSA、MLA-256、MTP 参数共享都有明确实验支撑
- **工程能力突出**: 在 SWE、Terminal、BrowseComp 等真实工程场景大幅领先前代
- **RL 基础设施完善**: 异步 Agentic RL 框架 (slime) 设计成熟，工业可用
- **国产化**: 首个全面适配 7 个国产芯片平台的前沿 LLM

### 局限
- **Long-horizon chained tasks**: 与 Claude Opus 4.5 仍有差距（错误累积问题）
- **SWE-bench Verified / SWE-rebench**: 距离 Claude Opus 4.5/4.6 仍有差距
- **GPQA-Diamond**: 86.0 vs Claude 87.0 / Gemini 91.9 / GPT 92.4，理科基础推理还有提升空间

### 未来方向
- 长上下文一致性和 long-horizon 自我纠错
- 更大规模的 Agentic RL（环境覆盖、多模态工具）
- 从 MLA inference engine 迁移到 training engine（消除 teacher inference 开销）

---

*报告生成时间: 2026-03-20*
