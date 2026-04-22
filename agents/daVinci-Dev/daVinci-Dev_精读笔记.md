# daVinci-Dev: Agent-native Mid-training for Software Engineering

**论文**: [arXiv:2601.18418](https://arxiv.org/abs/2601.18418) (Jan 2026)
**机构**: SII + SJTU + GAIR (刘鹏飞组)
**开源**: [Code](https://github.com/GAIR-NLP/daVinci-Dev) | [Models](https://huggingface.co/GAIR/daVinci-Dev-72B) | [Datasets](https://huggingface.co/datasets/GAIR/daVinci-Dev)
**Base Model**: Qwen2.5-Base (32B / 72B) — 注意不是 Coder 版本

---

## 1. 核心问题与动机

### 1.1 Post-training 的瓶颈

当前 coding agent 的主流训练路线：SFT on curated trajectories → RL from execution feedback。但存在根本局限：

| 问题 | 详细说明 |
|------|---------|
| **数量受限** | 能构建 executable 环境的 repo 少（README 不清晰、GPU 依赖等），且多数环境跑不通 |
| **多样性差** | 高级 agent 也只能解决小部分 issue，大量环境训练前就被过滤掉 |
| **能力天花板** | Post-training 受 base model 内在能力约束，某些 agentic reasoning 能力仅靠 post-training 学不到 |

→ **核心问题**: 能否在 mid-training 阶段就注入 foundational agentic behaviors？

### 1.2 Distribution Mismatch（关键 insight）

传统训练数据 vs. agent 部署时的分布不匹配：

```
传统训练数据:  静态代码快照 → 完整文件 / merged commits / 最终实现
Agent 部署:    动态交互序列 → localize → read → edit → test → revise 循环
```

**GitHub PR 的局限**：commit history 显示了 *what* 被改了，但隐藏了 *how* — 开发者如何发现相关文件、什么上下文驱动了编辑决策、test feedback 如何引导后续修改。

即使保留了 workflow 结构，训练数据通常也只展示成功的最终状态，忽略了 validation failures、error messages、iterative refinements — 这些恰恰是 agent 需要学的。

---

## 2. 方法论：Agent-Native Data

核心 thesis: **有效的 agentic mid-training 需要 large-scale + diverse 的 agent-native data** — 保留 agent 在 deployment 时体验的完整信息流和环境动态。

### 2.1 两种互补轨迹

| | Contextually-Native (D^ctx) | Environmentally-Native (D^env) |
|---|---|---|
| **优化目标** | Coverage & Diversity | Interaction Authenticity |
| **数据来源** | GitHub PR 的 metadata + files + commits | Docker 环境中的 agent rollout |
| **规模** | 68.6B tokens (10M PRs) | 3.1B tokens (7.4万条轨迹) |
| **包含内容** | Issue + 相关文件 + commit edits + LLM summary | Tool invocations + test executions + error messages |
| **关键特征** | 保留 localize→read→edit 的因果流 | 保留 edit→test→revise 的反馈循环 |
| **作用** | 教模型 *知识* — 软件工程的修改模式和多样性 | 教模型 *动态* — 如何与环境交互 |

**互补关系的精髓**: D^env 教模型 agent 交互的正确 *格式*，D^ctx 提供泛化所需的 *知识*。单独用 D^env 效果有限（Table 4: 47.1%），加上 D^ctx_py 后跳到 54.8%（+7.7%）。

### 2.2 Contextually-Native Trajectories 构建

#### 数据来源
- **D^ctx_gen** (26.7B tokens): top 10K starred repos，全语言，建立跨语言软件工程理解
- **D^ctx_py** (41.9B tokens): 7.4×10^5 个 Python repos（≥5 stars），与 SWE-Bench 对齐

#### Construction Pipeline

```
GitHub PR → Collection → Filtering → Reconstruction → Training Sample
                 │              │              │
                 ↓              ↓              ↓
           PR metadata    质量过滤:        3步重构:
           File contents  - merged PRs    1. Content Enhancement (Qwen3-235B生成摘要+优化commit msg)
           Linked issues  - 非bot生成      2. Relevant File Identification (逆向分析diff→文件集)
           Commit seq     - Python文件1-5  3. Template Organization (Issue→Files→Summary→Edits)
```

#### 关键设计决策
- **不做 factorization**: 不把 PR 拆成独立的 localization + editing 子任务（对比 Kimi-Dev）
- **保持因果完整性**: issue description + relevant files + edits 打包成一个 sample
- **两种格式**: D^ctx_gen 用 XML-like tags（类似 The Stack v2），D^ctx_py 用 Markdown + search-and-replace（模拟 agentic scaffold）
- **长度限制**: 丢弃 >32K tokens 的样本（保留 90%+ Python PRs）
- **去污染**: 移除所有 SWE-Bench Verified 涉及的 repo 的 PRs

### 2.3 Environmentally-Native Trajectories 构建

#### 关键区别
与模拟环境/只读环境中的 trajectory 对比（如 Kimi-Dev）：
- **真实 executable 环境**: 从真实 GitHub PR 构建 Docker image（follow SWE-REBENCH 方法论）
- **完整反馈循环**: agent 的 tool invocations、test executions、error messages 全部记录
- **不做过滤**: passing 和 non-passing 的轨迹都保留 — 失败轨迹同样提供学习信号

#### Pipeline
```
Real GitHub PR → Docker Image (SWE-REBENCH) → GLM-4.6 + SWE-Agent → 最多4次 rollout → 记录完整 action-observation 序列
```

#### 数据统计
- **Passing trajectories** D^env_pass: 1.85×10^4 条 (0.7B tokens) — 所有 test 通过
- **Non-passing trajectories** D^env_fail: 5.55×10^4 条 (2.4B tokens) — test 有失败
- 丢弃 >128K tokens 的轨迹
- 训练时 D^env_pass 上采样 3×

---

## 3. 训练细节

### 3.1 训练阶段

```
Pre-training (Qwen2.5-Base)
    ↓
Mid-training (MT): agent-native data, 1 epoch
    ↓
Post-training (SFT): agentic trajectories, 5 epochs
```

### 3.2 Mid-Training 配置

| 参数 | 值 |
|------|-----|
| Global batch size | 1024 |
| Peak LR | 8×10^-5 |
| Warmup ratio | 0.05 (5% of total steps) |
| Schedule | Cosine decay |
| Loss mask | None (全 token 都计算 loss) |
| Epochs | 1 |

**分阶段训练**: D^ctx 先 general (26.7B) 再 Python (41.9B)；D^ctx + D^env 也是 general 先，其余后。

### 3.3 SFT 配置

| 参数 | 值 |
|------|-----|
| Global batch size | 128 |
| Peak LR | 1×10^-5 |
| Warmup ratio | 0.10 |
| Loss mask | Standard (user + tool tokens masked) |
| Epochs | 5 |

### 3.4 MT 不用 loss mask 的设计理由

Mid-training 阶段 **不 mask** 任何 token — 与 SFT 不同。这意味着模型需要学习预测整个序列（包括 tool output、file contents），而不只是 agent response。这类似于 continued pre-training 的做法，目的是让模型从整个上下文中学习，而非仅学 "如何回复"。

---

## 4. 实验结果

### 4.1 主实验 (SWE-Bench Verified)

| Model | MT Data | SFT Data | Method | SWE-V |
|-------|---------|----------|--------|-------|
| **Qwen2.5-32B Series** |
| Baseline (Weak SFT) | - | D^SWE-smith | SFT | 34.8 |
| Baseline (Strong SFT) | - | D^env_pass | SFT | 53.0 |
| Ours (Weak SFT) | D^ctx | D^SWE-smith | SFT | 39.5 |
| Ours (Strong SFT) | D^ctx | D^env_pass | SFT | 54.1 |
| **daVinci-Dev-32B** | **D^ctx + D^env** | **D^env_pass** | **SFT** | **56.1** |
| **Qwen2.5-72B Series** |
| Baseline (Weak SFT) | - | D^SWE-smith | SFT | 38.0 |
| Baseline (Strong SFT) | - | D^env_pass | SFT | 56.6 |
| Kimi-Dev (72B) | D^AgentlessMT | D^SWE-smith | SFT | ≈46.0 |
| Kimi-Dev (72B) + RL | D^AgentlessMT | D^AgentlessRL + D^env_pass | SFT+RL | 48.6 |
| Ours (Strong SFT) | D^ctx | D^env_pass | SFT | **58.2** |
| **daVinci-Dev-72B** | **D^ctx + D^env** | **D^env_pass** | **SFT** | **58.5** |

**关键观察**:
- 仅用 D^ctx MT + Strong SFT 就达到 **58.2%**，超过 Kimi-Dev 的 SFT+RL (48.6%)
- D^env 在 72B 上额外贡献 +0.3%（从 58.2→58.5），但在 32B 上贡献更大 +2.0%（54.1→56.1）
- **token 效率**: 73.1B tokens vs Kimi-Dev ~150B tokens，不到一半

### 4.2 与 Open Recipes 对比 (Table 2)

| Model | Base | MT | Post | Scaffold | SWE-V |
|-------|------|-----|------|----------|-------|
| SWE-Mirror-LM (32B) | Inst. | No | SFT | MOpenHands | 52.2 |
| FrogBoss (32B, Qwen3) | Inst. | No | SFT | SWE Agent | 54.6 |
| SWE-Lego-Qwen3-32B | Inst. | No | SFT | OpenHands | 52.6 |
| **daVinci-Dev-32B** | **Base** | **Yes** | **SFT** | **SWE-Agent** | **56.1** |
| Kimi-Dev (72B) | Base | Yes | SFT+RL | SWE-Agent | 48.6 |
| **daVinci-Dev-72B** | **Base** | **Yes** | **SFT** | **SWE-Agent** | **58.5** |

注意 daVinci-Dev 用的是 **non-coder Base model** (Qwen2.5-Base)，而其他方法多用 Instruct 或 Coder 版本。

### 4.3 泛化能力 (Table 3)

MT mix (D^ctx_py + D^env) 不仅提升 agentic coding，还泛化到其他领域：

| Benchmark | 32B Base → MT | 72B Base → MT |
|-----------|---------------|---------------|
| HumanEval | 58.16 → 81.42 (+23.26) | 64.27 → 76.73 (+12.46) |
| EvalPlus | 50.13 → 71.31 (+21.18) | 56.04 → 69.45 (+13.41) |
| GPQA-Main | 38.17 → 38.84 (+0.67) | 43.30 → 44.87 (+1.57) |
| SciBench | 18.46 → 20.49 (+2.03) | 19.33 → 19.77 (+0.44) |

Code generation 提升巨大，科学推理也有小幅提升 — 说明 agentic decision-making patterns 有跨领域迁移能力。

---

## 5. 关键分析

### 5.1 Token 效率

| Method | MT Tokens | SWE-V (72B) |
|--------|-----------|-------------|
| Kimi-Dev | ~150B (70B raw + 20B synth ×4 upsample) | 48.6 |
| daVinci-Dev | **73.1B** | **58.5** |

效率来源：
1. Contextually-native 表示比 factorized 方法更接近 agent test distribution
2. Environmentally-native 轨迹比模拟轨迹更 authentic

### 5.2 Synergy: D^ctx 放大 D^env 的效果 (Table 4)

| MT Data | Tokens | SWE-V (32B) | SWE-V (72B) |
|---------|--------|-------------|-------------|
| **Zero-shot (No SFT)** |
| D^env alone | 4.5B | 43.7 | 47.1 |
| D^env + D^ctx_py | 46.4B | 49.9 | 54.8 |
| **With SFT** |
| D^ctx_py only | 41.9B | 52.9 | 56.5 |
| D^env + D^ctx_py | 46.4B | 53.6 | 57.8 |
| D^env + D^ctx (full) | 73.1B | **56.1** | **58.5** |

**核心发现**:
- D^env 单独 zero-shot 47.1% (72B)，加 D^ctx_py 后 **+7.7%** → D^ctx 提供修改模式的知识多样性
- MT 中包含 trajectory 后再 SFT 仍有提升（56.5→57.8），说明 MT 和 SFT 的 trajectory "double-dipping" 是有价值的 — MT 建立更深层的环境动态内化

### 5.3 Scaling Law

SWE-V 与 training steps 呈 **log-linear** 关系 (R² ≈ 0.90 for 72B, 0.89 for 32B)：
- 72B 模型沿训练持续攀升到 54.9% (zero-shot)
- 32B 模型平行到 49.9%
- **未饱和**: 性能仍在单调上升，继续训练应可持续提升

### 5.4 数据可扩展性

| 维度 | 当前规模 | 理论上限 |
|------|---------|---------|
| D^ctx_py | 1.3×10^7 PRs from 7.4×10^5 repos | ~3×10^8 PRs from ~10^9 public repos |
| D^ctx_gen | top 10K repos | 可扩展到更多 repos 和更多语言 |
| D^env | 基于 SWE-REBENCH 的 3,468 repos 21,336 tasks | 随 repo coverage 扩展而扩展 |

---

## 6. 与 Kimi-Dev 的对比

| 维度 | daVinci-Dev | Kimi-Dev |
|------|-------------|----------|
| **MT 数据范式** | Agent-native (保留因果流) | Factorized (localization 和 editing 分开训练) |
| **MT 数据量** | 73.1B tokens | ~150B tokens |
| **Post-training** | SFT only | SFT + RL |
| **Base model** | Qwen2.5-Base (非 coder) | Qwen2.5-Base |
| **SWE-V 结果** | 58.5% (72B) | 48.6% (72B, SFT+RL) |
| **核心差异** | 保留完整 action-observation loop | 不保留 agent 交互动态 |

Kimi-Dev 的 factorized approach 把 file retrieval 和 file editing 当成独立子任务训练 — 这破坏了 localization 和 editing 之间的依赖关系。daVinci-Dev 论证了**保持完整流程**（从 issue 到定位到编辑）是更高效的。

---

## 7. Limitations

1. **隐私问题**: D^ctx_gen 中未去除开发者标识，可能导致记忆贡献者姓名
2. **评估敏感性**: 使用了 patched evaluation harness 修复部分 benchmark 问题
3. **Scope**: 仅测了 Qwen2.5 系列和 SWE-Bench Verified 一个 benchmark

---

## 8. 核心 Takeaways & 思考

### 8.1 方法论启示

1. **Mid-training 是被低估的阶段**: 在 pre-training 和 post-training 之间注入 domain-specific structured data 可以大幅提升下游能力，且比 RL 更可扩展
2. **数据表示比数据量更重要**: 73B agent-native tokens > 150B factorized tokens — 保留因果结构的数据每个 token 承载更多信息
3. **Contextual + Environmental = 最佳组合**: 一个提供知识广度，一个提供交互深度；单独都不够
4. **Non-passing trajectories 有价值**: 不过滤失败轨迹，让模型学习错误恢复

### 8.2 对 VLM Agent 训练的启发

- **同样的 distribution mismatch 存在于 VLM agent**: VLM agent 在 GUI/web 环境中也需要 navigate → observe → act → verify 循环
- **Mid-training 范式可迁移**: 用真实的 multimodal interaction traces（而非 factorized screenshots + actions）做 mid-training
- **PR 数据的类比**: 对 VLM agent 可以收集真实用户操作序列（screen recordings → action sequences）做类似的 contextually-native 数据

### 8.3 Open Questions

1. **RL 会在这个 MT base 上带来多大额外收益？** 论文只做了 SFT，但 OctoThinker 论证了好的 MT 能提升 RL 的 sample efficiency 和 ceiling
2. **这个方法对 reasoning-heavy 任务的泛化？** GPQA 只提升了 ~1.5%，说明 agentic mid-training 主要提升的是 procedural/tool-use 能力而非 deep reasoning
3. **Trajectory 质量 vs 数量的 tradeoff？** D^env 用 GLM-4.6 生成，换更强的 model 做 rollout 会怎样？
4. **能否用同样的方法为非 SWE 的 agentic tasks 做 mid-training？** 比如 research agent、data analysis agent
