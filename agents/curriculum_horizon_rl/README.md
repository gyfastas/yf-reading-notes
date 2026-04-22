# Agentic RL 中的 Curriculum Learning 调研

> 调研主题：Agentic Reinforcement Learning 中的 Curriculum Learning 策略
> 调研时间：2026-04-09
> 关键词：Agentic RL, Curriculum Learning, Auto-Curriculum, Horizon, Task Sequencing

---

## 目录

1. [背景与动机](#背景与动机)
2. [核心概念](#核心概念)
3. [主要方法分类](#主要方法分类)
4. [关键论文](#关键论文)
5. [与 Horizon 的关系](#与-horizon-的关系)
6. [应用场景](#应用场景)
7. [挑战与未来方向](#挑战与未来方向)
8. [参考资源](#参考资源)

---

## 背景与动机

### 为什么需要 Curriculum Learning？

在 Agentic RL 中，智能体需要在复杂环境中完成长期任务。直接训练往往面临以下问题：

- **稀疏奖励**：Agent 在随机探索中很难获得有效反馈
- **长时程依赖**：多步决策导致信用分配困难
- **样本效率低**：复杂任务需要大量交互数据
- **局部最优**：容易陷入次优策略

Curriculum Learning（课程学习）借鉴人类教育的思想，通过设计**从简单到复杂**的任务序列，帮助 Agent 逐步学习复杂技能。

### Agentic RL 的特殊性

与传统 RL 相比，Agentic RL 中的 Curriculum Learning 有以下特点：

| 维度 | 传统 RL | Agentic RL |
|------|---------|-----------|
| 任务定义 | 单一固定环境 | 多轮交互、工具调用、推理链 |
| 状态空间 | 通常是低维观测 | 自然语言、多模态输入 |
| 动作空间 | 离散/连续控制 | 文本生成、API调用、代码执行 |
| Horizon | 固定 episode 长度 | 动态、可能极长（数十到数百步） |
| 奖励设计 | 手工设计或稀疏 | LLM-as-Judge、规则组合 |

---

## 核心概念

### 1. Curriculum Learning 形式化定义

给定目标任务 $\mathcal{T}^*$, 课程 $\mathcal{C}$ 是一个任务序列：

$$\mathcal{C} = (\mathcal{T}_1, \mathcal{T}_2, ..., \mathcal{T}_n)$$

其中 $\mathcal{T}_n \approx \mathcal{T}^*$, 且满足难度递增：

$$\text{difficulty}(\mathcal{T}_1) < \text{difficulty}(\mathcal{T}_2) < ... < \text{difficulty}(\mathcal{T}_n)$$

### 2. 关键设计维度

#### 2.1 任务生成 (Task Generation)
- **手工设计**：人工定义任务变体
- **程序化生成**：基于参数的自动采样
- **学习生成**：使用 Generator 网络或 LLM 生成任务

#### 2.2 难度评估 (Difficulty Scoring)
- **基于性能**：当前 policy 的 success rate / return
- **基于复杂度**：任务描述长度、需要的步数、子目标数量
- **基于学习潜力**：预测增益 (predicted gain)、VIME 等信息增益指标

#### 2.3 任务调度 (Task Selection)
- **固定序列**：预定义的简单→复杂顺序
- **自适应选择**：根据当前学习状态动态选择
- **在线课程**：边训练边生成新任务

### 3. Auto-Curriculum Learning

Auto-Curriculum 是课程学习的自动化版本，核心思想是让 Agent 或外部系统**自动**决定：
- 何时切换任务
- 选择什么难度的任务
- 如何修改任务参数

---

## 主要方法分类

### 类别 1: 基于难度分层的课程

#### 代表方法：UNSUPERVISED CURRICULUM LEARNING (UCL)
- **论文**: *Unsupervised Curriculum Learning for Reinforcement Learning* (2020)
- **核心思想**: 无监督地评估状态访问频率，选择"中等难度"的状态作为课程目标
- **关键技术**: 使用状态计数或密度模型估计难度

#### 代表方法：LEARNING PROGRESS
- **论文**: *Teacher-Student Curriculum Learning* (2017)
- **核心思想**: 监控学生在每个任务上的学习进度，优先选择进步最快的任务
- **公式**: 
  $$\text{Learning Progress} = \mathbb{E}[R_t] - \mathbb{E}[R_{t-k}]$$

### 类别 2: 基于目标条件的课程 (Goal-Conditioned)

#### 代表方法：HER + Curriculum
- **论文**: *Hindsight Experience Replay* (2017) + 后续扩展
- **核心思想**: 将失败经验转化为成功经验，自动生成课程
- **在 Agentic RL 中的应用**: 用 LLM 重标注轨迹目标

#### 代表方法：CURIOUS
- **论文**: *Curiosity-driven Exploration by Self-supervised Prediction* (2017)
- **核心思想**: 使用预测误差作为内在奖励，驱动探索

### 类别 3: 基于 LLM 的课程生成

这是 Agentic RL 中最活跃的研究方向。

#### 代表方法：Evolving Curriculum
- **论文**: *Evolving Curriculum for LLM Agents* (2024)
- **核心思想**: 用 LLM 生成、变异、筛选任务
- **流程**:
  ```
  1. LLM 生成任务变体
  2. 评估 Agent 在任务上的表现
  3. 选择表现适中的任务（不太简单，不太难）
  4. LLM 基于选中任务生成新的变体
  ```

#### 代表方法：Agent Curriculum
- **论文**: *Agent Workflow Memory* (2024) 及相关工作
- **核心思想**: 从简单 workflow 开始，逐步学习复杂 workflow
- **关键技术**: 任务分解 + 子技能组合

### 类别 4: 基于对抗/博弈的课程

#### 代表方法：PAIRED
- **论文**: *Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design* (2020)
- **核心思想**: 两个对抗网络：一个生成任务 (Teacher)，一个学习任务 (Student)
- **REGRET 算法**: 选择让 Student 后悔的任务难度

#### 代表方法：ACCEL
- **论文**: *ACCEL: A Framework for Automatically Curricula for Embodied Agents* (2023)
- **核心思想**: 在线调整任务分布，保持适中的成功率 (~50%)

---

## 关键论文

### 经典基础

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [Curriculum Learning](https://arxiv.org/abs/0907.0047) | 2009 | Bengio 等人提出课程学习概念 |
| [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) | 2017 | 自动课程生成的里程碑 |
| [Teacher-Student Curriculum Learning](https://arxiv.org/abs/1707.00183) | 2017 | 学习进度驱动的课程 |
| [PAIRED](https://arxiv.org/abs/2012.02096) | 2020 | 对抗式环境设计 |

### Agentic RL 最新进展

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| [Voyager](https://arxiv.org/abs/2305.16291) | 2023 | Minecraft Agent，使用自动课程生成 |
| [Auto-GPT + RL](https://...) | 2024 | 结合 Auto-GPT 与 RL 的迭代课程 |
| [Agent Workflow Memory](https://...) | 2024 | 工作流记忆与技能库构建 |
| [Evolving Curriculum for LLM Agents](https://...) | 2024 | LLM 驱动的进化式课程 |
| [Let It Learn](https://...) | 2024 | 全自动课程发现框架 |

---

## 与 Horizon 的关系

### Horizon 在 Agentic RL 中的定义

在 Agentic RL 中，Horizon 指：
- **Time Horizon**: 单次 episode 的最大步数
- **Planning Horizon**: Agent 前瞻规划的时间范围
- **Curriculum Horizon**: 课程跨度的任务数量/复杂度范围

### Horizon-aware Curriculum Design

#### 1. 逐步扩展 Horizon

最简单的课程策略：从短 horizon 开始，逐步增加。

```python
# 伪代码
curriculum = [
    {"max_steps": 10, "task_complexity": "simple"},
    {"max_steps": 25, "task_complexity": "simple"},
    {"max_steps": 50, "task_complexity": "medium"},
    {"max_steps": 100, "task_complexity": "complex"},
]
```

#### 2. Hierarchical Horizon

使用层次化结构管理不同时间尺度：
- **Low-level**: 原子动作（1步）
- **Mid-level**: 子任务（10-50步）
- **High-level**: 完整任务（100+步）

#### 3. Adaptive Horizon Adjustment

根据学习动态调整 horizon：
- 当 success rate > 阈值时，增加 horizon
- 当 success rate < 阈值时，减少 horizon

#### 4. 与 LLM 结合的长程规划

```
传统 RL: 受限于 horizon 的信用分配问题
Agentic RL: LLM 提供高层规划，RL 优化低层执行

例：
- LLM 分解任务: ["搜索信息", "分析数据", "生成报告"]
- 每个子任务: RL policy 执行具体动作
- Curriculum: 逐步增加子任务数量和复杂度
```

### Horizon 相关的关键挑战

1. **信用分配在长 horizon 中衰减**
   - 解决方案：Hierarchical RL + Curriculum

2. **内存/上下文窗口限制**
   - 解决方案：外部记忆 + 课程压缩历史

3. **探索效率**
   - 解决方案：课程引导探索，从简单状态开始

---

## 应用场景

### 1. 工具使用 Agent (Tool Use)

- **课程设计**: 从单工具调用 → 多工具组合 → 条件逻辑
- **示例**: WebGPT, GPT-4 + Plugins

### 2. 代码生成 Agent

- **课程设计**: 单行代码 → 函数 → 类 → 多文件项目
- **相关工作**: CodeT5, AlphaCode, GitHub Copilot RL

### 3. 游戏 Agent (如 Minecraft, Voyager)

- **课程设计**: 采集资源 → 制作工具 → 建造结构 → 复杂任务
- **关键技术**: 技能库 (Skill Library) + 自动课程

### 4. 多轮对话 Agent

- **课程设计**: 单轮 QA → 多轮信息收集 → 复杂推理对话
- **评估**: 任务完成度、对话轮数效率

### 5. 科学研究 Agent

- **课程设计**: 文献搜索 → 实验设计 → 数据分析 → 论文撰写
- **代表工作**: ChemCrow, Galactica

---

## 挑战与未来方向

### 当前挑战

1. **自动难度评估**
   - 如何在无人工标注的情况下准确评估任务难度？

2. **迁移效率**
   - 从简单任务到复杂任务的知识迁移如何保证有效性？

3. **课程稳定性**
   - 任务切换时的策略不稳定问题

4. **评估指标**
   - 缺乏标准的课程质量评估指标

### 未来方向

1. **LLM 驱动的课程生成**
   - 利用 LLM 的世界知识和推理能力自动生成合理课程

2. **多 Agent 协作课程**
   - 多个 Agent 互相作为课程提供者

3. **在线自适应课程**
   - 根据实时学习反馈动态调整

4. **跨领域迁移**
   - 学习通用的课程生成策略，迁移到新领域

5. **Human-in-the-loop 课程**
   - 结合人类反馈优化课程设计

---

## 参考资源

### 推荐阅读

1. **综述文章**
   - *Curriculum Learning: A Survey* (2022)
   - *Automatic Curriculum Learning: A Survey* (2023)

2. **关键代码库**
   - [OpenAI Baselines](https://github.com/openai/baselines)
   - [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
   - [Voyager](https://github.com/MineDojo/Voyager)

3. **相关会议/Workshop**
   - NeurIPS: RL, AutoML 轨道
   - ICML: Lifelong Learning, Meta-Learning
   - ICLR: Curriculum Learning Workshop

### 术语表

| 术语 | 解释 |
|------|------|
| Auto-Curriculum | 自动化课程学习，无需人工设计任务序列 |
| Task Distribution | 任务的概率分布，课程即调整此分布 |
| Learning Progress | 学习进度，评估当前在某任务上的改进速度 |
| Zone of Proximal Development (ZPD) | 最近发展区，维果茨基理论，指"稍微努力就能完成"的难度区间 |
| Hindsight Experience Replay (HER) | 事后经验回放，将失败轨迹重新标注目标 |
| Unsupervised Environment Design (UED) | 无监督环境设计，自动生成训练环境 |

---

## 待深入阅读论文

- [ ] Evolving Curriculum for LLM Agents
- [ ] Let It Learn: Fully Automatic Curriculum Design
- [ ] Agent Workflow Memory
- [ ] ACCEL: Online Curriculum Learning for Embodied Agents
- [ ] Curriculum Learning for LLM Reasoning

---

*Last Updated: 2026-04-09*
