# Dr. MAS: Stable Reinforcement Learning for Multi-Agent LLM Systems

**论文链接**: https://arxiv.org/abs/2602.08847
**作者**: Lang Feng, Longtao Zheng, Shuo He, Fuxiang Zhang, Bo An (NTU)
**发表日期**: 2026年2月9日
**代码**: https://github.com/langfengQ/DrMAS

---

## 1. 核心贡献速览

### 1.1 研究背景与问题

多智能体LLM系统（Multi-Agent LLM Systems, MAS）通过角色专业化实现复杂推理和工具使用，但RL后训练存在严重的**训练不稳定性**问题。

**核心问题**: 将GRPO（Group Relative Policy Optimization）直接扩展到多智能体场景时，会出现梯度范数爆炸（gradient-norm explosion），导致训练崩溃。

### 1.2 Dr. MAS的核心解法

**Agent-wise Advantage Normalization**: 每个智能体使用自身的奖励统计量（均值µk、标准差σk）进行优势归一化，而非全局统计量。

**效果**: 在数学推理和搜索任务上显著优于vanilla GRPO：
- Math: +5.6% avg@16, +4.6% pass@16
- Search: +15.2% avg@16, +13.1% pass@16
- 基本消除梯度尖峰

---

## 2. Agentic RL与End-to-End Multi-Agent System知识梳理

### 2.1 Agentic RL基础概念

**Agentic RL** 指的是让LLM作为智能体（agent）在环境中自主决策、执行动作、接收反馈并通过强化学习优化策略的范式。

| 特性 | 传统RLHF | Agentic RL |
|------|---------|-----------|
| 奖励来源 | 人类偏好/RM模型 | 可验证信号（答案正确性） |
| 动作空间 | 文本生成 | 工具调用、多轮交互、代码执行 |
| 轨迹长度 | 单轮/短序列 | 多轮、长程决策 |
| 典型方法 | PPO、DPO | GRPO、RLVR |

**RLVR (Reinforcement Learning from Verifiable Rewards)**: 利用可自动验证的信号（如数学题答案、代码执行结果）进行RL训练，无需人工标注偏好。

### 2.2 End-to-End Multi-Agent System架构

```
┌─────────────────────────────────────────────────────────────┐
│                 End-to-End Multi-Agent System               │
├─────────────────────────────────────────────────────────────┤
│  Multi-Agent Orchestration (多智能体编排层)                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ Agent 1 │───→│ Agent 2 │───→│ Agent 3 │                 │
│  │ Solver  │    │Verifier │    │ Answer  │                 │
│  └─────────┘    └─────────┘    └─────────┘                 │
│       ↑              ↑              ↑                      │
│  ┌────┴────┐    ┌────┴────┐    ┌────┴────┐                │
│  │Policy π₁│    │Policy π₂│    │Policy π₃│                │
│  │ θ₁(7B)  │    │ θ₂(7B)  │    │ θ₃(3B)  │                │
│  └─────────┘    └─────────┘    └─────────┘                │
├─────────────────────────────────────────────────────────────┤
│  RL Training Layer (RL训练层)                               │
│  ┌─────────────────────────────────────────┐                │
│  │  Trajectory Collector → RL Trainer      │                │
│  │  (GRPO/Dr.MAS optimization)             │                │
│  └─────────────────────────────────────────┘                │
├─────────────────────────────────────────────────────────────┤
│  Environment (环境层)                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  Math Env   │    │ Search Env  │    │  Tool Env   │      │
│  │  (可验证)    │    │ (检索API)   │    │ (函数调用)   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Group-Based RL方法家族

| 方法 | 核心思想 | 特点 |
|------|---------|------|
| **GRPO** (DeepSeek) | 组内相对优势归一化 | 无需critic，group内比较 |
| **RLOO** | REINFORCE Leave-One-Out | leave-one-out baseline |
| **Dr.GRPO** | 去偏GRPO | 解决token长度偏差 |
| **DAPO** | 分布式自适应策略优化 | 裁剪、过滤、长度惩罚 |
| **GSPO** | Group-Similarity PO | 考虑group内相似性 |
| **GiGPO** | Group-in-Group PO | 嵌套group结构 |

### 2.4 Multi-Agent RL训练的关键挑战

1. **异构数据分布**: 不同智能体被调用频率不同，样本量|Yk|差异大
2. **角色分化**: Solver/Verifier/Answer等不同角色的奖励分布差异显著
3. **信用分配**: 多轮交互中如何归因最终奖励到各智能体的动作
4. **系统复杂性**: 多模型协同训练、资源调度、编排管理

---

## 3. Dr. MAS算法详解

### 3.1 问题诊断：为什么GRPO在多智能体场景不稳定？

**GRPO的全局归一化公式**:
```
A^i_global = (R^i - µ) / σ

其中: µ = (1/N) Σ R^i,  σ² = (1/N) Σ(R^i - µ)²
```

**问题分析**:

在多智能体系统中，不同智能体的奖励分布差异很大：
- **Solver**: 直接生成答案，奖励方差大（对/错）
- **Verifier**: 判断逻辑，奖励相对稳定
- **Searcher**: 检索信息，奖励取决于检索质量

当使用全局baseline (µ, σ) 时：
- 某些智能体的奖励分布远高于/低于全局均值 → 优势估计系统性偏差
- 方差不匹配 → 梯度范数爆炸

**理论分析 (Lemma 4.2)**:

智能体k的梯度二阶矩：
```
E[||g̃^global_k||²] = E[||z||²] × (σ²_k + (µ_k - µ)²) / σ² + Δ_k
                              ↑
                    这是导致不稳定的罪魁祸首！
```

**Gradient-Norm Inflation (Proposition 4.3)**:

当以下任一条件发生时，梯度范数会线性增长甚至爆炸：
- |µ_k - µ|/σ → ∞ (均值偏离)
- σ²_k/σ² → ∞ (方差比例过大)

### 3.2 Dr. MAS的解决方案

**Agent-wise Normalization**:
```
A^i,k_agent = (R^i - µ_k) / σ_k

其中: µ_k = (1/|Yk|) Σ_{a∈Yk} R,  σ²_k = (1/|Yk|) Σ(R - µ_k)²
```

**关键改进**: 每个智能体只在自己被激活的steps上计算统计量

**归一化效果**:
```
E[||g̃^agent_k||²] = E[||z||²] + Δ_k
                    ↑
         不再有(σ²_k + (µ_k - µ)²)/σ²这个放大因子！
```

### 3.3 算法伪代码

```python
# Dr. MAS 核心算法

def compute_dr_mas_advantages(trajectories, agent_ids):
    """
    trajectories: List of N trajectories
    agent_ids: List of K agent identifiers
    """
    advantages = {}

    for k in agent_ids:
        # 收集智能体k的所有奖励
        Y_k = [R_i for i, traj in enumerate(trajectories)
               if agent_k_activated_in(traj, k)]

        # 计算智能体k自身的统计量
        mu_k = mean(Y_k)
        sigma_k = std(Y_k)

        # 计算agent-wise优势
        for i, R_i in enumerate(trajectory_rewards):
            if agent_k_activated_in(trajectories[i], k):
                A_ik = (R_i - mu_k) / (sigma_k + eps)
                advantages[(i, k)] = A_ik

    return advantages

def dr_mas_loss(policy_k, old_policy_k, trajectories_k, advantages_k):
    """PPO-clip with agent-wise advantages"""
    ratio = policy_k / old_policy_k

    clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)

    loss = -min(ratio * advantages_k, clipped_ratio * advantages_k)

    return loss.mean()
```

### 3.4 消融实验验证

| 归一化配置 | avg@16 | pass@16 | 说明 |
|-----------|--------|---------|------|
| (µ, σ) GRPO | 28.0 | 40.5 | 基线，性能最差 |
| (µ_k, σ) | 39.1 ↑11.1 | 53.5 ↑13.0 | 仅per-agent均值 |
| (µ, σ_k) | 42.9 ↑14.9 | 57.6 ↑17.1 | 仅per-agent标准差 |
| **(µ_k, σ_k) Dr.MAS** | **43.8** ↑15.8 | **58.3** ↑17.8 | **完整方案** |

**结论**: per-agent标准差(σ_k)的贡献比per-agent均值(µ_k)更大，因为不同角色的奖励spread差异更大。

---

## 4. Dr. MAS系统框架

### 4.1 系统架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dr. MAS Framework                            │
├─────────────────────────────────────────────────────────────────┤
│  Multi-Agent Orchestration (多智能体编排层)                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  User-defined Agent Workflow                            │    │
│  │  - Math: Solver ↔ Verifier loop                         │    │
│  │  - Search: Verifier → Searcher → Answer                 │    │
│  └────────────────────────────────────────┬────────────────┘    │
│                                           ↓                      │
│  Agent-to-WorkerGroup Mapping (智能体→工作组映射)               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  agent_id 1 ──→ wg_id 1 (Config 1)                      │    │
│  │  agent_id 2 ──→ wg_id 2 (Config 2)  ← LLM Sharing?      │    │
│  │  agent_id 3 ──→ wg_id 2 (Config 2)  ← 共享或独立         │    │
│  └────────────────────────────────────────┬────────────────┘    │
│                                           ↓                      │
│  Shared Resource Pool (共享资源池)                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Ray Placement Group on GPU Nodes                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │ ActorRollout│  │ ActorRollout│  │ ActorRollout│     │    │
│  │  │   (wg 1)    │  │   (wg 2)    │  │   (wg 3)    │     │    │
│  │  │  SGLang引擎  │  │  SGLang引擎  │  │  SGLang引擎  │     │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘     │    │
│  └────────────────────────────────────────┬────────────────┘    │
│                                           ↓                      │
│  RL Trainer (PPO/GRPO优化器)                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  1. 收集轨迹 → 2. 按wg分组 → 3. 各组独立更新            │    │
│  │  B → B_wg1, B_wg2, ... → ∇θ_k for each wg              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 关键设计特性

| 特性 | 说明 |
|------|------|
| **LLM Sharing** | 多个智能体可共享同一模型参数（θ₁=θ₂=...），通过role-specific prompts区分 |
| **Non-Sharing** | 各智能体独立参数，支持异构模型（如Verifier用7B，Searcher用3B） |
| **Per-Agent Config** | 每个智能体可配置独立的lr、batch size、warmup steps等 |
| **Resource Pooling** | 使用Ray Placement Group实现GPU资源的动态调度 |
| **SGLang后端** | 高性能推理引擎，支持高并发、低延迟的decode |

### 4.3 实验中的两种典型编排

**Math推理 - Two-Agent Loop**:
```
Query → Solver(生成解法) → Verifier(评估)
               ↑____________↓ (若未通过则重新迭代)
                        ↓ (通过后输出)
                    Final Answer
```

**搜索任务 - Three-Agent Hierarchy**:
```
Query → Verifier(判断信息是否充足?)
           ├─ No  → Searcher(检索) → 返回Verifier
           └─ Yes → Answer(生成最终答案)
```

---

## 5. 实验结果深度分析

### 5.1 Math推理任务 (Qwen3-4B/8B)

| 配置 | avg@16 | pass@16 | 关键观察 |
|------|--------|---------|---------|
| Single-Agent GRPO | 55.9 | 70.9 | 基线 |
| Multi-Agent + GRPO (sharing) | 56.8 | 70.7 | 几乎无提升 |
| Multi-Agent + Dr.MAS (sharing) | 59.0 ↑2.2 | 73.2 ↑2.6 | 稳定提升 |
| Multi-Agent + GRPO (non-sharing) | 57.5 | 74.4 | 略有提升 |
| Multi-Agent + Dr.MAS (non-sharing) | **61.1 ↑3.6** | **77.7 ↑3.3** | **最佳效果** |

**关键发现**:
- Non-sharing设置下性能提升更显著（参数独立→分布差异更大→Dr.MAS作用更明显）
- AIME'24提升最大: 42.7/66.7 → 54.8/80.0 (+12.1/+13.3)

### 5.2 多轮搜索任务 (Qwen2.5-3B/7B)

**惊人发现**: Vanilla GRPO在non-sharing设置下会**完全崩溃**！

Qwen2.5-7B Non-Sharing:
- GRPO: 28.0/40.5 (甚至低于Single-Agent的42.1/55.6)
- Dr.MAS: 43.8/58.3 (+15.8/+17.8)

**原因分析**: 梯度范数爆炸导致Search Agent被"遗忘"——模型学会了完全不调用Search Agent来避免高方差更新。

### 5.3 梯度稳定性可视化

| Agent | GRPO梯度行为 | Dr.MAS梯度行为 |
|-------|-------------|---------------|
| Verifier | 多次尖峰(peak ~1.0) | 平滑稳定(~0.5) |
| Answer | 初期尖峰(peak ~1.0) | 平滑稳定(~0.6) |
| Search | **严重尖峰**(peak >6.0) | **平滑稳定**(peak ~2.0) |

Search Agent在GRPO下梯度范数可达6+，而Dr.MAS将其控制在2左右。

### 5.4 异构模型分配的效率收益

**配置对比**:
- Homogeneous: 全部使用7B模型
- Heterogeneous: Verifier用7B，Search/Answer用3B

| 指标 | Homogeneous | Heterogeneous | 节省 |
|------|------------|---------------|------|
| Performance | 42.5/57.7 | 42.0/57.5 | 基本相当 |
| Latency | 83.9s | 57.4s | **-31.6%** |
| Cost | 97.5$ | 56.7$ | **-41.8%** |

**洞察**: 在层级化MAS中，仅需在顶层Verifier使用大模型即可保证决策质量，底层Agent使用小模型可大幅降低成本。

---

## 6. 核心要点总结

### 6.1 理论贡献

1. **首次理论分析**了GRPO在多智能体场景下的不稳定性根源：全局归一化baseline与异构智能体奖励分布的错配
2. **严格证明**了梯度范数爆炸的条件：(σ²_k + (µ_k - µ)²)/σ² → ∞
3. **提出简单有效**的Agent-wise Normalization，有理论保证地消除了不稳定性

### 6.2 实践贡献

1. **端到端框架**: 支持LLM sharing/non-sharing、per-agent配置、资源池调度
2. **显著性能提升**: Math +5.6%, Search +15.2%
3. **训练稳定性**: 基本消除梯度尖峰
4. **异构支持**: 不同size模型协同训练，降低成本40%+

### 6.3 适用场景

✅ **强烈推荐使用Dr.MAS的场景**:
- 多角色协同的MAS（Solver/Verifier/Executor等）
- 异构模型配置（大模型决策+小模型执行）
- 长程多轮交互（搜索、工具调用）
- 训练过程中观察到梯度尖峰或崩溃

⚠️ **可能收益有限的场景**:
- 单智能体设置（退化为标准GRPO）
- 所有智能体共享同一模型且角色差异很小
- 极短程交互（单轮问答）

---

## 7. 与相关工作的关系

### 7.1 Multi-Agent RL方法对比

| 方法 | 特点 | 与Dr.MAS的关系 |
|------|------|---------------|
| SPIRAL/MARSHAL | 双人self-play | Dr.MAS支持更一般的多智能体协作 |
| Chain-of-Agents | 蒸馏为单智能体 | Dr.MAS坚持端到端多智能体训练 |
| M-GRPO | 多智能体GRPO扩展 | Dr.MAS提供理论分析和稳定性保证 |
| Heterogeneous Group RL | 异构组RL | Dr.MAS的归一化策略与之互补 |

### 7.2 基础设施对比

| 框架 | 多智能体支持 | Dr.MAS的优势 |
|------|------------|-------------|
| veRL | 有限 | 原生支持agent编排和资源池 |
| OpenRLHF | 单智能体为主 | 支持per-agent配置 |
| AReaL | 单智能体 | 支持LLM sharing/non-sharing |
| verl-agent | 支持agentic | 提供异构模型调度和共享资源池 |

**说明**: Dr.MAS 是独立的系统框架，与 verl-agent 是相关工作关系。根据论文描述，verl-agent 等现有框架"either provide limited support for heterogeneous model assignments or lack a shared resource pool"，而 Dr.MAS 通过 Agent-Model Assignment 和 Shared Resource Pooling 机制解决了这些问题。

---

## 8. 局限与未来方向

### 8.1 当前局限

1. **信用分配**: 未完全解决跨智能体、跨轮次的细粒度信用分配问题
2. **大规模MAS**: 未在>3个智能体的场景下充分验证
3. **异步执行**: 当前假设同步执行，大规模场景可能需要异步支持

### 8.2 未来研究方向

1. **更细粒度的优势估计**: 考虑step-level或token-level的信用分配
2. **自适应模型分配**: 动态决定哪个智能体使用什么规模的模型
3. **多模态扩展**: 扩展到视觉-语言多智能体系统
4. **在线学习**: 支持持续在线适应的MAS训练

---

## 9. 训练方式与数据集详解

### 9.1 Multi-Agent是一起训练的吗？

**是的，Dr. MAS是端到端（End-to-End）联合训练所有智能体。**

#### 训练流程细节

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Dr. MAS 联合训练流程                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  每轮训练迭代 (Training Iteration):                                      │
│                                                                         │
│  Step 1: 分布式Rollout采集                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Parallel for i = 1 to N (group size)                           │    │
│  │    1. 采样任务 x ~ p(X)                                          │    │
│  │    2. 运行多智能体编排 O 生成长序列交互轨迹 τ_i                   │    │
│  │       - Solver生成 → Verifier评估 → 可能重新迭代                  │    │
│  │       - 或: Verifier判断 → Searcher检索 → Answer生成            │    │
│  │    3. 轨迹级奖励 R_i = R(τ_i) (所有steps共享同一奖励)            │    │
│  │    4. 记录每个step: (agent_id, action, state, R_i)              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  Step 2: Agent-wise Advantage计算 (Dr.MAS核心)                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  for k = 1 to K (每个agent):                                     │    │
│  │    Y_k = {所有agent k被激活的steps}                              │    │
│  │    µ_k = mean(R_i for steps in Y_k)                              │    │
│  │    σ_k = std(R_i for steps in Y_k)                               │    │
│  │    A_i,k = (R_i - µ_k) / σ_k  (对每个step)                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                              ↓                                          │
│  Step 3: 按WorkerGroup分组更新 (联合训练的关键)                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  for each worker group wg_id:                                    │    │
│  │    B_wg = {属于该wg的所有agent的transitions}                     │    │
│  │    使用PPO-clip目标函数更新该wg对应的LLM参数                     │    │
│  │    (共享参数的agent会一起更新，非共享的独立更新)                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  关键点:                                                                │
│  - 所有agent在同一个rollout batch中交互                                 │
│  - 共享reward signal（轨迹级奖励）                                      │
│  - 但各自使用agent-wise normalized advantage                            │
│  - 共享模型的agent一起更新，非共享的独立更新                            │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Sharing vs Non-Sharing训练

| 设置 | 参数更新方式 | 适用场景 |
|------|-------------|---------|
| **LLM Sharing** | 所有agent共享同一套θ，一起更新 | 角色差异主要通过prompt区分，节省显存 |
| **Non-Sharing** | 每个agent有自己的θ_k，独立更新 | 需要真正独立的模型能力，异构模型配置 |

**注意**: 即使在Non-Sharing设置下，所有agent仍然是在同一个训练循环中、基于同一批交互数据进行联合优化的。

---

### 9.2 核心概念详解：Agent激活与统计量计算

#### 什么是"只在自己被激活的steps上计算统计量"？

这是Dr.MAS最核心的创新点。让我通过一个具体例子来说明：

##### 示例：Three-Agent Search任务的一次Rollout

假设我们有一个**Group Size = 4**（同时采样4条轨迹），每条轨迹最多4轮交互：

```
Task: "What government position was held by the woman who portrayed Corliss Archer?"

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         Group中的4条Trajectory                                        │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Trajectory 1 (成功，R=1):                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: Verifier判断 → "需要搜索"                                               │  │
│  │ Step 2: Searcher检索 → "Corliss Archer played by..."                           │  │
│  │ Step 3: Verifier判断 → "信息充足"                                               │  │
│  │ Step 4: Answer生成 → "Secretary of State"                                      │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│  Agent调用: [Verifier, Searcher, Verifier, Answer]                                  │
│                                                                                     │
│  Trajectory 2 (失败，R=0):                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: Verifier判断 → "需要搜索"                                               │  │
│  │ Step 2: Searcher检索 → "相关信息..."                                           │  │
│  │ Step 3: Verifier判断 → "需要更多信息"                                           │  │
│  │ Step 4: Searcher检索 → "Shirley Temple..."                                     │  │
│  │ Step 5: Verifier判断 → "信息充足"                                               │  │
│  │ Step 6: Answer生成 → "Ambassador" (错误答案)                                    │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│  Agent调用: [Verifier, Searcher, Verifier, Searcher, Verifier, Answer]              │
│                                                                                     │
│  Trajectory 3 (成功，R=1):                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: Verifier判断 → "需要搜索"                                               │  │
│  │ Step 2: Searcher检索 → "Shirley Temple was..."                                 │  │
│  │ Step 3: Verifier判断 → "信息充足"                                               │  │
│  │ Step 4: Answer生成 → "Chief of Protocol"                                       │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│  Agent调用: [Verifier, Searcher, Verifier, Answer]                                  │
│                                                                                     │
│  Trajectory 4 (失败，R=0):                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │ Step 1: Verifier判断 → "信息充足" (错误判断)                                    │  │
│  │ Step 2: Answer生成 → "Actress" (错误答案)                                       │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│  Agent调用: [Verifier, Answer]                                                      │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

##### Step 1: 确定每个Agent被激活的Steps (构建Y_k)

```
收集所有Steps：
┌────────────────────────────────────────────────────────────────────────┐
│ Total Steps in Group = 4 + 6 + 4 + 2 = 16 steps                         │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Y_Verifier = {                                                        │
│    (τ₁, step1, R=1),   ← Trajectory 1, Step 1, 奖励1                   │
│    (τ₁, step3, R=1),   ← Trajectory 1, Step 3, 奖励1                   │
│    (τ₂, step1, R=0),   ← Trajectory 2, Step 1, 奖励0                   │
│    (τ₂, step3, R=0),   ← Trajectory 2, Step 3, 奖励0                   │
│    (τ₂, step5, R=0),   ← Trajectory 2, Step 5, 奖励0                   │
│    (τ₃, step1, R=1),   ← Trajectory 3, Step 1, 奖励1                   │
│    (τ₃, step3, R=1),   ← Trajectory 3, Step 3, 奖励1                   │
│    (τ₄, step1, R=0)    ← Trajectory 4, Step 1, 奖励0                   │
│  }  → |Y_Verifier| = 8 steps                                           │
│                                                                        │
│  Y_Searcher = {                                                        │
│    (τ₁, step2, R=1),                                                   │
│    (τ₂, step2, R=0),                                                   │
│    (τ₂, step4, R=0),                                                   │
│    (τ₃, step2, R=1)                                                    │
│  }  → |Y_Searcher| = 4 steps                                           │
│                                                                        │
│  Y_Answer = {                                                          │
│    (τ₁, step4, R=1),                                                   │
│    (τ₂, step6, R=0),                                                   │
│    (τ₃, step4, R=1),                                                   │
│    (τ₄, step2, R=0)                                                    │
│  }  → |Y_Answer| = 4 steps                                             │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

**关键观察**:
- Verifier被调用了8次，Searcher和Answer各被调用4次
- 不同Agent的激活频率差异很大（这就是论文说的"agents are often invoked at different frequencies"）

##### Step 2: 计算每个Agent自己的统计量

```python
# Verifier的统计量 (基于Y_Verifier中的8个样本)
mu_Verifier = mean([1, 1, 0, 0, 0, 1, 1, 0]) = 0.5
sigma_Verifier = std([1, 1, 0, 0, 0, 1, 1, 0]) ≈ 0.53

# Searcher的统计量 (基于Y_Searcher中的4个样本)
mu_Searcher = mean([1, 0, 0, 1]) = 0.5
sigma_Searcher = std([1, 0, 0, 1]) ≈ 0.58

# Answer的统计量 (基于Y_Answer中的4个样本)
mu_Answer = mean([1, 0, 1, 0]) = 0.5
sigma_Answer = std([1, 0, 1, 0]) ≈ 0.58

# 注意：虽然这个例子中均值恰好相同，但实际训练中它们通常不同！
```

##### Step 3: 计算Agent-wise Advantage

```python
# Dr.MAS: 每个Agent用自己的统计量
A_Verifier(τ₁,step1) = (R=1 - 0.5) / 0.53 ≈ +0.94
A_Searcher(τ₁,step2) = (R=1 - 0.5) / 0.58 ≈ +0.86
A_Verifier(τ₁,step3) = (R=1 - 0.5) / 0.53 ≈ +0.94
A_Answer(τ₁,step4) = (R=1 - 0.5) / 0.58 ≈ +0.86

# 对比：GRPO使用全局统计量
mu_global = mean([1,0,1,0]) = 0.5  # 所有trajectory的reward
sigma_global = std([1,0,1,0]) ≈ 0.58

A_GRPO(所有step) = (R - 0.5) / 0.58  # 所有Agent用同一个baseline！
```

##### 为什么这很重要？

考虑一个更现实的场景：

```
场景：Verifier通常正确(高奖励)，Searcher经常检索失败(低奖励)

真实情况：
- Y_Verifier: 奖励 mostly 1s → mu_Verifier = 0.9, sigma_Verifier = 0.3
- Y_Searcher: 奖励 mixed → mu_Searcher = 0.4, sigma_Searcher = 0.5

GRPO问题：
- 全局 mu = 0.65, sigma = 0.48
- Verifier的advantage = (0.9 - 0.65) / 0.48 = 0.52 (被低估！)
- Searcher的advantage = (0.4 - 0.65) / 0.48 = -0.52 (被高估！)

Dr.MAS修正：
- Verifier: (0.9 - 0.9) / 0.3 = 0 (合理的baseline)
- Searcher: (0.4 - 0.4) / 0.5 = 0 (合理的baseline)
```

**核心洞见**: 当Agent的角色不同导致奖励分布不同时，全局baseline会系统性偏离某些Agent的真实表现，导致梯度不稳定。

---

### 9.3 完整的Multi-Agent交互流程

#### Math任务：Solver ↔ Verifier Loop

```
输入问题: "Find the expected number of regions..."

    ┌─────────────┐
    │   Start     │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Solver    │ ← Policy π_Solver (生成解法)
    │  "我认为答案是379"  │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Verifier   │ ← Policy π_Verifier (评估解法)
    │ "推理有缺陷，拒绝"  │
    │  <verify>reject</verify>  │
    └──────┬──────┘
           │
           └────────────────┐
                            │ (未通过，重新迭代)
           ┌────────────────┘
           ↓
    ┌─────────────┐
    │   Solver    │ ← 再次调用，基于反馈改进
    │ "重新计算后答案是204" │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Verifier   │
    │ "推理正确"  │
    │  <verify>approve</verify> │
    └──────┬──────┘
           ↓ (通过)
    ┌─────────────┐
    │   End       │ ← 输出最终答案 "204"
    │  奖励 R=1   │
    └─────────────┘

关键特性：
- Solver和Verifier被调用次数不确定（1-3轮迭代）
- 整个trajectory只有一个奖励 R=1
- Solver的Y_k包含2个steps，Verifier的Y_k包含2个steps

---

### 9.4 Verifier和Searcher的奖励机制详解

#### 核心问题：中间Agent的Reward从哪来？

在Dr.MAS中，**所有Agent共享同一个Trajectory-level Reward**，没有per-agent或per-step的独立reward。这是RLVR (Reinforcement Learning from Verifiable Rewards) 的典型特征。

#### Math任务的Reward计算

```python
# 伪代码：Math任务的Reward函数
def compute_math_reward(trajectory, ground_truth_answer):
    """
    输入：完整的Solver-Verifier交互轨迹
    输出：Binary reward (1/0)
    """
    # 1. 从轨迹中提取最终答案
    # 轨迹格式：[(step1, Solver, "解法..."), (step2, Verifier, "<verify>approve</verify>"), ...]
    final_answer = extract_final_answer(trajectory)

    # 2. 与标准答案比对（精确匹配或语义等价）
    if verify_answer(final_answer, ground_truth_answer):
        R = 1  # 成功
    else:
        R = 0  # 失败

    return R

# 实际示例
trajectory_1 = [
    (step=1, agent="Solver", action="379"),
    (step=2, agent="Verifier", action="<verify>reject</verify>"),
    (step=3, agent="Solver", action="204"),
    (step=4, agent="Verifier", action="<verify>approve</verify>")
]
ground_truth = "204"
R = compute_math_reward(trajectory_1, ground_truth)  # R = 1 ✓

# 所有steps共享同一个R=1：
# - Solver step1: R=1 (尽管输出"379"是错误的，但因为最终答案正确，它得到正奖励)
# - Verifier step2: R=1 (拒绝错误答案的行为被奖励)
# - Solver step3: R=1 (生成正确答案被奖励)
# - Verifier step4: R=1 (批准正确答案被奖励)
```

#### Search任务的Reward计算

```python
# 伪代码：Search任务的Reward函数
def compute_search_reward(trajectory, ground_truth_answer):
    """
    输入：完整的Verifier-Searcher-Answer交互轨迹
    输出：Binary reward (1/0)
    """
    # 1. 从轨迹中提取最终答案
    final_answer = extract_answer_from_last_step(trajectory)

    # 2. 与参考答案比对
    if fuzzy_match(final_answer, ground_truth_answer):
        R = 1
    else:
        R = 0

    return R

# 实际示例
trajectory_search = [
    (step=1, agent="Verifier", action="<verify>no</verify>"),      # 判断需要搜索
    (step=2, agent="Searcher", action="query: Mark Dismore birthplace"),
    (step=3, agent="Verifier", action="<verify>no</verify>"),      # 还需要更多信息
    (step=4, agent="Searcher", action="query: Greenfield Indiana county"),
    (step=5, agent="Verifier", action="<verify>yes</verify>"),     # 信息充足
    (step=6, agent="Answer", action="Hancock County")
]
ground_truth = "Hancock County"
R = compute_search_reward(trajectory_search, ground_truth)  # R = 1 ✓

# 所有steps共享R=1：
# - Verifier step1: R=1 (正确判断需要搜索)
# - Searcher step2: R=1 (检索相关信息)
# - Verifier step3: R=1 (正确判断信息仍不足)
# - Searcher step4: R=1 (检索到关键信息)
# - Verifier step5: R=1 (正确判断可以回答)
# - Answer step6: R=1 (生成正确答案)
```

#### 为什么这种设计有效？

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Credit Assignment问题                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  问题：所有steps共享同一个R，如何区分哪个Agent做得好？                        │
│                                                                             │
│  答案：通过Group Relative的比较机制（GRPO/Dr.MAS的核心）                     │
│                                                                             │
│  示例：Group Size = 4，同一问题的4次尝试                                     │
│                                                                             │
│  Trajectory 1: Verifier错 → Searcher错 → Verifier错 → Answer错   → R=0    │
│  Trajectory 2: Verifier对 → Searcher对 → Verifier对 → Answer对   → R=1    │
│  Trajectory 3: Verifier对 → Searcher错 → Verifier对 → Answer错   → R=0    │
│  Trajectory 4: Verifier对 → Searcher对 → Verifier对 → Answer对   → R=1    │
│                                                                             │
│  对于"Searcher被激活的steps"（假设是step2）：                                │
│  Y_Searcher = [(τ1,s2,R=0), (τ2,s2,R=1), (τ3,s2,R=0), (τ4,s2,R=1)]        │
│                                                                             │
│  Dr.MAS计算：                                                               │
│  mu_Searcher = mean([0,1,0,1]) = 0.5                                       │
│  sigma_Searcher = std([0,1,0,1]) ≈ 0.58                                    │
│                                                                             │
│  Advantage计算：                                                            │
│  A(τ1,s2) = (0 - 0.5) / 0.58 = -0.86  ← 这次检索表现差于平均                │
│  A(τ2,s2) = (1 - 0.5) / 0.58 = +0.86  ← 这次检索表现好于平均                │
│                                                                             │
│  结果：即使共享R，通过group内比较，Searcher还是能学到"哪些检索策略更好"     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 直观理解：Verifier的Reward机制

```
场景：Verifier作为"路由器"决定下一步

情况1：Verifier错误地说"信息充足"，导致Answer生成错误答案
  → Trajectory失败，R=0
  → 所有Verifier的steps得到负advantage（相对于其他成功的trajectory）
  → Verifier学到：这种状态下应该说"需要搜索"

情况2：Verifier正确地说"需要搜索"，Searcher检索到关键信息
  → Trajectory成功，R=1
  → Verifier的steps得到正advantage
  → Verifier学到：这种状态下应该调用Searcher

关键：Verifier不是通过"判断正确/错误"的即时奖励来学习，
     而是通过"最终答案是否正确"的长期结果来学习
```

#### 代码层面的实现

```python
# Dr.MAS核心的Reward和Advantage计算逻辑

class DrMASTrainer:
    def compute_rewards(self, trajectories):
        """
        计算trajectory-level reward（所有Agent共享）
        """
        rewards = []
        for traj in trajectories:
            # 调用环境验证最终答案
            R = self.environment.verify(traj.final_answer)
            rewards.append(R)
        return rewards  # [1, 0, 1, 0, ...]

    def compute_agent_wise_advantages(self, trajectories, rewards):
        """
        Dr.MAS核心：为每个Agent计算自己的advantage
        """
        advantages = {}

        for agent_id in self.agents:
            # 1. 收集该Agent被激活的所有steps
            Y_k = []
            for i, traj in enumerate(trajectories):
                for step in traj.steps:
                    if step.agent_id == agent_id:
                        Y_k.append((i, step, rewards[i]))  # (traj_idx, step, R)

            # 2. 计算该Agent自己的统计量
            R_k = [r for _, _, r in Y_k]
            mu_k = np.mean(R_k)
            sigma_k = np.std(R_k)

            # 3. 计算advantage
            for traj_idx, step, R in Y_k:
                A = (R - mu_k) / (sigma_k + 1e-8)
                advantages[(traj_idx, step.step_id)] = A

        return advantages

    def update_policy(self, trajectories):
        # 1. 计算trajectory rewards（所有Agent共享）
        rewards = self.compute_rewards(trajectories)

        # 2. Dr.MAS: Agent-wise advantage normalization
        advantages = self.compute_agent_wise_advantages(trajectories, rewards)

        # 3. 各Agent用自己的advantage更新策略
        for agent_id in self.agents:
            agent_advantages = filter_by_agent(advantages, agent_id)
            self.agents[agent_id].update(agent_advantages)
```

---

#### 总结

| 问题 | 答案 |
|------|------|
| Verifier/Seracher有自己的reward吗？ | **没有**，所有Agent共享trajectory-level R |
| R是怎么计算的？ | 基于**最终答案正确性** (1/0) |
| 中间决策怎么学？ | 通过Group内比较（GRPO机制）+ Dr.MAS的Agent-wise归一化 |
| Verifier怎么知道判断对错？ | 通过最终答案是否正确来反推（信用分配） |

这种设计的优点是**简单且可扩展**（不需要设计复杂的per-step reward），缺点是**信用分配模糊**（中间步骤的贡献难以精确量化）。Dr.MAS通过Agent-wise Normalization缓解了这个问题。

#### Search任务：Verifier → Searcher → Answer 层级

```
输入问题: "In which county is Mark Dismore's birthplace?"

    ┌─────────────┐
    │   Start     │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Verifier   │ ← Policy π_Verifier
    │ "信息不足，需要搜索" │
    │  <verify>no</verify>   │
    └──────┬──────┘
           ↓ (路由到Searcher)
    ┌─────────────┐
    │  Searcher   │ ← Policy π_Searcher
    │ "搜索: Birthplace of Mark Dismore" │
    │ 调用Search API，获取结果           │
    └──────┬──────┘
           ↓ (返回信息给Verifier)
    ┌─────────────┐
    │  Verifier   │
    │ "Greenfield, Indiana，但不知道county" │
    │  <verify>no</verify>   │
    └──────┬──────┘
           ↓ (再次路由到Searcher)
    ┌─────────────┐
    │  Searcher   │
    │ "搜索: County where Greenfield Indiana" │
    └──────┬──────┘
           ↓ (返回信息)
    ┌─────────────┐
    │  Verifier   │
    │ "信息充足，Hancock County" │
    │  <verify>yes</verify>   │
    └──────┬──────┘
           ↓ (路由到Answer)
    ┌─────────────┐
    │   Answer    │ ← Policy π_Answer
    │ "Hancock County" │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   End       │
    │  奖励 R=1   │
    └─────────────┘

关键特性：
- Verifier作为"路由器"决定下一步调用哪个Agent
- Searcher可能被调用0次、1次或多次
- 整个trajectory只有一个奖励
- 各Agent的Y_k大小：Verifier=3, Searcher=2, Answer=1
```

#### 为什么这种设计有挑战？

1. **动态调用次数**: 不同trajectory中同一Agent被调用次数不同
2. **异构奖励分布**: Verifier经常成功(高奖励)，Searcher容易失败(低奖励)
3. **信用分配**: 最终奖励应该归因于哪个Agent的哪个决策？

Dr.MAS通过Agent-wise Normalization解决了第2个问题，使得即使奖励分布不同，每个Agent的梯度更新也是稳定的。

---

### 9.5 数据集来源与开源情况

#### 开源状态总结

| 项目 | 开源状态 | 获取方式 |
|------|---------|---------|
| **Dr.MAS代码** | ✅ 开源 | https://github.com/langfengQ/DrMAS |
| **训练数据集** | ⚠️ 需自行准备 | 使用开源数据集+预处理脚本 |
| **评估Benchmarks** | ✅ 开源 | 公开学术benchmark |

**注意**: 论文作者没有直接发布训练数据文件，但提供了**数据预处理脚本**，用户需要用脚本自行处理开源数据集。

---

#### Math任务数据集

**训练数据来源**: DAPO-Math (Yu et al., 2025) — 这是开源的

```
DAPO-Math数据集特点:
├── 类型: 多样化数学问题
├── 格式: 问题 + 可验证的答案/解法
├── 奖励信号: 答案正确性 (pass/fail)
├── 规模: 大规模合成+真实数学问题混合
└── 开源状态: ✅ DAPO系统是开源的 (https://github.com/BytedTsinghuaUIAT/DAPO)
```

**数据获取路径**:
```bash
# 通过Dr.MAS代码库中的预处理脚本准备
python examples/data_preprocess/drmas_math.py

# DAPO-Math数据可通过DAPO开源项目获取
# https://github.com/BytedTsinghuaUIAT/DAPO
```

**评估Benchmarks** (均为开源学术benchmark):
| Benchmark | 类型 | 开源链接 |
|-----------|------|---------|
| AIME'24/25 | 竞赛数学 | 公开竞赛题目 |
| AMC'23 | 竞赛数学 | 公开竞赛题目 |
| MATH500 | 学术数学 | https://github.com/hendrycks/math |
| Minerva | 科学数学 | Google公开 |
| OlympiadBench | 奥赛级别 | https://github.com/OpenBMB/OlympiadBench |

#### Search任务数据集

**训练数据**:
```
训练集: NQ + HotpotQA 的混合
├── NQ (Natural Questions)
│   └── 来源: Google搜索日志中的真实用户问题
│   └── 类型: 单跳问答
│   └── 规模: ~300k问题
│   └── 开源链接: https://ai.google.com/research/NaturalQuestions
│
└── HotpotQA
    └── 来源: 维基百科的多跳推理问题
    └── 类型: 需要2跳推理的复杂问题
    └── 规模: ~100k问题
    └── 开源链接: https://hotpotqa.github.io/
```

**数据获取路径**:
```bash
# 1. 下载检索索引 (通过Dr.MAS提供的脚本)
python examples/search/searchr1_download.py --local_dir $local_dir
# 会下载: wiki-18.jsonl.gz (维基百科数据) 和 e5_Flat.index (E5检索索引)

# 2. 数据预处理
python examples/data_preprocess/drmas_search.py
```

**评估Benchmarks** (均为开源):
| Benchmark | 类型 | 开源链接 |
|-----------|------|---------|
| NQ | 单跳QA | https://ai.google.com/research/NaturalQuestions |
| TriviaQA | 单跳QA | http://nlp.cs.washington.edu/triviaqa/ |
| PopQA | 单跳QA | https://github.com/AlexTMallen/popqa |
| HotpotQA | 多跳QA | https://hotpotqa.github.io/ |
| 2WikiMultiHop | 多跳QA | https://github.com/AndrewZhe/2wikimultihopqa |
| MuSiQue | 多跳QA | https://github.com/StonyBrookNLP/musique |
| Bamboogle | 多跳QA | 随论文发布的数据集 |

**检索器**: E5 (Wang et al., 2022)
- 开源链接: https://github.com/microsoft/unilm/tree/master/e5

---

### 9.6 如何复现论文数据环境

#### 快速开始 (基于Dr.MAS官方代码)

```bash
# 1. 克隆代码库
git clone https://github.com/langfengQ/DrMAS.git
cd DrMAS

# 2. Math任务数据准备
python examples/data_preprocess/drmas_math.py

# 3. Search任务数据准备
# 3.1 下载维基百科索引和E5检索器
python examples/search/searchr1_download.py --local_dir ./data/search
# 3.2 预处理数据集
python examples/data_preprocess/drmas_search.py
```

#### 数据集开源情况总览

| 数据集 | 论文中使用 | 开源状态 | 获取方式 |
|--------|-----------|---------|---------|
| DAPO-Math | 训练 | ✅ 开源 | DAPO项目 + Dr.MAS预处理脚本 |
| NQ | 训练+评估 | ✅ 开源 | Google AI官方 |
| HotpotQA | 训练+评估 | ✅ 开源 | 项目官网 |
| MATH500 | 评估 | ✅ 开源 | Hendrycks GitHub |
| TriviaQA | 评估 | ✅ 开源 | 项目官网 |
| PopQA | 评估 | ✅ 开源 | GitHub |
| 2WikiMultiHop | 评估 | ✅ 开源 | GitHub |
| MuSiQue | 评估 | ✅ 开源 | GitHub |
| AIME/AMC | 评估 | ✅ 公开 | 竞赛官方发布 |
| OlympiadBench | 评估 | ✅ 开源 | OpenBMB GitHub |

#### 关键依赖

```bash
# 检索环境需要
faiss-gpu  # Facebook AI Similarity Search

# 维基百科数据源
wiki-18.jsonl.gz  # 2018年维基百科dump

# 预训练检索器
e5_Flat.index  # E5模型生成的向量索引
```

**实验设置**:
- Rollout group size: 5
- Max turn: 4 (最多4轮搜索-验证迭代)
- 评估指标: avg@16 (16次采样的平均正确率), pass@16 (16次中至少1次正确的比例)

### 9.7 训练配置细节

```python
# Math任务配置
{
    "model": "Qwen3-4B/8B",
    "rollout_group_size": 8,
    "training_data": "DAPO-Math corpus",
    "orchestration": "Solver ↔ Verifier 循环",
    "reward": "答案正确性 (0/1)"
}

# Search任务配置
{
    "model": "Qwen2.5-3B/7B",
    "rollout_group_size": 5,
    "max_turn": 4,
    "training_data": "NQ + HotpotQA 混合",
    "retriever": "E5",
    "orchestration": "Verifier → Searcher → Answer 层级",
    "reward": "答案正确性 (0/1)"
}
```

---

## 10. 关键公式速查

| 公式 | 名称 | 说明 |
|------|------|------|
| A^i = (R^i - µ)/σ | GRPO全局优势 | µ, σ为全局统计量 |
| A^i,k = (R^i - µ_k)/σ_k | Dr.MAS Agent-wise优势 | µ_k, σ_k为智能体k的统计量 |
| E[‖g̃_k‖²] = E[‖z‖²] × (σ²_k + (µ_k - µ)²)/σ² | 梯度二阶矩(GRPO) | 不稳定来源 |
| E[‖g̃_k‖²] = E[‖z‖²] + Δ_k | 梯度二阶矩(Dr.MAS) | 稳定 |

---

---

## 11. 深入讨论

### 11.1 问题1：Trajectory-level Loss回传，为什么两个Model的Group Relative Reward可以不一样？

这是一个关于Dr.MAS核心机制的关键问题。让我用一个具体例子来解释：

#### 直观示例

假设我们有一个**Group Size = 4**（同一问题的4次尝试），使用Solver+Verifier两个Agent：

```
Group: 4条Trajectory解决同一道数学题

Trajectory 1 (成功, R=1):
  Solver(step1): "尝试解法A"
  Verifier(step2): "<verify>reject</verify>"  ← 拒绝
  Solver(step3): "尝试解法B"
  Verifier(step4): "<verify>approve</verify>" ← 批准

Trajectory 2 (失败, R=0):
  Solver(step1): "尝试解法C"
  Verifier(step2): "<verify>reject</verify>"
  Solver(step3): "尝试解法D"
  Verifier(step4): "<verify>reject</verify>"  ← 达到最大轮数，失败

Trajectory 3 (成功, R=1):
  Solver(step1): "尝试解法E"
  Verifier(step2): "<verify>approve</verify>" ← 一次就批准

Trajectory 4 (失败, R=0):
  Solver(step1): "尝试解法F"
  Verifier(step2): "<verify>reject</verify>"
  Solver(step3): "尝试解法G"
  Verifier(step4): "<verify>approve</verify>" ← 批准但答案错！(Verifier犯错)
```

#### 构建Y_k集合（这是关键！）

```python
# Solver被激活的steps (Y_Solver)
Y_Solver = [
    (τ1, step1, R=1),  # "尝试解法A"
    (τ1, step3, R=1),  # "尝试解法B"
    (τ2, step1, R=0),  # "尝试解法C"
    (τ2, step3, R=0),  # "尝试解法D"
    (τ3, step1, R=1),  # "尝试解法E"
    (τ4, step1, R=0),  # "尝试解法F"
    (τ4, step3, R=0),  # "尝试解法G"
]

# Verifier被激活的steps (Y_Verifier)
Y_Verifier = [
    (τ1, step2, R=1),  # "拒绝A"
    (τ1, step4, R=1),  # "批准B"
    (τ2, step2, R=0),  # "拒绝C"
    (τ2, step4, R=0),  # "拒绝D"
    (τ3, step2, R=1),  # "批准E"
    (τ4, step2, R=0),  # "拒绝F"
    (τ4, step4, R=0),  # "批准G但答案错" (Verifier的错误判断)
]
```

#### 计算各自的统计量

```python
# Solver的统计量
R_Solver = [1, 1, 0, 0, 1, 0, 0]
mu_Solver = mean([1, 1, 0, 0, 1, 0, 0]) = 3/7 ≈ 0.43
sigma_Solver = std([1, 1, 0, 0, 1, 0, 0]) ≈ 0.53

# Verifier的统计量
R_Verifier = [1, 1, 0, 0, 1, 0, 0]  # 注意：τ4的批准是错误判断，但reward还是0
mu_Verifier = mean([1, 1, 0, 0, 1, 0, 0]) = 3/7 ≈ 0.43
sigma_Verifier = std([1, 1, 0, 0, 1, 0, 0]) ≈ 0.53

# 这个例子中碰巧相同，但考虑更现实的情况...
```

#### 更现实的场景（分布不同）

```
现实情况：Verifier通常比Solver更容易"正确"

假设10条Trajectory：
- Solver的尝试：混合成功和失败 → mu_Solver = 0.5
- Verifier的判断：大部分正确（即使最终答案错，Verifier的判断过程往往是对的）→ mu_Verifier = 0.7

GRPO的问题：
- 全局 mu = 0.6
- Solver的advantage：用0.6做baseline，对其偏高
- Verifier的advantage：用0.6做baseline，对其偏低

Dr.MAS的解决：
- Solver用自己的mu=0.5做baseline
- Verifier用自己的mu=0.7做baseline
- 各自都有合适的参考点
```

#### 核心洞见

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      为什么可以"不一样"？                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Group是Trajectory级别的                                                  │
│     - 4个trajectory组成一个group                                             │
│     - 每个trajectory只有一个R                                                │
│                                                                             │
│  2. 但Y_k是Agent级别的                                                        │
│     - Y_Solver包含Solver被激活的所有steps                                     │
│     - Y_Verifier包含Verifier被激活的所有steps                                 │
│     - 这两个集合的reward分布可以不同！                                         │
│                                                                             │
│  3. Group Relative是相对于"同Agent的其他样本"                                 │
│     - Solver比较的是：自己在不同trajectory中的表现                           │
│     - Verifier比较的是：自己在不同trajectory中的表现                         │
│     - 而不是Solver和Verifier之间互相比较！                                    │
│                                                                             │
│  类比：                                                                      │
│  - 像一个团队项目，最终成绩一样（trajectory R）                               │
│  - 但每个人有自己的"横向比较"（同role的其他成员表现）                         │
│  - 程序员和测试员的评价标准不一样，不能混在一起评！                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 数学解释

```python
# GRPO（错误的做法）
A_i = (R_i - mu_global) / sigma_global
# 所有Agent用同一个baseline

# Dr.MAS（正确的做法）
A_i,k = (R_i - mu_k) / sigma_k
# 每个Agent k有自己的mu_k和sigma_k
# mu_k = mean([R for steps in Y_k])
# 即使R_i相同，不同Agent的mu_k不同，advantage也不同
```

---

### 11.2 问题2：Dr.MAS如何处理树状(Tree) Trajectory？

#### 当前Dr.MAS的假设：链式(Chain)结构

论文中所有的trajectory都是**线性序列**：
```
Chain: s1 → s2 → s3 → ... → sT
       ↓    ↓    ↓          ↓
      a1   a2   a3         aT
       ↓    ↓    ↓          ↓
     Agent Agent Agent     Agent
```

#### 树状结构的挑战

```
Tree结构的场景：

                    Verifier
                   /        \
                 Yes         No
                /              \
            Answer          Searcher
                            /      \
                         Doc1      Doc2
                          |         |
                       Verifier   Verifier
                          |         |
                        Answer    Answer

或者更复杂的分支：

                    Planner
          /          |          \
    Search_A      Search_B     Direct_Answer
        |            |            |
    Verifier     Verifier      Verifier
      ...          ...           ...
```

树状结构带来的问题：
1. **Multiple paths**: 同一问题可以有多个解决路径
2. **Branching factor**: 一个Agent可能产生多个子分支
3. **Reward assignment**: 某个分支的R如何归因到前面的决策？

#### Dr.MAS对树状结构的适用性分析

**情况1：Sequential Tree（论文支持）**
```
实际上是链式，只是逻辑上是树：

Verifier决策 → 选择路径A或B → 继续

这种可以转化为链式：
Verifier("选择A") → A的执行 → ...
Verifier("选择B") → B的执行 → ...

Dr.MAS支持：只需要把"选择"作为一个action
```

**情况2：Parallel Tree（论文未明确支持）**
```
Verifier同时调用多个Searcher并行检索：

Verifier(step1)
  ├─ Searcher_A(step2a) ─┐
  ├─ Searcher_B(step2b) ─┼─ Verifier(step3) ─ Answer(step4)
  └─ Searcher_C(step2c) ─┘

挑战：
- Y_Searcher需要包含step2a, 2b, 2c
- 但它们的R相同（最终答案决定）
- 这与链式结构本质上没有区别
```

**情况3：General DAG/Tree（需要扩展）**
```
更一般的结构，有汇聚点：

     A1         A2
      \         /
       \       /
        \     /
         \   /
       Verifier
           |
          B1

挑战：
- A1和A2并行执行
- 它们的rewards如何分配？
- Verifier的决策基于A1和A2的综合输出
```

#### 可能的扩展方案

**方案1：路径分解（Path Decomposition）**
```python
# 将tree分解为多条chain，每条单独训练
def decompose_tree_to_chains(tree_trajectory):
    """把树分解为从根到叶的所有路径"""
    paths = []
    for leaf in tree.leaves():
        path = get_path_from_root(leaf)
        paths.append(path)
    return paths

# 每条路径作为一个独立的trajectory训练
# 共享相同的环境reward
```

**方案2：Node-level Credit Assignment**
```python
# 更细粒度的信用分配
def compute_node_advantage(node, tree):
    """
    考虑子树的成功率
    """
    if node.is_leaf():
        return node.reward

    # 父节点的advantage基于子节点的加权平均
    child_advantages = [compute_node_advantage(c, tree) for c in node.children]
    weights = [c.visit_count for c in node.children]

    return weighted_mean(child_advantages, weights)
```

**方案3：MCTS-style Value Estimation**
```python
# 借鉴蒙特卡洛树搜索
class TreeGRPO:
    def __init__(self):
        self.visit_counts = {}
        self.value_estimates = {}

    def compute_advantage(self, node):
        # 基于多个rollout的统计
        # 类似MCTS中的Q值
        pass
```

#### 论文的局限性

```
论文明确承认的局限（Section 6 Conclusions and Limitations）：

"Dr. MAS does not resolve all sources of instability in multi-agent LLM RL
 (e.g., credit assignment across agents and turns)."

"Furthermore, although our framework supports flexible multi-agent orchestration
 and resource pooling, we have not evaluated settings with a much larger number
 of agents."

解读：
- 信用分配问题（credit assignment）未完全解决
- 未在大规模/复杂拓扑下验证
- 树状/图状结构属于开放问题
```

#### 实际建议

```
当前Dr.MAS支持的编排模式：
✅ 链式：A → B → C → D
✅ 循环：A → B → (迭代) → C
✅ 条件分支（序列化）：A → (if x then B else C) → D

当前Dr.MAS不支持或需要扩展：
❌ 真正的并行：A同时调用B和C
❌ 复杂的DAG结构
❌ 需要子树聚合的场景

Workaround（变通方案）：
- 将并行转化为序列：A → B → C → D（虽然效率低）
- 或者：A → Coordinator → (B then C) → D
```

---

### 11.3 总结

| 问题 | 核心答案 |
|------|---------|
| **为什么两个Model的Group Relative可以不一样？** | 因为Y_k是Agent-specific的，每个Agent比较的是自己在不同trajectory中的表现，而不是和其他Agent比较 |
| **Dr.MAS支持树状Trajectory吗？** | 论文只验证了链式结构。树状/并行结构需要扩展，核心挑战是信用分配和reward归因 |

*这两个问题都指向了Multi-Agent RL的核心开放问题：在复杂的Agent交互拓扑中，如何有效地分配信用（Credit Assignment）。Dr.MAS解决了其中的一个重要子问题（异构Agent的梯度稳定性），但更一般的结构仍是研究前沿。*

---

---

## 12. 深入讨论：上下文隔离与Loss计算机制

### 12.1 Verifier和Searcher的上下文如何隔离？

这是一个关于实际代码实现的关键问题。从论文附录C的Prompt Templates可以看出关键设计：

#### 核心设计：文本形式的上下文传递

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Dr.MAS中的Context处理方式                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  每个Agent收到的是Composite Prompt（组合式提示）：                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Composite Prompt = env_prompt + team_context + role_instruction   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  1. env_prompt: 任务描述 + 交互历史（memory_context）                       │
│  2. team_context: 队友的输出（以文本形式）                                   │
│  3. role_instruction: 当前Agent的角色定义                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 具体示例

**Search任务的交互流程：**

```python
# Step 1: Verifier被激活
verifier_input = """
# Task Introduction
You are a member of an expert multi-agent team tasked with answering the given question step-by-step.
The question is: "What government position was held by Shirley Temple?"

Prior to this step, your team has already taken 0 step(s).
Below is the interaction history where <search></search> wrapped the past search queries
and <information></information> wrapped the corresponding retrieved information.
History: {memory_context}  # 初始为空

# Your Role
You are a "Verifier Agent" acting as a router...
"""

verifier_output = "<verify>no</verify>"  # 判断需要搜索

# ==========================================
# Step 2: Searcher被激活
searcher_input = """
# Task Introduction
You are a member of an expert multi-agent team...
Prior to this step, your team has already taken 1 step(s).
Below is the interaction history...
History:
  Step 1: Verifier Agent output: <verify>no</verify>

# Your Teammates' Outputs at Step 2
Verifier Agent at step 1: <verify>no</verify>

# Your Role
You are a "Search Agent"...
"""

searcher_output = "<think>...</think>\n<search>Shirley Temple government position</search>"

# ==========================================
# Step 3: Verifier再次被激活
verifier_input = """
# Task Introduction
You are a member of an expert multi-agent team...
Prior to this step, your team has already taken 2 step(s).
Below is the interaction history...
History:
  Step 1: Verifier Agent output: <verify>no</verify>
  Step 2: Search Agent output: <search>Shirley Temple government position</search>
  Step 2: Retrieved information: <information>Shirley Temple Black served as the
           United States Ambassador to Czechoslovakia (1989-1992) and Chief of
           Protocol of the United States (1976-1977).</information>

# Your Role
You are a "Verifier Agent" acting as a router...
"""

verifier_output = "<verify>yes</verify>"  # 信息充足
```

#### 关键洞察：没有共享Hidden States

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    错误的理解 vs 正确的理解                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ❌ 错误的理解：                                                             │
│     认为Verifier和Searcher共享一个大的Transformer模型                        │
│     需要在同一个forward pass中处理                                           │
│     需要考虑attention mask来隔离不同部分                                     │
│                                                                             │
│  ✅ 正确的理解：                                                             │
│     Verifier和Searcher是两个独立的模型（可能共享参数，但推理独立）           │
│     每个Agent在自己的forward pass中处理输入                                  │
│     上下文通过文本形式的"history"和"team_context"传递                        │
│     不需要mask，因为本来就是独立的调用                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 12.2 在算Loss时如何处理？

#### Rollout阶段的独立推理

```python
# 伪代码：Rollout阶段的执行流程

def rollout_trajectory(task, agents, orchestra):
    """
    生成一条完整的trajectory
    """
    trajectory = []
    state = {"task": task, "history": []}

    for step in range(max_steps):
        # 1. Orchestra决定哪个Agent应该执行
        agent_id = orchestra.select_agent(state)
        agent = agents[agent_id]

        # 2. 构建当前Agent的输入Prompt
        prompt = build_prompt(
            env_prompt=state["task"],
            memory_context=format_history(state["history"]),
            team_context=format_last_output(state["history"]),
            role=agent.role
        )

        # 3. 当前Agent独立执行推理
        # 注意：这里是一个独立的model.forward()调用！
        with torch.no_grad():
            output = agent.model.generate(prompt)

        # 4. 记录这个step
        trajectory.append({
            "step": step,
            "agent_id": agent_id,
            "prompt": prompt,        # 完整的输入
            "output": output,        # 生成的输出
            "state": state.copy()
        })

        # 5. 更新state，供下一个Agent使用
        state["history"].append({
            "agent_id": agent_id,
            "output": output
        })

        # 6. 检查是否结束
        if is_terminal(state):
            break

    # 7. 计算trajectory-level reward
    reward = compute_reward(trajectory, task.ground_truth)

    return trajectory, reward
```

#### Training阶段的Loss计算

```python
# 伪代码：Training阶段的Loss计算

def compute_drmas_loss(batch, agents):
    """
    batch包含多条trajectory，每条trajectory有多个steps
    """

    # Step 1: 按Agent分组收集数据
    agent_batches = {agent_id: [] for agent_id in agents}

    for trajectory in batch:
        R = trajectory.reward
        for step in trajectory.steps:
            agent_id = step.agent_id
            agent_batches[agent_id].append({
                "prompt": step.prompt,
                "output": step.output,
                "R": R  # 同一个trajectory的所有steps共享同一个R
            })

    # Step 2: 对每个Agent计算自己的统计量
    agent_stats = {}
    for agent_id, data in agent_batches.items():
        rewards = [d["R"] for d in data]
        agent_stats[agent_id] = {
            "mu": np.mean(rewards),
            "sigma": np.std(rewards) + 1e-8
        }

    # Step 3: 计算每个Agent的Loss
    total_loss = 0
    for agent_id, data in agent_batches.items():
        agent = agents[agent_id]
        mu_k = agent_stats[agent_id]["mu"]
        sigma_k = agent_stats[agent_id]["sigma"]

        agent_loss = 0
        for item in data:
            # Dr.MAS的Agent-wise advantage
            A = (item["R"] - mu_k) / sigma_k

            # PPO-clip loss（标准的单模型loss）
            # 注意：每个Agent只看到自己的prompt和output
            old_logprob = compute_logprob(agent.old_model, item["prompt"], item["output"])
            new_logprob = compute_logprob(agent.model, item["prompt"], item["output"])
            ratio = torch.exp(new_logprob - old_logprob)

            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
            loss = -torch.min(ratio * A, clipped_ratio * A)

            agent_loss += loss

        total_loss += agent_loss / len(data)

    return total_loss
```

---

### 12.3 图解：完整的上下文隔离机制

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        实际代码执行流程                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: Verifier执行                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Input:  "Task: Shirley Temple...\nHistory: []\nYour role: Verifier"  │ │
│  │       ↑                                                               │ │
│  │       │ 这是一个完整的Prompt文本                                       │ │
│  │       │ 不包含任何Searcher的信息                                       │ │
│  │  ┌────┴────┐                                                          │ │
│  │  │ Verifier│ 独立的模型调用                                            │ │
│  │  │  Model  │  model.generate(prompt)                                   │ │
│  │  └────┬────┘                                                          │ │
│  │  Output: "<verify>no</verify>"                                        │ │
│  └───────┬───────────────────────────────────────────────────────────────┘ │
│          │                                                                  │
│          │ 将输出加入History                                               │
│          ↓                                                                  │
│  Step 2: Searcher执行                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Input:  "Task: Shirley Temple...\nHistory: [Verifier: <no>]\n        │ │
│  │          Your role: Searcher"                                         │ │
│  │       ↑                                                               │ │
│  │       │ 注意：History中包含Verifier的输出文本                           │ │
│  │       │ 但Searcher看不到Verifier的hidden states！                       │ │
│  │  ┌────┴────┐                                                          │ │
│  │  │Searcher │ 独立的模型调用                                            │ │
│  │  │  Model  │  model.generate(prompt)                                   │ │
│  │  └────┬────┘                                                          │ │
│  │  Output: "<search>Shirley Temple...</search>"                         │ │
│  └───────┬───────────────────────────────────────────────────────────────┘ │
│          │                                                                  │
│          │ 将输出加入History                                               │
│          ↓                                                                  │
│  Step 3: Verifier再次执行                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  Input:  "Task: Shirley Temple...\nHistory: [Verifier: <no>,         │ │
│  │          Searcher: <search>..., Information: <...>]\n                 │ │
│  │          Your role: Verifier"                                         │ │
│  │       ↑                                                               │ │
│  │       │ 注意：这是一个全新的forward pass                               │ │
│  │       │ 与Step 1的Verifier调用完全独立！                               │ │
│  │       │ 没有共享KV cache或hidden states                                │ │
│  │  ┌────┴────┐                                                          │ │
│  │  │ Verifier│ 独立的模型调用                                            │ │
│  │  │  Model  │  model.generate(prompt)                                   │ │
│  │  └────┴────┘                                                          │ │
│  │  Output: "<verify>yes</verify>"                                       │ │
│  └───────────┴───────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 12.4 关键结论

| 问题 | 答案 |
|------|------|
| **Verifier和Searcher的上下文如何隔离？** | 通过文本形式的`memory_context`传递，每个Agent只看到自己Prompt中的历史记录 |
| **需要mask吗？** | **不需要**！因为每个Agent是独立的模型调用，不是共享一个forward pass |
| **Loss怎么算？** | 每个Agent独立计算自己的PPO loss，使用Agent-wise normalized advantage |
| **Hidden states共享吗？** | **不共享**！每次调用都是独立的`model.generate()` |

#### 类比理解

```
传统单模型多轮对话：
User → Assistant → User → Assistant
（在一个continued context中，共享KV cache）

Dr.MAS多Agent协作：
User → AgentA → User → AgentB → User → AgentA
（每个Agent是独立的API调用，上下文通过文本传递）
```

这种设计的**优点**是简单、灵活、易于调试；**缺点**是可能有信息损失（文本传递不如hidden states丰富）。

---

*整理日期: 2026-03-17*
*整理者: 王明达 (AI Agent)*
