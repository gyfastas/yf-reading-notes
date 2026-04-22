# Attention Residuals 论文精读报告

> **论文标题**: Attention Residuals
> **作者**: Kimi Team (Guangyu Chen, Yu Zhang, Jianlin Su 等)
> **机构**: Moonshot AI (月之暗面)
> **发表日期**: 2026/03/16
> **arXiv**: https://arxiv.org/abs/2603.15031
> **GitHub**: https://github.com/MoonshotAI/Attention-Residuals

---

## 一、研究背景与动机

### 1.1 传统残差连接的局限性

现代 LLM 中，带 PreNorm 的残差连接是标准组件：

```
h_l = h_{l-1} + f_{l-1}(h_{l-1})
```

但存在三个核心问题：

| 问题 | 说明 |
|------|------|
| **无选择性访问** | 每层只能访问前一层输出，无法直接访问更早层的独立输出 |
| **不可逆信息损失** | 深层聚合的信息无法在后续层被选择性恢复 |
| **输出增长** | 深层需要学习越来越大的输出来在累加残差中保持影响力，导致训练不稳定 |

### 1.2 核心观察：Time-Depth Duality

作者观察到**序列维度和深度维度的对偶性**：
- RNNs 通过 recurrence 在 **time** 维度压缩信息
- Residual connections 通过累加在 **depth** 维度压缩信息
- Transformer 用 attention 解决了 RNN 的时间瓶颈 → 能否用 attention 解决残差的深度瓶颈？

---

## 二、核心方法：Attention Residuals

### 2.1 基本定义

**公式 (1)** - Attention Residuals 核心更新规则：

```
h_l = α_{0→l} · h_1 + Σ_{i=1}^{l-1} α_{i→l} · f_i(h_i)
```

其中 α_{i→l} 是层特定的注意力权重，满足 Σ α = 1。

对比标准残差：
```
# 标准残差 (固定权重)
h_l = h_1 + f_1(h_1) + f_2(h_2) + ... + f_{l-1}(h_{l-1})

# Attention Residuals (学习权重)
h_l = softmax-attention({h_1, f_1(h_1), ..., f_{l-1}(h_{l-1})})
```

### 2.2 两种变体

#### (1) Full Attention Residuals

**公式 (2)(3)(4)** - 完整的 softmax attention 机制：

```
α_{i→l} = exp(w_l^T · RMSNorm(v_i)) / Σ_j exp(w_l^T · RMSNorm(v_j))

其中:
- q_l = w_l (层特定的可学习伪查询向量)
- k_i = v_i = {h_1 (i=0) 或 f_i(h_i) (i≥1)} (value 也是 key)
- RMSNorm 防止大输出层主导 attention 权重
```

**复杂度分析**：
- 计算：O(L²d) — 每层对所有前层做 attention
- 内存：O(Ld) — 需要保存所有层输出

#### (2) Block Attention Residuals (实用版本)

为解决大规模训练的可扩展性问题，提出 Block AttnRes：

**核心思想**：将 L 层划分为 N 个 block，每个 block 内做累加，block 之间做 attention

**公式 (5)** - Block 表示：
```
b_n = Σ_{j∈B_n} f_j(h_j)  (第 n 个 block 的累加和)
```

**复杂度改进**：
| 方案 | 内存 | 通信 | 计算 |
|------|------|------|------|
| Full AttnRes | O(Ld) | O(Ld) | O(L²d) |
| Block AttnRes | O(Nd) | O(Nd) | O(N²d) |
| 标准残差 | O(d) | O(d) | O(Ld) |

实际设置：N ≈ 8 时可恢复大部分 Full AttnRes 的性能收益。

---

## 三、系统设计：大规模训练的基础设施

### 3.1 两阶段计算策略

**Algorithm 1** - Block AttnRes 的两阶段计算：

```python
# Phase 1: 并行的 block 间 attention
# 对 block 内所有层同时计算对之前所有 block 的 attention
Q = [w_l for l in B_n]           # [S, d]
K = V = [b_0, ..., b_{n-1}]      # [n, d]
o_l^(1), m_l^(1), l_l^(1) = AttnWithStats(Q, K, V)

# Phase 2: 顺序的 block 内 attention + online softmax merge
for l in B_n:
    # block 内局部 attention
    o_l^(2), m_l^(2), l_l^(2) = AttnWithStats(w_l, b_n^i, b_n^i)

    # online softmax merge 合并 Phase 1 和 Phase 2
    h_l = merge(o_l^(1), m_l^(1), l_l^(1), o_l^(2), m_l^(2), l_l^(2))

    # 更新 block 内累加和
    b_n^{i+1} = b_n^i + f_l(h_l)
```

### 3.2 Pipeline 并行优化

**跨 stage 缓存 (Cross-stage caching)**：

在 pipeline 并行中，关键优化是避免重复传输：
- 朴素实现：每轮传输所有累积的 block → 通信量 O(C²)
- 缓存优化：每个 stage 本地缓存已接收的 block → 通信量 O(P² + VP²)

**效果**：将每轮通信从 O(C) 降低到 O(P)，实现 V× 提升。

### 3.3 内存高效预填充

- 128K token 序列 + 8 blocks 需要 N·T·d = 15GB 内存
- 通过 Tensor Parallel 将表示分片到 P 个设备
- Phase 1 在各设备独立执行，Phase 2 通过 all-gather 合并
- **结果**：每设备内存从 15GB → 1.9GB (P=8)，配合 chunked prefill 可降至 <0.3GB

---

## 四、实验验证

### 4.1 Scaling Laws

**设置**：5 种模型规模 (194M-528M 激活参数)，在相同 compute budget 下比较

**公式 (图 4)**：
```
Baseline:    L = 1.891 × C^{-0.057}
Block AttnRes: L = 1.870 × C^{-0.058}
Full AttnRes:  L = 1.865 × C^{-0.057}
```

**关键发现**：
- AttnRes 在所有规模上都一致优于基线
- Block AttnRes 在 5.6 PFLOP/s-days 时达到 1.692 loss，相当于基线 1.25× compute 的优势
- Full 和 Block 的差距随规模缩小，最大规模仅差 0.001

### 4.2 主要结果 (48B 模型)

**训练设置**：
- 模型：Kimi Linear 48B (3B 激活参数，48B 总参数)
- 数据：1.4T tokens
- Block AttnRes：6 层/block，9 blocks + token embedding

**训练动态分析 (图 5)**：

| 指标 | 基线 | Block AttnRes | 观察 |
|------|------|---------------|------|
| 验证 Loss | 较高 | 较低 | 训练全程保持优势，后期差距扩大 |
| 输出幅度 | 随深度单调增长 | 周期性有界 | 解决 PreNorm dilution 问题 |
| 梯度幅度 | 早期层异常大 | 各层更均匀 | 梯度分布更健康 |

**下游任务 (表 3)**：

| 任务类型 | 基准 | AttnRes | 提升 |
|----------|------|---------|------|
| MMLU | 73.5 | 74.6 | +1.1 |
| GPQA-Diamond | 36.9 | 44.4 | **+7.5** |
| HumanEval | 59.1 | 62.2 | +3.1 |
| Math | 53.5 | 57.1 | +3.6 |
| CMMLU | 82.0 | 82.9 | +0.9 |

**规律**：在需要多步推理的任务上提升最明显（GPQA、Math、Code）。

### 4.3 消融实验

**表 4 - 组件消融** (16 层模型)：

| 变体 | Loss | 说明 |
|------|------|------|
| Baseline (PreNorm) | 1.766 | - |
| DenseFormer | 1.767 | 无改进 |
| mHC | 1.747 | 有改进 |
| AttnRes Full | **1.737** | 最佳 |
| AttnRes Block (S=4) | 1.746 | 接近 Full |

**关键发现**：
- **输入依赖查询** vs **输入独立**：1.731 vs 1.749 → 输入依赖更有优势但需顺序解码
- **Softmax vs Sigmoid**：1.737 vs 1.741 → softmax 的竞争性归一化更优
- **RMSNorm 的重要性**：去掉后 Full 1.743，Block 1.750 → 对 Block 更关键

**图 6 - Block Size 影响**：
- S=1: 标准残差 (1.766)
- S=2,4,8: 约 1.746
- S=16,32: 接近基线
- **推荐**：N ≈ 8 (即 S ≈ L/8)

### 4.4 学习到的 Attention 模式 (图 8)

对 16 层模型的 attention 权重可视化：

1. **保持局部性**：每层主要关注直接前驱层
2. **学习到的跳跃连接**：存在明显的非对角浓度（如第 4 层关注早期层，第 15-16 层回连）
3. **层专业化**：
   - Pre-Attention：保持较宽的接受野
   - Pre-MLP：更尖锐的对角依赖
4. **Block AttnRes 保持结构**：Block 化不会破坏学到的模式

---

## 五、结构化视角：残差作为深度混合矩阵

### 5.1 统一框架

**公式**：输入到第 l 层可表示为：
```
h_l = Σ_{i=0}^{l-1} M_{l→i} · v_i
```

其中 M 是深度混合矩阵，v_0 = h_1 (embedding)，v_i = f_i(h_i) (i≥1)。

### 5.2 不同方法的矩阵形式 (图 9)

| 方法 | 矩阵 M 特性 |
|------|------------|
| 标准残差 | 全 1 下三角矩阵 |
| Highway | 1-半可分矩阵，输入依赖标量门 |
| (m)HC | m-半可分矩阵，m 并行流 |
| **Full AttnRes** | 稠密、秩-L 矩阵，输入依赖 softmax |
| **Block AttnRes** | 块对角 + 块间连接，秩介于 N 和 N+S 之间 |

### 5.2 残差变体的统一分类 (表 5)

**Single-state recurrence** (每层只访问 h_{l-1}):
- Residual, ReZero, LayerScale, Highway, DeepNorm, KEEL

**Multi-state recurrence** (访问多个流):
- SiameseNorm, HC/mHC, DDL

**Cross-layer access** (访问所有前层输出):
- DenseNet, DenseFormer, MRLA
- **AttnRes (Ours)**: 唯一使用 softmax 归一化 + 输入依赖权重 + 选择性跨层访问

---

## 六、核心洞察与贡献总结

### 6.1 主要贡献

1. **Attention Residuals**: 用学习的深度 attention 替代固定残差累加
2. **Block AttnRes**: 实用的可扩展变体，内存/通信从 O(Ld) 降到 O(Nd)
3. **基础设施设计**: 跨 stage 缓存、两阶段计算、内存高效预填充
4. **实验验证**: 在 48B 模型/1.4T tokens 上验证，全面超越基线

### 6.2 关键洞察

1. **深度 attention sinks**: 某些层无论输入如何都持续吸引高权重（类似序列中的 attention sinks）

2. **架构重分配**: AttnRes 改变了最优的 depth-width-attention 权衡，允许更深的网络

3. **结构保持压缩**: Block AttnRes 的 block 化是一种隐式正则化，保持关键信息流

4. **训练动态改善**:
   - 解决 PreNorm dilution（输出幅度有界）
   - 梯度分布更均匀
   - 早期层不再需要学习极大输出

### 6.3 局限与未来方向

- 当前只在 modest depth (L<1000) 上验证
- 未来硬件可能支持 Full AttnRes 而非 Block 版本
- 可探索线性复杂度的深度注意力替代方案

---

## 七、与 VLM/LLM 研究的关联

### 7.1 对 VLM 的意义

1. **更稳定的深层训练**: VLM 通常需要更深的网络处理视觉-语言对齐，AttnRes 的深度稳定性有帮助
2. **更好的梯度流**: 多模态训练中梯度问题更严重，均匀的梯度分布有益
3. **长序列支持**: 48B 模型上验证的长上下文能力可迁移到 VLM

### 7.2 可能的扩展方向

- 在 VLM backbone 中替换残差连接
- 探索 vision encoder 和 language decoder 使用不同的 AttnRes 配置
- 结合 cross-modal attention 和 depth-wise attention

---

## 八、关键公式速查

```python
# Full Attention Residuals
h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i
α_{i→l} = softmax(q_l^T · RMSNorm(k_i))
q_l = w_l (learnable pseudo-query)
k_i = v_i = {h_1 if i=0 else f_i(h_i)}

# Block Attention Residuals
# 分 N 个 blocks，每 block S = L/N 层
b_n = Σ_{j∈B_n} f_j(h_j)  # block 内累加
h_l = attention([b_0, ..., b_{n-1}, partial_sum])  # block 间 attention
```

---

## 九、参考资料

- 论文: https://arxiv.org/abs/2603.15031
- 代码: https://github.com/MoonshotAI/Attention-Residuals
- Kimi Linear: https://arxiv.org/abs/2510.26692
- Moonlight: https://arxiv.org/abs/2502.16982

---

*精读报告生成时间: 2026-03-17*
*分析师: 王明达 (Claude Code)*
