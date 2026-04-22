# Horizon-Aware Curriculum Learning in Agentic RL

> 深度分析：Horizon 在 Agentic RL 课程学习中的关键作用

---

## 1. Horizon 的多重含义

### 1.1 Time Horizon ($H$)

单次 episode 的最大步数。在 Agentic RL 中通常指：
- 最大 LLM 调用次数
- 最大工具调用次数
- 最大思考-行动循环次数

### 1.2 Planning Horizon

Agent 在决策时考虑的未来时间范围：
- **短规划**：仅考虑下一步行动
- **长规划**：考虑多步后果

### 1.3 Curriculum Horizon

整个课程覆盖的复杂度范围：
- 任务数量
- 难度跨度
- 概念依赖深度

---

## 2. 长 Horizon 带来的挑战

### 2.1 信用分配问题

在长序列中，奖励信号难以追溯到具体动作：

```
问题：G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...

当 H = 100 时：
- γ^100 ≈ 0  (γ=0.99)
- 早期动作的梯度几乎为0
```

**课程解决方案**：
- 从短 horizon 开始 (H=5)
- 逐步扩展到长 horizon (H=100)
- 中间阶段使用分层 RL

### 2.2 探索效率

长 horizon 导致状态空间指数级增长：

```
|A| = 10 (动作空间)
H = 10  → 10^10 可能序列
H = 100 → 10^100 可能序列
```

**课程解决方案**：
- 限制早期任务的可达状态
- 逐步开放新区域/新动作
- 使用 HER 将失败转化为成功

### 2.3 内存与上下文限制

Agentic RL 通常使用 LLM 作为策略：

```
限制因素：
- LLM 上下文窗口 (4k-200k tokens)
- 长历史导致注意力分散
- 推理成本随序列增长
```

**课程解决方案**：
- 早期：短历史，完整保留
- 中期：引入摘要机制
- 后期：使用外部记忆 (RAG)

---

## 3. Horizon-Based Curriculum 策略

### 3.1 逐步扩展 Horizon (Progressive Horizon Expansion)

**算法**：
```python
def progressive_horizon_curriculum():
    H_schedule = [5, 10, 20, 50, 100]
    threshold = 0.8  # success rate threshold
    
    for H_target in H_schedule:
        while agent.success_rate < threshold:
            train_on_horizon(H_target)
        # 达到阈值后进入下一阶段
```

**适用场景**：
- 任务完成有明确终点
- 难度主要由步数决定

### 3.2 Hierarchical Horizon Decomposition

**核心思想**：将长 horizon 分解为短 horizon 子任务

```
完整任务 (H=100):
[搜索论文] → [阅读摘要] → [提取方法] → [对比分析] → [撰写总结]

子任务 (各 H=20):
1. 搜索论文: 给定关键词，找到相关论文
2. 阅读摘要: 给定论文，总结要点
3. 提取方法: 给定摘要，提取方法论
4. 对比分析: 给定多篇文章方法，做对比
5. 撰写总结: 给定对比结果，生成综述
```

**课程设计**：
1. 单独训练每个子任务
2. 学习子任务间的转移
3. 端到端微调完整任务

### 3.3 Adaptive Horizon Adjustment

**核心思想**：根据学习状态动态调整 horizon

**算法 (基于成功率)**：
```python
def adaptive_horizon(agent, H_current, performance):
    if performance > 0.9:
        return H_current * 1.5  # 增加难度
    elif performance < 0.3:
        return H_current * 0.8  # 降低难度
    return H_current
```

**算法 (基于学习进度)**：
```python
def learning_progress_horizon(agent, task_history):
    # 计算最近k个任务的进步速度
    progress = compute_learning_progress(agent, task_history)
    
    # 选择能带来最大进步的任务难度
    if progress > threshold:
        return increase_difficulty()
    else:
        return decrease_difficulty()
```

### 3.4 Curriculum with Hindsight Replay

**HER for Agentic RL**：

```python
# 传统 HER: 将最终状态作为目标
# Agentic HER: 使用 LLM 重标注轨迹

def hindsight_relabeling(trajectory, task_description):
    # 原始任务失败
    if not trajectory.success:
        # 用 LLM 提取实际完成的子任务
        achieved_goal = llm_extract_achievement(trajectory)
        
        # 重标注为新任务的成功经验
        new_task = f"完成: {achieved_goal}"
        return (trajectory, new_task, reward=1.0)
    
    return None
```

---

## 4. 与 LLM 结合的特殊考虑

### 4.1 LLM-as-Curriculum-Designer

**方法**：使用 LLM 生成任务变体并估计难度

```python
def llm_generate_task_variants(base_task, difficulty_range):
    prompt = f"""
    基于以下基础任务，生成5个不同难度的变体：
    
    基础任务: {base_task}
    
    难度范围: 1-5 (1最简单，5最难)
    
    对每个变体：
    1. 描述任务
    2. 估计难度 (1-5)
    3. 估计完成所需步数
    4. 列出需要的技能
    """
    
    return llm.generate(prompt)
```

### 4.2 LLM-as-Horizon-Controller

**方法**：使用 LLM 决定何时扩展 horizon

```python
def llm_decide_horizon_expansion(agent_history, current_horizon):
    prompt = f"""
    当前训练状态：
    - Horizon: {current_horizon}
    - 最近10次成功率: {agent_history.success_rates}
    - 平均完成步数: {agent_history.avg_steps}
    
    是否应该：
    A. 保持当前 horizon
    B. 增加 horizon (+10步)
    C. 减少 horizon (-5步)
    
    请分析并给出决策。
    """
    
    decision = llm.generate(prompt)
    return parse_decision(decision)
```

---

## 5. 评估指标

### 5.1 Horizon-Specific Metrics

| 指标 | 定义 | 用途 |
|------|------|------|
| Normalized Return | $\frac{G}{H \cdot r_{max}}$ | 跨 horizon 比较 |
| Success Rate vs H | 不同 H 下的成功率 | 判断 horizon 扩展时机 |
| Sample Efficiency | $\frac{\text{Success}}{\text{Interactions}}$ | 评估学习效率 |
| Curriculum Length | 达到目标所需的课程阶段数 | 评估课程质量 |
| Transfer Gap | $\text{Perf}_{\text{curriculum}} - \text{Perf}_{\text{no curriculum}}$ | 课程带来的提升 |

### 5.2 Agentic RL 特有指标

- **LLM 调用次数**：实际的 API 调用成本
- **有效思考比例**：推理步骤中实际有用的比例
- **工具使用准确率**：工具选择和参数填充的正确率
- **任务分解合理性**：LLM 生成的子任务是否可执行

---

## 6. 实现建议

### 6.1 代码框架

```python
class HorizonCurriculumManager:
    def __init__(self, config):
        self.horizon_schedule = config.horizon_schedule
        self.current_stage = 0
        self.performance_buffer = []
        
    def get_current_horizon(self):
        return self.horizon_schedule[self.current_stage]
    
    def update(self, episode_result):
        self.performance_buffer.append(episode_result.success)
        
        if len(self.performance_buffer) >= 100:
            success_rate = np.mean(self.performance_buffer)
            
            if success_rate > self.config.threshold:
                self.advance_stage()
                self.performance_buffer = []
    
    def advance_stage(self):
        if self.current_stage < len(self.horizon_schedule) - 1:
            self.current_stage += 1
            logger.info(f"Advanced to stage {self.current_stage}, "
                       f"H={self.get_current_horizon()}")
```

### 6.2 超参数选择

| 参数 | 建议值 | 说明 |
|------|--------|------|
| Initial H | 5-10 | 确保初始成功率 > 30% |
| H growth rate | 1.5-2x | 指数增长或线性增长 |
| Success threshold | 0.7-0.9 | 进入下一阶段的门槛 |
| Buffer size | 50-100 | 评估性能的样本数 |
| Max stages | 5-10 | 防止过度细分 |

---

## 7. 案例研究

### 7.1 Voyager (Minecraft Agent)

**课程设计**：
- Level 1: 采集木头 (H≈5)
- Level 2: 制作工具 (H≈10)
- Level 3: 采集矿石 (H≈20)
- Level 4: 建造结构 (H≈50)
- Level 5: 复杂任务 (H≈100+)

**关键技术**：
- 技能库 (Skill Library)：保存可复用的短 horizon 策略
- 自动课程生成：基于当前库存和技能自动生成新任务

### 7.2 Web Agent (如 WebGPT)

**课程设计**：
- Level 1: 单步检索 (H=1)
- Level 2: 多步导航 (H=5)
- Level 3: 信息整合 (H=10)
- Level 4: 复杂查询 (H=20)

**关键技术**：
- 搜索历史作为课程进度指标
- 点击成功率决定 horizon 扩展

---

## 8. 未来方向

1. **理论上界分析**：Curriculum + Horizon 的最优策略理论
2. **元学习课程**：学习如何为不同任务生成课程
3. **多智能体课程**：智能体之间互为课程
4. **跨模态课程**：视觉→语言→行动的渐进学习
5. **终身学习**：持续扩展的无限课程

---

*Last Updated: 2026-04-09*
