# Agentic RL Curriculum Learning 调研总结

## 核心洞察

### 1. 为什么 Agentic RL 特别需要 Curriculum Learning？

| 挑战 | 传统 RL | Agentic RL |
|------|---------|-----------|
| 动作空间 | 离散/连续 | 文本生成 + 工具调用 |
| Horizon | 固定 episode | 动态、可能极长 |
| 奖励 | 手工设计 | LLM-as-Judge |
| 状态 | 低维观测 | 多模态 + 历史上下文 |

### 2. Horizon 与 Curriculum 的关系

```
短 Horizon (H=5-10)          长 Horizon (H=100+)
    ↓                             ↓
简单子任务                    复杂完整任务
    ↓                             ↓
高成功率                      需要分解
    ↓                             ↓
快速学习基础技能              组合已学技能
    ↓                             ↓
进入下一阶段                  最终目标
```

### 3. 四种主流方法

#### 方法 1: 难度分层
- **代表**: UCL, Learning Progress
- **核心**: 根据性能自动调整任务难度

#### 方法 2: 目标条件 (HER)
- **代表**: Hindsight Experience Replay
- **核心**: 将失败经验重标注为成功

#### 方法 3: LLM 驱动
- **代表**: Evolving Curriculum, Voyager
- **核心**: 用 LLM 生成、评估、调整任务

#### 方法 4: 对抗设计
- **代表**: PAIRED, ACCEL
- **核心**: 生成器 vs 学习者博弈

### 4. 关键实现要点

```python
# Horizon-Aware Curriculum 核心逻辑
curriculum = {
    "stage1": {"H": 5,  "task": "单步工具调用",     "threshold": 0.8},
    "stage2": {"H": 10, "task": "多步序列",         "threshold": 0.8},
    "stage3": {"H": 25, "task": "条件逻辑",         "threshold": 0.8},
    "stage4": {"H": 50, "task": "复杂推理链",       "threshold": 0.7},
}

def should_advance(agent, stage):
    return agent.success_rate > stage.threshold
```

### 5. 必读论文清单

| 优先级 | 论文 | 年份 | 核心贡献 |
|--------|------|------|----------|
| ⭐⭐⭐ | Hindsight Experience Replay | 2017 | HER，自动课程基础 |
| ⭐⭐⭐ | PAIRED | 2020 | 对抗式环境设计 |
| ⭐⭐⭐ | Voyager | 2023 | Agentic RL + 自动课程 |
| ⭐⭐ | ACCEL | 2023 | 在线自适应课程 |
| ⭐⭐ | Agent Workflow Memory | 2024 | 工作流课程学习 |
| ⭐ | Let It Learn | 2024 | 全自动课程设计 |

### 6. 快速启动建议

**如果你正在设计 Agentic RL 系统：**

1. **评估当前 horizon**
   - 记录成功 episode 的平均步数
   - 如果 > 50，强烈建议使用课程

2. **设计课程阶段**
   - 从 H=5-10 的简单子任务开始
   - 每阶段增加 1.5-2x 的 horizon
   - 设置 0.7-0.9 的成功率门槛

3. **选择课程策略**
   - 有明确子任务分解 → 手工课程
   - 需要自动生成 → LLM 驱动课程
   - 需要最优难度 → 对抗/自适应课程

4. **监控指标**
   - 各阶段成功率
   - 跨阶段迁移效率
   - 总样本效率

---

## 文件结构

```
paper_reading/agents/curriculum_horizon_rl/
├── README.md              # 完整调研文档
├── SUMMARY.md             # 本总结
├── horizon_analysis.md    # Horizon 深度分析
└── papers.json            # 论文列表（含链接）
```

## 飞书文档

- 主文档：https://bytedance.larkoffice.com/docx/BqPpd2znnoNhrlxXltElZcx4gUg

---

*调研完成时间：2026-04-09*
