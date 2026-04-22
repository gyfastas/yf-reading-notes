# Agent Framework 论文索引

> 按范式演进顺序排列

## 基础范式

| 文件 | 论文 | 核心贡献 |
|------|------|---------|
| `Toolformer-2302.04761.pdf` | **Toolformer** (Schick et al., 2023) | 自监督学习工具调用，LLM 自己学何时/如何调工具 |
| `ReAct-2210.03629.pdf` | **ReAct** (Yao et al., 2022) | Thought-Action-Observation 交替，推理与行动协同 |
| `Reflexion-2303.11366.pdf` | **Reflexion** (Shinn et al., 2023) | 失败后语言自我反思存 memory，verbal RL |
| `Tree-of-Thoughts-2305.10601.pdf` | **Tree of Thoughts** (Yao et al., 2023) | 分支搜索推理路径，非线性规划 |

## 代码动作范式

| 文件 | 论文 | 核心贡献 |
|------|------|---------|
| `CodeAct-2402.01030.pdf` | **CodeAct** (Wang et al., 2024) | 动作=Python代码，持久解释器，替代结构化工具调用 |
| `SWE-agent-2405.15793.pdf` | **SWE-agent** (Yang et al., 2024) | ACI 接口设计，专为软件工程任务优化的 agent-computer 交互 |
| `OpenHands-2407.16741.pdf` | **OpenHands** (Wang et al., 2024) | CodeAct 完整实现平台，多 agent runtime |

## 多智能体范式

| 文件 | 论文 | 核心贡献 |
|------|------|---------|
| `AutoGen-2308.08155.pdf` | **AutoGen** (Wu et al., 2023) | 多 agent 对话框架，可编程的 agent 交互模式 |

## GUI / 计算机操作范式

| 文件 | 论文 | 核心贡献 |
|------|------|---------|
| `GUI-Agent-Survey-2501.12599.pdf` | **GUI Agent Survey** (2025) | GUI agent 综述，Computer Use 范式全景 |

## 协议标准化

| 文件 | 论文 | 核心贡献 |
|------|------|---------|
| `MCP-AgentProtocol-2406.08689.pdf` | **Agent Protocol** (2024) | agent 间通信协议标准化 |

---

## 范式演进脉络

```
Toolformer (2023)      自学调工具
    ↓
ReAct (2022)           Thought-Action-Observation 循环
    ↓
Reflexion (2023)       失败 → 语言反思 → memory → 重试
Tree-of-Thoughts (2023) 分支搜索，非线性规划
    ↓
CodeAct (2024)         动作 = Python 代码，持久状态
SWE-agent (2024)       专业化 ACI 接口
OpenHands (2024)       CodeAct 完整平台
    ↓
AutoGen (2023-)        多 agent 编排，orchestrator 模式
    ↓
Computer Use (2024)    动作 = GUI 操作（截图→点击/输入）
    ↓
MCP / A2A (2024-25)    工具/agent 互操作协议标准化
```
