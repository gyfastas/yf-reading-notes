# AgentGym 技术报告

**项目链接**: https://github.com/WooooDyy/AgentGym
**论文**: [arXiv:2406.04151](https://arxiv.org/abs/2406.04151) (ACL 2025)
**RL版本论文**: [arXiv:2509.08755](https://arxiv.org/abs/2509.08755)
**项目主页**: https://agentgym.github.io/
**数据集**: Hugging Face `AgentGym/AgentTraj-L`, `AgentGym/AgentEval`

---

## 1. 项目概述

### 1.1 核心定位

AgentGym 是一个用于开发和评估**基于大语言模型的通用智能体（General-Purpose LLM Agents）**的开放平台。与针对特定任务设计的Agent框架不同，AgentGym 的目标是创建能够在多种环境中自我进化的通用智能体。

### 1.2 核心功能

| 功能模块 | 说明 |
|---------|------|
| **14+ 环境支持** | 涵盖网页导航、文本游戏、家务任务、数字游戏、具身任务、工具使用、编程等领域 |
| **统一接口** | 采用 ReAct 格式实现统一的交互接口（Thought + Action） |
| **高质量数据集** | AgentTraj-L：大规模轨迹数据集用于训练 |
| **评估基准** | AgentEval：全面的跨环境评估套件 |
| **自我进化方法** | AgentEvol：探索Agent在跨任务、跨环境中的自我进化能力 |
| **RL训练支持** | AgentGym-RL (2025.09新增)：支持多轮强化学习训练 |

---

## 2. 与 verl-agent 的对比分析

### 2.1 核心差异对比表

| 维度 | AgentGym | verl-agent |
|------|----------|------------|
| **核心目标** | 通用Agent开发+评估平台，支持多种环境 | 专注于Agent的RL训练框架 |
| **环境数量** | 14+ 种异构环境 | 主要支持特定任务环境 |
| **架构设计** | 微服务架构，HTTP API解耦 | 基于veRL的单体架构 |
| **训练方法** | 行为克隆(BC) + 进化 + RL (AgentGym-RL) | 主要专注于RL训练 |
| **轨迹数据** | 提供大规模高质量轨迹数据集 | 训练时在线生成 |
| **环境实现** | 模块化、可扩展的环境服务 | 通常与训练代码紧耦合 |
| **交互格式** | ReAct（Thought + Action） | 灵活的文本/工具调用格式 |
| **可视化** | 提供交互式前端界面 | 命令行/日志为主 |

### 2.2 架构设计差异

```
AgentGym (微服务架构):
┌─────────────────────────────────────────────────────────────┐
│                    AgentGym Platform                        │
├─────────────────────────────────────────────────────────────┤
│  AgentController (训练/评估控制器)                           │
│       │                                                     │
│       │ HTTP API                                            │
│       ↓                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │EnvServer │  │EnvServer │  │EnvServer │  ...             │
│  │(WebShop) │  │(ALFWorld)│  │ (SciWorld)│                  │
│  └──────────┘  └──────────┘  └──────────┘                  │
│     独立部署    独立部署      独立部署                       │
│     不同端口    不同端口      不同端口                       │
└─────────────────────────────────────────────────────────────┘

verl-agent (单体架构):
┌─────────────────────────────────────────────────────────────┐
│                    verl-agent Framework                     │
├─────────────────────────────────────────────────────────────┤
│  RL Trainer (veRL)                                          │
│       │                                                     │
│       │ 内部函数调用                                         │
│       ↓                                                     │
│  Environment (与训练代码紧耦合)                              │
│       │                                                     │
│       ↓                                                     │
│  Tool/API/Env                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 适用场景对比

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| 研究通用Agent能力 | AgentGym | 多环境、跨任务评估 |
| 特定任务的RL优化 | verl-agent | 专注RL训练流程 |
| 需要预训练轨迹数据 | AgentGym | 提供AgentTraj-L数据集 |
| 快速环境原型开发 | AgentGym | 模块化EnvServer设计 |
| 大规模并行RL训练 | verl-agent | 基于veRL的高效并行 |
| 多模态Agent研究 | AgentGym | 支持具身、网页、工具等多模态 |

---

## 3. 架构设计详解

### 3.1 微服务架构

AgentGym 采用**微服务架构**，这是其与 verl-agent 最核心的架构差异：

> "different environments are deployed on different servers or ports and provide encapsulated HTTP services externally. This decouples the environments from other parts."

**核心优势**:
1. **解耦**：环境独立于训练/评估代码
2. **可扩展**：新增环境只需部署新的EnvServer
3. **并发**：每个环境可以独立扩展和负载均衡
4. **语言无关**：EnvServer可以用任何语言实现（只要提供HTTP接口）

### 3.2 核心组件

```
┌─────────────────────────────────────────────────────────────────┐
│                      AgentGym Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              AgentController (核心控制器)                │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │   │
│  │  │   Evaluator │ │ DataCollector│ │   Trainer   │       │   │
│  │  │   (评估)    │ │  (数据收集)  │ │   (训练)    │       │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘       │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │ HTTP API                          │
│                            ↓                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              EnvClient (环境客户端封装)                  │   │
│  │     提供统一接口: create, step, reset, observation       │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │ HTTP Request                      │
│          ┌─────────────────┼─────────────────┐                 │
│          ↓                 ↓                 ↓                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  EnvServer   │  │  EnvServer   │  │  EnvServer   │         │
│  │  (WebShop)   │  │  (ALFWorld)  │  │  (SciWorld)  │         │
│  │  :3001       │  │  :3002       │  │  :3003       │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 标准API接口

每个 EnvServer 必须实现以下 RESTful API：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/createEnv` | POST | 创建新环境实例 |
| `/observation` | GET | 获取当前观察 |
| `/available_actions` | GET | 获取当前可用动作 |
| `/step` | POST | 执行动作，返回新观察 |
| `/reset` | POST | 重置环境 |

**请求/响应示例**:
```python
# 创建环境
POST /createEnv
Response: {"env_id": "webshop_001", "status": "created"}

# 执行动作
POST /step
Body: {"env_id": "webshop_001", "action": "search[black shoes]"}
Response: {
    "observation": "You are looking for black shoes...",
    "reward": 0,
    "done": False,
    "info": {}
}
```

---

## 4. 环境(Env)部分实现详解

### 4.1 环境分类与实现

AgentGym 包含14种环境，按类型分类：

#### 4.1.1 网页交互类

| 环境 | 描述 | 原始仓库 | 实现目录 |
|------|------|---------|---------|
| **WebShop** | 电商购物网站模拟 | princeton-nlp/WebShop | agentenv-webshop |
| **WebArena** | 真实网页任务基准 | web-arena-x/webarena | agentenv-webarena |

**实现特点**:
- WebShop：基于文本的电商环境，支持搜索、浏览、购买
- WebArena：在真实网站上执行任务（如预订、购物、信息查询）

#### 4.1.2 文本游戏类

| 环境 | 描述 | 原始仓库 | 实现目录 |
|------|------|---------|---------|
| **MAZE** | 迷宫导航 | LMRL-Gym | agentenv-lmrlgym |
| **Wordle** | 猜词游戏 | LMRL-Gym | agentenv-lmrlgym |
| **TextCraft** | 文本版Minecraft | archiki/ADaPT | agentenv-textcraft |

**实现特点**:
- 纯文本交互，基于规则的状态转换
- 需要Agent理解指令、推理和规划

#### 4.1.3 具身智能类

| 环境 | 描述 | 原始仓库 | 实现目录 |
|------|------|---------|---------|
| **ALFWorld** | 家务任务模拟 | alfworld/alfworld | agentenv-alfworld |
| **SciWorld** | 科学实验模拟 | allenai/ScienceWorld | agentenv-sciworld |
| **BabyAI** | 网格世界导航 | mila-iqia/babyai | agentenv-babyai |

**实现特点**:
- ALFWorld：基于TextWorld的家务任务（找物品、清洁等）
- SciWorld：科学实验场景，需要操作实验器材
- BabyAI：视觉+指令的网格导航

#### 4.1.4 工具使用类

| 环境 | 描述 | 原始仓库 | 实现目录 |
|------|------|---------|---------|
| **Weather** | 天气查询工具 | hkust-nlp/AgentBoard | agentenv-tool |
| **Movie** | 电影信息查询 | hkust-nlp/AgentBoard | agentenv-tool |
| **Academia** | 学术搜索工具 | hkust-nlp/AgentBoard | agentenv-tool |
| **Sheet** | 电子表格操作 | hkust-nlp/AgentBoard | agentenv-tool |
| **TODOList** | 待办事项管理 | hkust-nlp/AgentBoard | agentenv-tool |

**实现特点**:
- 统一的工具调用接口
- Agent学习何时、如何调用外部API

#### 4.1.5 编程/数据库类

| 环境 | 描述 | 原始仓库 | 实现目录 |
|------|------|---------|---------|
| **BIRD** | SQL生成与执行 | AlibabaResearch/DAMO-ConvAI | agentenv-sqlgym |

**实现特点**:
- 基于真实数据库的SQL查询任务
- 评估Agent理解和生成SQL的能力

### 4.2 环境实现代码结构

以 `agentenv-webshop` 为例：

```python
# agentenv_webshop/server.py
from flask import Flask, request, jsonify
from webshop import WebShopEnv  # 原始环境

app = Flask(__name__)
envs = {}  # 环境实例管理

@app.route('/createEnv', methods=['POST'])
def create_env():
    env_id = generate_env_id()
    envs[env_id] = WebShopEnv()
    return jsonify({"env_id": env_id, "status": "created"})

@app.route('/step', methods=['POST'])
def step():
    data = request.json
    env_id = data['env_id']
    action = data['action']

    env = envs[env_id]
    obs, reward, done, info = env.step(action)

    return jsonify({
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    })

@app.route('/reset', methods=['POST'])
def reset():
    env_id = request.json['env_id']
    obs = envs[env_id].reset()
    return jsonify({"observation": obs})

if __name__ == '__main__':
    app.run(port=3001)  # 每个环境不同端口
```

### 4.3 环境封装的关键设计

```
┌─────────────────────────────────────────────────────────────────┐
│                   EnvServer 封装层                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                Original Environment                      │   │
│  │  (WebShop/ALFWorld/SciWorld/etc.)                        │   │
│  │       ↓ 原生API（各异）                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ↓ 封装                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              AgentGym EnvServer                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │ 统一接口：                                       │   │   │
│  │  │ - /createEnv → env.reset()                       │   │   │
│  │  │ - /step → env.step(action)                       │   │   │
│  │  │ - /observation → env.get_obs()                   │   │   │
│  │  │ - /available_actions → env.get_actions()         │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ↓ HTTP                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              AgentController                            │   │
│  │              (训练/评估代码)                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 开发新环境的步骤

根据官方文档，添加新环境只需：

1. **创建EnvServer** 实现标准接口
```python
# myenv_server.py
from flask import Flask
app = Flask(__name__)

@app.route('/createEnv', methods=['POST'])
def create_env(): ...

@app.route('/step', methods=['POST'])
def step(): ...
# ... 其他接口
```

2. **注册到配置文件**
```json
{
  "environments": {
    "myenv": {
      "port": 3005,
      "entry": "myenv_server.py"
    }
  }
}
```

3. **启动服务**
```bash
python myenv_server.py --port 3005
```

---

## 5. AgentTraj-L 轨迹数据集

### 5.1 数据集概览

**AgentTraj-L** 是AgentGym提供的大规模高质量轨迹数据集：

| 环境 | 轨迹数 | 评估数 |
|------|--------|--------|
| WebShop | 3,930 | 200 |
| ALFWorld | 2,420 | 200 |
| SciWorld | 2,120 | 200 |
| BIRD | 3,000 | 200 |
| ... | ... | ... |

**数据格式**:
```json
{
  "conversations": [
    {"from": "human", "value": "Task: Find me a coffee maker..."},
    {"from": "gpt", "value": "Thought: I need to search for coffee makers.\nAction: search[coffee maker]"},
    {"from": "human", "value": "Observation: You see 5 results..."},
    ...
  ],
  "item_id": "webshop_5238",
  "loss": [0, 1, 1, ...]  // 哪些turn参与训练
}
```

### 5.2 数据集用途

1. **行为克隆(BC)**：直接监督学习
2. **AgentEvol进化**：作为初始种群
3. **RL初始策略**：预训练模型

---

## 6. AgentGym-RL：强化学习支持

### 6.1 新增功能（2025.09）

AgentGym-RL 扩展了原框架，支持多轮RL训练：

**特点**:
- 支持PPO、GRPO等RL算法
- 兼容现有EnvServer架构
- 支持轨迹级别的奖励分配
- 提供RL训练可视化

**架构**:
```
AgentGym-RL = AgentGym + RL Trainer

┌─────────────────────────────────────────┐
│           RL Trainer (新增)              │
│  ┌─────────────┐ ┌─────────────────┐   │
│  │  Trajectory │ │  Advantage      │   │
│  │  Collector  │ │  Computation    │   │
│  └──────┬──────┘ └────────┬────────┘   │
│         │                  │            │
│         └────────┬─────────┘            │
│                  ↓                      │
│           Policy Update                 │
└──────────────────┬──────────────────────┘
                   │ HTTP API
                   ↓
┌─────────────────────────────────────────┐
│        Existing EnvServers              │
└─────────────────────────────────────────┘
```

---

## 7. 技术栈与安装

### 7.1 技术栈

| 类型 | 技术 |
|------|------|
| 后端 | Python (60.8%) |
| 前端 | Vue.js (18.7%), JavaScript (12.7%) |
| 通信 | HTTP/RESTful API |
| 部署 | Docker, 多端口服务 |

### 7.2 安装方式

```bash
# 方式1: PyPI安装
pip install agentenv

# 方式2: 源码安装
git clone https://github.com/WooooDyy/AgentGym.git
cd AgentGym
pip install -e .

# 启动环境服务（每个环境）
python -m agentenv.webshop.server --port 3001
python -m agentenv.alfworld.server --port 3002
# ...
```

---

## 8. 总结

### 8.1 AgentGym vs verl-agent 核心区别

| 维度 | AgentGym | verl-agent |
|------|----------|------------|
| **架构** | 微服务，HTTP解耦 | 单体，紧耦合 |
| **环境** | 14+异构环境，即插即用 | 需自定义环境 |
| **数据** | 提供预训练轨迹 | 在线生成 |
| **用途** | 通用Agent研究平台 | RL训练框架 |
| **可视化** | 交互式前端 | 命令行 |

### 8.2 适用建议

- **用AgentGym**：研究通用Agent能力、需要多环境评估、需要预训练数据、快速原型开发
- **用verl-agent**：专注RL算法优化、需要大规模并行训练、特定任务的深度优化

### 8.3 独特价值

AgentGym 的独特价值在于其**"平台化"**设计：
1. **标准化**：统一的环境接口降低开发门槛
2. **可扩展**：微服务架构支持无限扩展
3. **数据驱动**：提供高质量轨迹数据加速研究
4. **全链路**：覆盖训练-评估-进化的完整流程

---

*整理日期: 2026-03-18*
*整理者: 王明达 (AI Agent)*
