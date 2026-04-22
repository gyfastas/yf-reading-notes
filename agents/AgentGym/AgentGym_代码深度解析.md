# AgentGym 代码深度解析

**基于源码的全面架构分析**

---

## 1. 整体架构概览

### 1.1 三层架构设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AgentGym 架构分层                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Layer 3: 训练与评估层 (Trainer & Controller)                    │   │
│  │  ├── BCTrainer: 行为克隆训练                                     │   │
│  │  ├── AgentEvolTrainer: 进化训练                                  │   │
│  │  └── DistributedEvaluator: 分布式评估                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│                              │ HTTP/API                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Layer 2: 环境客户端层 (EnvClient) - agentenv/agentenv/envs/     │   │
│  │  ├── WebshopEnvClient, ALFWorldEnvClient, SciWorldEnvClient     │   │
│  │  ├── WeatherEnvClient, MovieEnvClient, TODOEnvClient            │   │
│  │  └── ... (14个环境的客户端实现)                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│                              │ HTTP (RESTful API)                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Layer 1: 环境服务器层 (EnvServer) - agentenv-*/                 │   │
│  │  ├── WebShop Server (FastAPI) - Port 3001                        │   │
│  │  ├── ALFWorld Server (FastAPI) - Port 3002                       │   │
│  │  ├── SciWorld Server (FastAPI) - Port 3003                       │   │
│  │  └── ... (每个环境独立进程/端口)                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心设计原则

| 设计原则 | 实现方式 | 优势 |
|---------|---------|------|
| **微服务架构** | 每个EnvServer独立部署 | 环境隔离、独立扩展 |
| **统一接口** | 标准HTTP RESTful API | 降低接入成本 |
| **语言无关** | HTTP通信 | EnvServer可用任意语言实现 |
| **客户端封装** | 统一EnvClient基类 | 训练代码与环境解耦 |

---

## 2. 环境服务器 (EnvServer) 实现深度分析

### 2.1 标准API接口规范

每个EnvServer必须实现以下RESTful端点：

| 端点 | 方法 | 功能 | 参数 | 响应 |
|------|------|------|------|------|
| `/` | GET | 健康检查 | - | `"ok"` |
| `/create` | POST | 创建环境实例 | - | `env_id: int` |
| `/step` | POST | 执行动作 | `env_idx`, `action` | `observation`, `reward`, `done` |
| `/reset` | POST | 重置环境 | `env_idx` | `observation` |
| `/observation` | GET | 获取观察 | `env_idx` | `observation: str` |
| `/list_envs` | GET | 列出所有环境 | - | `List[int]` |

### 2.2 三种服务器实现模式

#### 模式A: FastAPI + 单例EnvServer (WebShop, SQLGym)

```python
# agentenv-webshop/agentenv_webshop/server.py
from fastapi import FastAPI
from .environment import webshop_env_server  # 单例

app = FastAPI()

@app.post("/create", response_model=int)
async def create():
    """创建新环境"""
    env = webshop_env_server.create()  # 调用单例方法
    return env

@app.post("/step", response_model=StepResponse)
def step(step_query: StepQuery):
    state, reward, done, info = webshop_env_server.step(
        step_query.env_idx, step_query.action
    )
    return StepResponse(state=state, reward=reward, done=done, info=info)
```

**关键实现类** (`environment.py`):

```python
class WebshopEnvServer:
    def __init__(self) -> None:
        self._max_id = 0
        self.env = {}      # 存储所有环境实例
        self.ls = []       # 环境ID列表
        self.sz = 8000     # 最大环境数量限制

    def create(self) -> int:
        # 随机选择产品索引
        idx = random.randint(0, 48950076)

        # 使用gym.make创建原始环境
        self.env[idx] = gym.make(
            "WebAgentTextEnv-v0",
            observation_mode="text",
            num_products=1000,
        )
        self.env[idx].reset()
        return idx

    def step(self, env_idx, action: str):
        return self.env[env_idx].step(action)

# 全局单例
webshop_env_server = WebshopEnvServer()
```

#### 模式B: FastAPI + 配置驱动Wrapper (ALFWorld, SciWorld)

```python
# agentenv-alfworld/agentenv_alfworld/server.py
from fastapi import FastAPI
from .env_wrapper import server  # 预初始化的wrapper

app = FastAPI()

@app.post("/create")
async def create():
    return server.create()

@app.post("/reset")
async def reset(body: ResetRequestBody):
    # 需要指定game和world_type
    return server.reset(body.id, body.game, body.world_type)
```

**关键实现类** (`env_wrapper.py`):

```python
class ALFWorld_Wrapper:
    def __init__(self, **kwargs):
        # 加载数据路径
        self.data_path = kwargs.get("data_path", None)
        os.environ["ALFWORLD_DATA"] = self.data_path

        # 加载配置文件
        self.config_path = kwargs.get("config_path", None)
        self.config = load_config(self.config_path)

        # 加载游戏映射
        self.games = []
        self._load_game_mappings()

        # 线程锁保护
        self._lock = threading.Lock()

    def create(self):
        with self._lock:
            idx = self._max_id
            self._max_id += 1

        # 初始化TextWorld环境
        self.env[idx] = SingleAlfredTWEnv(self.config)
        self.info[idx] = {"done": False, "reward": 0, "deleted": False}
        return {"id": idx}

    def reset(self, idx: int, game: int, world_type: str):
        # 根据game索引选择特定任务
        self.env[idx].game_files = [self.games[game]]
        self.env[idx].num_games = 1
        self.env_init[idx] = self.env[idx].init_env(batch_size=1)

        ob, info = self.env_init[idx].reset()
        available_actions = info.get("admissible_commands", [[]])[0]

        return {
            "id": idx,
            "observation": ob,
            "available_actions": available_actions,
            "task_type": "/".join(info["extra.gamefile"][0].split("/")[-3:-1]),
        }

# 模块加载时预初始化
server = ALFWorld_Wrapper(
    data_path=os.environ["ALFWORLD_DATA"],
    config_path=".../base_config.yaml"
)
```

#### 模式C: Quart (Async Flask) + 异步处理 (WebArena)

```python
# agentenv-webarena/agentenv_webarena/server.py
from quart import Quart, jsonify, request
import asyncio

app = Quart(__name__)
_max_id = 0
_max_id_lock = asyncio.Lock()

@app.route("/create", methods=["POST"])
async def create():
    global _max_id
    async with _max_id_lock:  # 异步锁保护
        env_idx = _max_id
        _max_id += 1

    # 异步调用同步环境方法
    env = await asyncio.to_thread(webarena_env_server.create, env_idx)
    return jsonify({"env_idx": env})

@app.route("/step", methods=["POST"])
async def step():
    step_query = await request.get_json()
    step_data = await asyncio.to_thread(
        webarena_env_server.step,
        step_query["env_idx"],
        step_query["action"]
    )
    return jsonify({
        "observation": step_data[0],
        "reward": step_data[1],
        "terminated": step_data[2],
        "truncated": step_data[3],
        "info": step_data[4],
    })

# Cookie过期检查
@app.route("/reset", methods=["POST"])
async def reset():
    if check_cookies_expiration():
        # 自动重新登录
        subprocess.run(["python", "browser_env/auto_login.py"])

    obs, info, sites, object = await asyncio.to_thread(
        webarena_env_server.reset,
        reset_query["env_idx"],
        reset_query["seed"],
        reset_query["options"]
    )
```

### 2.3 环境实现对比分析

| 环境 | 服务器框架 | 原始环境 | 特殊处理 |
|------|-----------|---------|---------|
| **WebShop** | FastAPI | `gym.make("WebAgentTextEnv-v0")` | 产品搜索、点击动作 |
| **ALFWorld** | FastAPI | TextWorld | 任务类型映射、 admissible commands |
| **SciWorld** | FastAPI | ScienceWorld | 视觉模式、对象树 |
| **WebArena** | Quart | Playwright + 真实网站 | Cookie管理、自动登录 |
| **Weather** | FastAPI | 工具调用环境 | API调用解析、奖励计算 |
| **SQLGym** | FastAPI | BIRD-SQL | SQL执行验证 |
| **TextCraft** | FastAPI | TextCraft | Minecraft crafting逻辑 |

---

## 3. 环境客户端 (EnvClient) 实现分析

### 3.1 核心抽象类

```python
# agentenv/agentenv/controller/env.py
class BaseEnvClient(metaclass=ABCMeta):
    def __init__(self, action_format: ActionFormat = "react") -> None:
        self.action_format = ActionFormat(action_format)

    @abstractmethod
    def __len__(self) -> int:
        """返回环境总大小"""

    @abstractmethod
    def observe(self) -> str:
        """解析服务器响应，返回给LLM的文本"""

    @abstractmethod
    def step(self, action) -> StepOutput:
        """解析模型输出，调用环境服务器"""

    @abstractmethod
    def reset(self, idx: int) -> None:
        """重置环境"""
```

### 3.2 典型客户端实现 (WebShop)

```python
# agentenv/agentenv/envs/webshop.py
class WebshopEnvClient(BaseEnvClient):
    adapter_cls = WebshopAdapter

    def __init__(self, env_server_base: str, data_len: int,
                 *args, timeout: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        # 创建环境实例
        ok = requests.post(f"{self.env_server_base}/create",
                          timeout=self.timeout)
        self.env_id = ok.json()  # 保存环境ID

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """带重试的POST请求"""
        data["env_idx"] = self.env_id
        max_retries = 5
        for attempt in range(max_retries):
            res = requests.post(
                f"{self.env_server_base}/{path}",
                json=data,
                timeout=self.timeout,
            )
            if res.status_code == 503:
                time.sleep(0.1)  # 服务不可用，稍等重试
            elif res.status_code == 200:
                break
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        """GET请求"""
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        return res.json()

    def step(self, action: str) -> StepOutput:
        # 1. 使用Adapter解析模型输出
        action = WebshopAdapter.action_parser(action, self.action_format)

        # 2. 调用环境服务器
        response = self._post("step", {"action": action})

        return StepOutput(
            state=response["state"],
            reward=response["reward"],
            done=response["done"],
        )

    def observe(self) -> dict[str, Any]:
        return self._get("observation")

    def reset(self, idx: int) -> dict[str, Any]:
        return self._post("reset", {"session_id": idx})
```

### 3.3 Adapter模式解析

AgentGym支持3种Action Format，通过Adapter统一转换：

```python
class ActionFormat(Enum):
    REACT = "react"                    # Thought + Action
    FUNCTION_CALLING = "function_calling"  # OpenAI函数调用格式
    CODE_AS_ACTION = "code_as_action"      # Python代码格式
```

**WebShop的Adapter实现**:

```python
class WebshopAdapter(BaseAdapter):
    # 对话起始模板
    conversation_start_dict = {
        ActionFormat.REACT: (
            ConversationMessage({
                "from": "human",
                "value": "You are web shopping...\nYour response should use the following format:\n\nThought:\nI think ... \n\nAction: \nclick[something]"
            }),
            ConversationMessage({"from": "gpt", "value": "Ok."}),
        ),
        ActionFormat.FUNCTION_CALLING: (
            ConversationMessage({
                "from": "human",
                "value": f"You are web shopping...\n\n{format_function_call_prompt(WEBSHOP_FUNCTION_DESCRIPTION)}"
            }),
            ConversationMessage({"from": "gpt", "value": "Ok."}),
        ),
    }

    @staticmethod
    def parse_function_calling(text: str) -> ActionWithTought:
        """解析函数调用格式为环境动作"""
        _fn_call = json.loads("{" + text.split("{", 1)[-1].rsplit("}", 1)[0] + "}")
        thought = _fn_call["thought"]
        fn_name = _fn_call["function_name"]
        args = _fn_call["arguments"]

        if fn_name == "search":
            action = f"search[{args['keywords']}]"
        else:
            action = f"click[{args['item']}]"

        return ActionWithTought(thought=thought, action=action)

    @staticmethod
    def parse_code_as_action(text: str) -> ActionWithTought:
        """解析Python代码为环境动作"""
        def search(keywords: str):
            return f"search[{keywords}]"
        def click(item: str):
            return f"click[{item}]"

        text = extract_python_code_blocks(text)
        action = eval(text, {}, {"search": search, "click": click})
        thought = parse_python_code_comments(text)

        return ActionWithTought(thought=thought, action=action)
```

---

## 4. 训练与评估层深度分析

### 4.1 任务执行循环

```python
# agentenv/agentenv/controller/task.py
class BaseTask:
    def _generate_experience_one(
        self,
        agent: Agent | APIAgent,
        client: BaseEnvClient,
        idx: int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> ExperienceOutput:

        # 1. 重置环境
        client.reset(idx)
        reward = 0.0
        done = False
        state = client.observe()

        # 2. 构建初始对话
        conversation = list(client.conversation_start)
        conversation.append(ConversationMessage({
            "from": "human", "loss": None, "value": state
        }))

        rounds = 0
        while not done:
            # 3. 生成动作
            generated_tokens = agent.generate(
                [conversation_tokenized["input_ids"]], generation_config
            )[0]
            generated_text = tokenizer.decode(generated_tokens)

            # 4. 执行动作
            step_output = client.step(generated_text)
            state, reward, done = step_output.state, step_output.reward, step_output.done

            # 5. 更新对话历史
            conversation.append(ConversationMessage({
                "from": "gpt", "loss": True, "value": generated_text
            }))
            conversation.append(ConversationMessage({
                "from": "human", "loss": None, "value": state
            }))

            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break

        return ExperienceOutput(
            conversation=conversation,
            reward=reward,
            text=conversation_tokenized["text"],
            seq_ids=conversation_tokenized["input_ids"],
            action_mask=conversation_tokenized["action_mask"],
        )
```

### 4.2 行为克隆训练器 (BC Trainer)

```python
# agentenv/agentenv/trainer/bc_trainer.py
class BCTrainer(BaseTrainer):
    def __init__(self, agent: Agent, tasks: Sequence[BaseTask], args):
        self.agent = agent
        self.tasks = tasks
        self.args = asdict(args)

        # 初始化分布式训练
        self.create_accelerator()

        # 加载数据集
        self.get_raw_dataset()
        self.get_train_dataloader()

        # 初始化优化器
        self.init_train_stuff()

    def train(self):
        for epoch in range(self.args["num_train_epochs"]):
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.agent.model):
                    # 前向传播
                    outputs = self.agent.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )

                    # 计算loss (只计算action部分的loss)
                    loss = self.compute_loss(outputs, batch)

                    # 反向传播
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
```

---

## 5. 关键代码文件索引

### 5.1 核心框架代码

| 文件路径 | 功能说明 |
|---------|---------|
| `agentenv/agentenv/controller/env.py` | BaseEnvClient抽象类定义 |
| `agentenv/agentenv/controller/task.py` | BaseTask任务执行逻辑 |
| `agentenv/agentenv/controller/types.py` | 数据类型定义(ActionFormat, StepOutput等) |
| `agentenv/agentenv/trainer/bc_trainer.py` | 行为克隆训练器 |
| `agentenv/agentenv/trainer/agentevol_trainer.py` | AgentEvol进化训练器 |

### 5.2 环境服务器代码

| 环境 | Server文件 | Environment/Wrapper文件 |
|------|-----------|------------------------|
| WebShop | `agentenv-webshop/agentenv_webshop/server.py` | `environment.py` |
| ALFWorld | `agentenv-alfworld/agentenv_alfworld/server.py` | `env_wrapper.py` |
| SciWorld | `agentenv-sciworld/agentenv_sciworld/server.py` | `environment.py` |
| WebArena | `agentenv-webarena/agentenv_webarena/server.py` | `environment.py` |
| Weather | `agentenv-tool/agentenv_weather/weather_server.py` | `weather_environment.py` |
| TODOList | `agentenv-tool/agentenv_todo/todo_server.py` | `todo_environment.py` |
| SQLGym | `agentenv-sqlgym/agentenv_sqlgym/server.py` | `environment.py` |
| TextCraft | `agentenv-textcraft/agentenv_textcraft/server.py` | `env_wrapper.py` |

### 5.3 环境客户端代码

| 文件路径 | 说明 |
|---------|------|
| `agentenv/agentenv/envs/webshop.py` | WebShop客户端 + WebshopAdapter |
| `agentenv/agentenv/envs/alfworld.py` | ALFWorld客户端 + ALFWorldAdapter |
| `agentenv/agentenv/envs/sciworld.py` | SciWorld客户端 + SciWorldAdapter |
| `agentenv/agentenv/envs/weather.py` | Weather客户端 |
| `agentenv/agentenv/envs/babyai.py` | BabyAI客户端 |

---

## 6. 架构设计亮点

### 6.1 微服务解耦

```
优势:
1. 环境独立部署 - 不同环境可有不同依赖
2. 独立扩展 - 高负载环境可单独扩容
3. 故障隔离 - 单个环境崩溃不影响其他环境
4. 语言无关 - EnvServer可用任何语言实现
```

### 6.2 统一接口设计

```
所有环境共享相同API契约:
- /create → 返回 env_id
- /step → 返回 (observation, reward, done)
- /reset → 返回 observation
- /observation → 返回当前观察

这使得添加新环境只需实现这4个端点
```

### 6.3 Adapter模式的多格式支持

```
同一环境支持多种Agent交互格式:
- ReAct: Thought + Action (最常用)
- Function Calling: OpenAI风格函数调用
- Code as Action: Python代码执行

通过Adapter统一转换为环境动作
```

### 6.4 线程安全设计

```python
# ALFWorld的线程锁示例
class ALFWorld_Wrapper:
    def __init__(self, **kwargs):
        self._lock = threading.Lock()

    def create(self):
        with self._lock:  # 保护_max_id递增
            idx = self._max_id
            self._max_id += 1
```

---

## 7. 如何添加新环境

### 7.1 步骤概览

```
Step 1: 创建EnvServer (FastAPI/Flask)
    ├── 实现 /create, /step, /reset, /observation
    ├── 封装原始环境
    └── 运行在不同端口

Step 2: 创建EnvClient
    ├── 继承 BaseEnvClient
    ├── 实现 HTTP调用
    └── 实现 Adapter (可选)

Step 3: 创建Task
    ├── 继承 BaseTask
    ├── 指定 env_client_cls
    └── 配置 client_args

Step 4: 注册到配置文件
```

### 7.2 最小EnvServer示例

```python
from fastapi import FastAPI
import gym

app = FastAPI()
envs = {}
next_id = 0

@app.post("/create")
async def create():
    global next_id
    envs[next_id] = gym.make("MyEnv-v0")
    envs[next_id].reset()
    next_id += 1
    return next_id - 1

@app.post("/step")
async def step(data: dict):
    env_id = data["env_idx"]
    action = data["action"]
    obs, reward, done, info = envs[env_id].step(action)
    return {"observation": obs, "reward": reward, "done": done}

@app.get("/observation")
async def observation(env_idx: int):
    return envs[env_idx].get_obs()
```

---

*分析日期: 2026-03-18*
*基于 AgentGym GitHub 仓库代码*
