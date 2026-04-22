# Coding Agent Benchmarks 使用指南

**复杂代码生成与软件工程Agent评测基准大全**

---

## 目录

1. [概述](#1-概述)
2. [SWE-bench 使用指南](#2-swe-bench-使用指南)
3. [HumanEval+ / EvalPlus 使用指南](#3-humaneval--evalplus-使用指南)
4. [AgentCoder / InterCode 使用指南](#4-agentcoder--intercode-使用指南)
5. [CodeFuse-Eval 使用指南](#5-codefuse-eval-使用指南)
6. [其他Benchmarks](#6-其他benchmarks)
7. [快速选择指南](#7-快速选择指南)

---

## 1. 概述

### 1.1 Benchmark复杂度金字塔

```
                    ┌─────────────────┐
                    │   SWE-bench     │  ← 最复杂：真实GitHub Issue
                    │  (真实软件工程)  │     多文件修改、测试验证
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  AgentCoder/    │  ← 中等复杂：多轮交互
                    │  InterCode      │     代码执行反馈
                    └────────┬────────┘
                             │
            ┌────────────────▼────────────────┐
            │      HumanEval+ / MBPP+         │  ← 基础：单函数生成
            │   (代码生成 + 增强测试覆盖)       │
            └────────────────┬────────────────┘
                             │
        ┌────────────────────▼────────────────────┐
        │           DS-1000 / CoderEval           │  ← 入门级：数据科学
        │                                          │     简单代码补全
        └─────────────────────────────────────────┘
```

### 1.2 对比总览

| Benchmark | 类型 | 实例数 | 难度 | 公开状态 | 核心特点 |
|-----------|------|--------|------|----------|----------|
| **SWE-bench** | 软件工程 | 2,294 | ⭐⭐⭐⭐⭐ | ✅ 完全公开 | 真实GitHub Issue |
| **SWE-bench Lite** | 软件工程 | 300 | ⭐⭐⭐⭐ | ✅ 完全公开 | SWE-bench子集 |
| **HumanEval+** | 代码生成 | 164 | ⭐⭐ | ✅ 完全公开 | 增强测试覆盖 |
| **MBPP+** | 代码生成 | 427 | ⭐⭐ | ✅ 完全公开 | 更多测试用例 |
| **AgentCoder** | 交互式编码 | 数千 | ⭐⭐⭐⭐ | ✅ 完全公开 | 多轮代码生成 |
| **InterCode** | 交互式编码 | 500+ | ⭐⭐⭐ | ✅ 完全公开 | 真实执行环境 |
| **CodeFuse-Eval** | 综合评测 | 多任务 | ⭐⭐⭐ | ✅ 完全公开 | 中文支持 |
| **DS-1000** | 数据科学 | 1,000 | ⭐⭐⭐ | ✅ 完全公开 | 数据科学代码 |

---

## 2. SWE-bench 使用指南

### 2.1 项目信息

- **官方网站**: https://www.swebench.com/
- **GitHub**: https://github.com/princeton-nlp/SWE-bench
- **论文**: [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)
- **数据集**: Hugging Face `princeton-nlp/SWE-bench`

### 2.2 安装

```bash
# 克隆仓库
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench

# 安装依赖
pip install -e .

# 或使用pip安装
pip install swebench
```

### 2.3 数据集加载

```python
from datasets import load_dataset

# 加载完整数据集
swebench = load_dataset("princeton-nlp/SWE-bench", "default")

# 加载Lite版本（推荐入门使用）
swebench_lite = load_dataset("princeton-nlp/SWE-bench", "lite")

# 查看数据格式
example = swebench['test'][0]
print(f"Instance ID: {example['instance_id']}")
print(f"Repo: {example['repo']}")
print(f"Issue: {example['problem_statement'][:200]}...")
```

**数据格式说明**:
```python
{
    "instance_id": "django__django-11011",  # 唯一标识
    "repo": "django/django",                # 仓库名
    "problem_statement": "...",             # Issue描述
    "hint_text": "...",                     # 可选提示
    "created_at": "2023-01-01T00:00:00Z",   # 创建时间
    "version": "4.0",                       # 受影响版本
    "base_commit": "abc123",                # 基线commit
    "patch": "...",                         # 正确修复patch
    "test_patch": "..."                     # 测试代码patch
}
```

### 2.4 运行评估

#### 方式1: 使用官方CLI工具

```bash
# 准备预测结果文件 (predictions.jsonl)
# 格式: {"instance_id": "...", "model_patch": "...", "model_name_or_path": "..."}

# 运行评估
python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench \
    --predictions_path predictions.jsonl \
    --max_workers 4 \
    --cache_dir /path/to/cache
```

#### 方式2: 使用Docker（推荐）

```bash
# 1. 构建评估镜像
cd swebench/harness/docker
docker build -t swebench-eval .

# 2. 准备预测文件 predictions.jsonl

# 3. 运行评估容器
docker run -v $(pwd)/predictions.jsonl:/predictions.jsonl \
    -v /var/run/docker.sock:/var/run/docker.sock \
    swebench-eval \
    --predictions_path /predictions.jsonl \
    --max_workers 4
```

#### 方式3: Python API

```python
from swebench.harness.run_evaluation import run_evaluation

# 准备预测结果
predictions = [
    {
        "instance_id": "django__django-11011",
        "model_patch": "diff --git a/file.py b/file.py...",
        "model_name_or_path": "my-agent"
    },
    # ...
]

# 保存预测
import json
with open("predictions.jsonl", "w") as f:
    for p in predictions:
        f.write(json.dumps(p) + "\n")

# 运行评估
run_evaluation(
    dataset_name="princeton-nlp/SWE-bench",
    predictions_path="predictions.jsonl",
    max_workers=4,
    cache_dir="./cache"
)
```

### 2.5 生成预测结果

```python
import json
from datasets import load_dataset

# 加载数据集
swebench = load_dataset("princeton-nlp/SWE-bench", "lite", split="test")

# 你的Agent推理逻辑
def solve_issue(instance):
    """你的Agent解决issue的逻辑"""
    issue_description = instance['problem_statement']
    # ... 你的Agent代码 ...
    patch = generate_patch(issue_description)
    return patch

# 生成预测
predictions = []
for instance in swebench:
    patch = solve_issue(instance)
    predictions.append({
        "instance_id": instance["instance_id"],
        "model_patch": patch,
        "model_name_or_path": "my-agent"
    })

# 保存
with open("predictions.jsonl", "w") as f:
    for p in predictions:
        f.write(json.dumps(p) + "\n")
```

### 2.6 评估指标解读

```python
# 评估结果会生成在指定目录
# 主要指标:
# - resolved: 是否成功修复（通过所有测试）
# - test_results: 测试执行结果
# - build_results: 环境构建结果

{
    "instance_id": "django__django-11011",
    "resolved": True,  # 成功修复
    "test_results": {
        "passed": 5,
        "failed": 0,
        "skipped": 0
    },
    "build_success": True,
    "elapsed_time": 120.5
}
```

### 2.7 本地调试单个实例

```python
from swebench.harness.utils import clone_repo, checkout_commit
import subprocess

# 1. 获取实例
instance = swebench['test'][0]
instance_id = instance['instance_id']
repo_name = instance['repo']
base_commit = instance['base_commit']

# 2. 克隆仓库
repo_dir = f"/tmp/{repo_name.replace('/', '_')}"
clone_repo(f"https://github.com/{repo_name}.git", repo_dir)

# 3. 切换到基线commit
checkout_commit(repo_dir, base_commit)

# 4. 应用你的修复
with open(f"{repo_dir}/fix.patch", "w") as f:
    f.write(your_patch)

subprocess.run(["git", "apply", "fix.patch"], cwd=repo_dir)

# 5. 运行测试
test_cmd = instance.get('test_cmd', 'pytest')
subprocess.run(test_cmd.split(), cwd=repo_dir)
```

---

## 3. HumanEval+ / EvalPlus 使用指南

### 3.1 项目信息

- **官方网站**: https://evalplus.github.io/
- **GitHub**: https://github.com/evalplus/evalplus
- **论文**: [arXiv:2305.01210](https://arxiv.org/abs/2305.01210)
- **特点**: 80倍更多测试用例，防止overfitting

### 3.2 安装

```bash
pip install evalplus
```

### 3.3 快速开始

#### 评估HumanEval+

```bash
# 方式1: 使用CLI
evalplus.evaluate \
    --dataset humaneval \
    --samples samples.jsonl \
    --backend native  # 或 docker

# 方式2: 带额外测试的严格模式
evalplus.evaluate \
    --dataset humaneval \
    --samples samples.jsonl \
    --base-only  # 仅使用原始测试
    --test-details  # 显示详细测试结果
```

#### 评估MBPP+

```bash
evalplus.evaluate \
    --dataset mbpp \
    --samples mbpp_samples.jsonl \
    --backend native
```

### 3.4 Python API使用

```python
from evalplus.evaluate import evaluate

# 准备样本
# samples.jsonl 格式:
# {"task_id": "HumanEval/0", "solution": "def has_close_elements(...): ..."}

results = evaluate(
    dataset="humaneval",  # 或 "mbpp"
    samples="samples.jsonl",
    backend="native",     # "native" 或 "docker"
    base_only=False,      # True=仅原始测试, False=使用增强测试
    parallel=4,           # 并行数
    test_details=True     # 返回详细结果
)

print(f"Pass@1: {results['pass@1']}")
print(f"Pass@10: {results['pass@10']}")
```

### 3.5 生成代码样本

```python
from evalplus.data import get_human_eval_plus, get_mbpp_plus

# 加载数据集
problems = get_human_eval_plus()
# 或
problems = get_mbpp_plus()

# problems格式:
# {
#     "HumanEval/0": {
#         "task_id": "HumanEval/0",
#         "prompt": "def has_close_elements(numbers: List[float], threshold: float): ...",
#         "entry_point": "has_close_elements",
#         "canonical_solution": "...",
#         "test": "...",
#         "plus_test": "..."  # 额外的测试
#     }
# }

# 你的代码生成模型
def generate_code(prompt):
    # ... 调用你的模型 ...
    return generated_code

# 生成样本
import json

samples = []
for task_id, problem in problems.items():
    code = generate_code(problem['prompt'])
    samples.append({
        "task_id": task_id,
        "solution": code
    })

with open("samples.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")
```

### 3.6 查看增强测试

```python
from evalplus.data import get_human_eval_plus

problems = get_human_eval_plus()
problem = problems["HumanEval/0"]

# 原始测试
print("原始测试数量:", len(problem['test']))

# EvalPlus增强测试
print("增强测试数量:", len(problem['plus_test']))

# 增强测试的特点:
# - 更多边界情况
# - 更大输入规模
# - 防止硬编码
```

---

## 4. AgentCoder / InterCode 使用指南

### 4.1 InterCode

**项目信息**:
- **GitHub**: https://github.com/princeton-nlp/intercode
- **论文**: [arXiv:2306.14898](https://arxiv.org/abs/2306.14898)
- **特点**: 真实执行环境，多轮交互

#### 安装

```bash
git clone https://github.com/princeton-nlp/intercode.git
cd intercode
pip install -e .
```

#### 基本使用

```python
from intercode.envs import BashEnv, PythonEnv, SQLExecEnv

# Bash环境
env = BashEnv(
    image="intercode-bash",  # Docker镜像
    traj_dir="./traj",       # 轨迹保存目录
    verbose=True
)

# 重置环境
observation = env.reset()
print(observation)

# 执行命令
action = "ls -la"
observation, reward, done, info = env.step(action)
print(f"Observation: {observation}")
print(f"Reward: {reward}")

# 关闭环境
env.close()
```

#### Python执行环境

```python
from intercode.envs import PythonEnv

env = PythonEnv(
    image="intercode-python",
    traj_dir="./traj"
)

observation = env.reset()

# 执行Python代码
action = """
import numpy as np
arr = np.array([1, 2, 3])
print(arr.mean())
"""
observation, reward, done, info = env.step(action)
```

#### SQL执行环境

```python
from intercode.envs import SQLExecEnv

env = SQLExecEnv(
    data_path="./database.sqlite",  # 数据库文件
    traj_dir="./traj"
)

observation = env.reset()

# 执行SQL
action = "SELECT * FROM users WHERE age > 25;"
observation, reward, done, info = env.step(action)
```

### 4.2 AgentCoder

**项目信息**:
- **GitHub**: https://github.com/NL2Code/AgentCoder
- **论文**: [arXiv:2312.02033](https://arxiv.org/abs/2312.02033)
- **特点**: 多轮代码生成，测试反馈迭代

#### 安装

```bash
git clone https://github.com/NL2Code/AgentCoder.git
cd AgentCoder
pip install -r requirements.txt
```

#### 运行AgentCoder

```python
from agentcoder import AgentCoder

# 初始化
agent = AgentCoder(
    model="gpt-4",  # 或 "gpt-3.5-turbo", "claude-3"等
    max_iterations=5,
    temperature=0.7
)

# 问题描述
problem = """
Write a function to find the longest substring without repeating characters.
Example:
Input: "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
"""

# 生成代码
code, iterations = agent.solve(problem)

print(f"Generated code:\n{code}")
print(f"Iterations: {iterations}")
```

#### 评估流程

```python
from agentcoder.evaluation import evaluate_on_humaneval

results = evaluate_on_humaneval(
    agent=agent,
    dataset_path="HumanEval.jsonl",
    output_path="results.jsonl",
    max_workers=4
)

print(f"Pass@1: {results['pass@1']}")
```

---

## 5. CodeFuse-Eval 使用指南

### 5.1 项目信息

- **GitHub**: https://github.com/codefuse-ai/codefuse-eval
- **组织**: 蚂蚁集团 CodeFuse
- **特点**: 中文支持，多语言，多任务

### 5.2 安装

```bash
git clone https://github.com/codefuse-ai/codefuse-eval.git
cd codefuse-eval
pip install -e .
```

### 5.3 数据集

CodeFuse-Eval 包含多个任务：

| 任务 | 说明 | 语言 |
|------|------|------|
| 代码补全 | Line/Block级别补全 | Python, Java, JS等 |
| 代码生成 | 自然语言到代码 | 多语言 |
| 代码翻译 | 跨语言翻译 | 多语言 |
| 代码摘要 | 生成文档字符串 | 多语言 |
| 测试生成 | 生成单元测试 | 多语言 |

### 5.4 使用示例

```python
from codefuse_eval import load_dataset, evaluate

# 加载数据集
dataset = load_dataset("codefuse-ai/codefuse-eval", "code-completion")

# 查看数据格式
example = dataset['test'][0]
print(f"Prefix: {example['prefix']}")
print(f"Suffix: {example['suffix']}")
print(f"Ground Truth: {example['ground_truth']}")

# 你的模型推理
def complete_code(prefix, suffix):
    # ... 调用你的模型 ...
    return generated_code

# 生成预测
predictions = []
for example in dataset['test']:
    pred = complete_code(example['prefix'], example['suffix'])
    predictions.append({
        "task_id": example['task_id'],
        "prediction": pred
    })

# 评估
results = evaluate(
    dataset=dataset,
    predictions=predictions,
    metric="em"  # Exact Match 或 bleu等
)

print(f"Accuracy: {results['accuracy']}")
```

### 5.5 多语言支持

```python
from codefuse_eval import MultiLanguageEvaluator

evaluator = MultiLanguageEvaluator(languages=["python", "java", "javascript"])

for lang in ["python", "java", "javascript"]:
    dataset = load_dataset("codefuse-ai/codefuse-eval", f"code-generation-{lang}")
    results = evaluator.evaluate(dataset, your_predictions)
    print(f"{lang}: {results['pass@1']}")
```

---

## 6. 其他Benchmarks

### 6.1 DS-1000 (数据科学代码)

**项目**: https://github.com/HKUNLP/DS-1000

```bash
pip install ds1000

# 使用
from ds1000 import DS1000Dataset
ds1000 = DS1000Dataset()

# 包含7个库: Numpy, Pandas, TensorFlow, PyTorch, SciPy, Scikit-learn, Matplotlib
```

### 6.2 CoderEval

**项目**: https://github.com/AMiner/CoderEval

```bash
git clone https://github.com/AMiner/CoderEval.git

# 包含真实项目中的代码生成任务
```

### 6.3 MultiPL-E

**项目**: https://github.com/nuprl/MultiPL-E

- 支持18种编程语言的HumanEval翻译版本

```bash
pip install multipl-e

# 评估多语言代码生成
```

### 6.4 LiveCodeBench

**项目**: https://livecodebench.github.io/

- 包含时间敏感的问题，防止数据污染

```bash
pip install livecodebench
```

---

## 7. 快速选择指南

### 7.1 根据研究目标选择

| 研究目标 | 推荐Benchmark | 原因 |
|----------|---------------|------|
| **软件工程Agent** | SWE-bench Lite | 真实GitHub Issue，端到端评估 |
| **代码生成基础能力** | HumanEval+ / MBPP+ | 标准基准，测试覆盖强 |
| **多轮交互编程** | AgentCoder / InterCode | 支持代码执行反馈 |
| **中文场景** | CodeFuse-Eval | 中文支持，多任务 |
| **数据科学** | DS-1000 | 数据科学专用任务 |
| **多语言支持** | MultiPL-E | 18种语言 |
| **防止数据污染** | LiveCodeBench | 时间敏感问题 |

### 7.2 根据资源选择

| 资源情况 | 推荐方案 |
|----------|----------|
| **计算资源充足** | SWE-bench Full + HumanEval+ + AgentCoder |
| **计算资源有限** | SWE-bench Lite (300实例) |
| **快速验证** | HumanEval+ (164实例) |
| **无Docker环境** | HumanEval+ (native backend) |

### 7.3 完整评估Pipeline示例

```python
# 综合评估脚本示例
import json
from datasets import load_dataset
from evalplus.evaluate import evaluate as evalplus_eval
from swebench.harness.run_evaluation import run_evaluation

class ComprehensiveCodeEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate_humaneval_plus(self, model):
        """评估HumanEval+"""
        # 生成代码
        samples = self.generate_humaneval_samples(model)
        with open("humaneval_samples.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        # 评估
        results = evalplus_eval(
            dataset="humaneval",
            samples="humaneval_samples.jsonl",
            backend="native"
        )
        self.results['humaneval_plus'] = results
        return results

    def evaluate_swe_bench_lite(self, model):
        """评估SWE-bench Lite"""
        # 加载数据
        swebench = load_dataset("princeton-nlp/SWE-bench", "lite", split="test")

        # 生成修复
        predictions = []
        for instance in swebench:
            patch = model.fix_issue(instance)
            predictions.append({
                "instance_id": instance["instance_id"],
                "model_patch": patch,
                "model_name_or_path": model.name
            })

        with open("swebench_predictions.jsonl", "w") as f:
            for p in predictions:
                f.write(json.dumps(p) + "\n")

        # 评估（需要Docker）
        run_evaluation(
            dataset_name="princeton-nlp/SWE-bench",
            predictions_path="swebench_predictions.jsonl",
            max_workers=4
        )

    def generate_report(self):
        """生成综合报告"""
        report = f"""
# Code Agent Evaluation Report

## HumanEval+ Results
- Pass@1: {self.results.get('humaneval_plus', {}).get('pass@1', 'N/A')}
- Pass@10: {self.results.get('humaneval_plus', {}).get('pass@10', 'N/A')}

## Summary
- Overall Score: {self.compute_overall_score()}
        """
        return report

# 使用示例
evaluator = ComprehensiveCodeEvaluator()
results = evaluator.evaluate_humaneval_plus(your_model)
print(evaluator.generate_report())
```

### 7.4 资源链接汇总

| Benchmark | GitHub | 文档 | HuggingFace |
|-----------|--------|------|-------------|
| SWE-bench | [princeton-nlp/SWE-bench](https://github.com/princeton-nlp/SWE-bench) | [swebench.com](https://www.swebench.com/) | `princeton-nlp/SWE-bench` |
| EvalPlus | [evalplus/evalplus](https://github.com/evalplus/evalplus) | [evalplus.github.io](https://evalplus.github.io/) | - |
| InterCode | [princeton-nlp/intercode](https://github.com/princeton-nlp/intercode) | - | - |
| AgentCoder | [NL2Code/AgentCoder](https://github.com/NL2Code/AgentCoder) | - | - |
| CodeFuse-Eval | [codefuse-ai/codefuse-eval](https://github.com/codefuse-ai/codefuse-eval) | - | `codefuse-ai/codefuse-eval` |
| DS-1000 | [HKUNLP/DS-1000](https://github.com/HKUNLP/DS-1000) | - | - |

---

*整理日期: 2026-03-18*
*基于各Benchmark官方文档*
