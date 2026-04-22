# Post-Training 论文精读笔记索引

本文档汇总了 2025 年关于 LLM/VLM Post-Training 的关键论文精读笔记。

## 📁 目录结构

```
paper_reading/llm_vlm/mid-training/mid-train-sft/
├── pretrain-midtrain.html        # Pre-training Token Scaling 调研
├── mid-training-survey.html      # Mid-Training综述
├── midtraining-bridges.html      # Mid-training桥接Pre/Post-training
├── adept.html                    # ADEPT: 自适应扩展领域适配
├── domain-midtraining.html       # 领域Mid-training实践
├── sft-rl-relationship/          # SFT与RL关系研究
│   ├── sft-rl-saturation.html    # SFT & RL 饱和点分析
│   ├── long-cot-degradation.html # 小模型长CoT退化现象
│   └── mid-training-impact-on-post-training.html  # 预训练对后训练的影响
└── iterative-post-training/      # 迭代后训练方法
    ├── relift.html               # 动态SFT-RL切换
    ├── openvlthinker.html        # 迭代SFT-RL自我进化
    ├── rloop.html                # 迭代策略初始化
    ├── ton.html                  # 选择性推理
    ├── video-r1.html             # 视频推理强化 (T-GRPO)
    ├── r1-vl.html                # 步骤级密集奖励 (StepGRPO)
    └── mm-eureka.html            # RLOO + 在线过滤
```

## 📊 论文分类

### 一、Pre-training Token Scaling

| 文献/调研 | 范围 | 核心发现 | 文件 |
|------------|------|----------|------|
| **Chinchilla** (Hoffmann et al., 2022) | 2019-2025 | ~20 tokens/param计算最优 | [pretrain-midtrain.html](pretrain-midtrain.html) |
| **Kaplan vs Chinchilla** | Scaling Laws | 两大流派的统一 | [pretrain-midtrain.html](pretrain-midtrain.html) |
| **Inference-Aware** (2024+) | 推理成本 | 50-1000+:1比例更经济 | [pretrain-midtrain.html](pretrain-midtrain.html) |
| **Llama 3 / DeepSeek V3** | 工业实践 | 1875:1、FP8高效训练 | [pretrain-midtrain.html](pretrain-midtrain.html) |

### 二、Mid-Training 核心研究

| 论文 | 核心贡献 | 文件 |
|------|----------|------|
| **Mid-Training Survey** (arXiv:2510.06826) | Mid-training首篇综述 | [mid-training-survey.html](mid-training-survey.html) |
| **Midtraining Bridges** (ICLR 2025) | 桥接Pre/Post-training分布 | [midtraining-bridges.html](midtraining-bridges.html) |
| **ADEPT** (arXiv:2510.10071) | 自适应扩展+动态解耦调优 | [adept.html](adept.html) |
| **Domain Mid-training** | 医疗/法律/金融实践案例 | [domain-midtraining.html](domain-midtraining.html) |

### 三、SFT & RL 关系研究

| 论文 | 核心发现 | 文件 |
|------|----------|------|
| **SFT & RL Saturation** (arXiv:2506.16029) | ~80B tokens后饱和，>160B OOD退化 | [sft-rl-saturation.html](sft-rl-relationship/sft-rl-saturation.html) |
| **Long CoT Degradation** | ≤3B模型长CoT训练退化高达75% | [long-cot-degradation.html](sft-rl-relationship/long-cot-degradation.html) |
| **Mid-training Impact** | 预训练预算决定后训练上限 | [mid-training-impact-on-post-training.html](sft-rl-relationship/mid-training-impact-on-post-training.html) |

### 四、Iterative Post-Training

| 论文 | 核心方法 | 文件 |
|------|----------|------|
| **ReLIFT** (arXiv:2506.07527) | 困难问题动态切换SFT | [relift.html](iterative-post-training/relift.html) |
| **OpenVLThinker** (arXiv:2503.17352) | 迭代SFT-RL自我进化 | [openvlthinker.html](iterative-post-training/openvlthinker.html) |
| **RLoop** (arXiv:2511.04285) | 迭代策略初始化 | [rloop.html](iterative-post-training/rloop.html) |

### 五、VLM-Specific RL Methods

| 论文 | 核心方法 | 特点 | 文件 |
|------|----------|------|------|
| **TON** (NeurIPS 2025) | Thought Dropout + GRPO | 选择性推理，90% token减少 | [ton.html](iterative-post-training/ton.html) |
| **Video-R1** (arXiv:2503.21776) | T-GRPO | 时序对比奖励 | [video-r1.html](iterative-post-training/video-r1.html) |
| **R1-VL** (ICCV 2025) | StepGRPO | 步骤级密集奖励 | [r1-vl.html](iterative-post-training/r1-vl.html) |
| **MM-Eureka** (arXiv:2503.07365) | RLOO + Online Filtering | 训练稳定性 | [mm-eureka.html](iterative-post-training/mm-eureka.html) |

## 🔑 核心洞察汇总

### 1. 训练预算分配
- **预训练：** 80%预算，最佳停止点~80B tokens（80×模型大小）
- **SFT：** 10%预算，50K-100K样本
- **RL：** 10%预算，4-8 epochs

### 2. Mid-training价值
- **定位：** 介于Pre-training和Post-training之间的关键阶段
- **目的：** 领域知识注入，桥接通用与专门化分布
- **数据：** 数千亿tokens精选领域数据（70-80%领域 + 20-30%通用）
- **时机：** Pre-training完成约80%时开始效果最佳
- **学习率：** Pre-training peak LR的10-30%

### 3. SFT vs RL 关系
- SFT提供**先验/归纳偏置**
- RL负责**优化和突破**
- 两者**交替**优于**顺序**

### 4. 小模型特殊考虑
- ≤3B模型长CoT训练需>220K样本避免退化
- 密集奖励（StepGRPO）对小模型更重要
- 可考虑跳过SFT直接RL

### 5. VLM Post-Training 范式
- **GRPO** 是主流算法
- **规则奖励** 足够，无需PRM
- **任务专用奖励设计**（时序、选择性、步骤级）

### 6. Mid-training实践要点
- **功能专业化：** 不同层/单元重要性不同，应差异化处理（ADEPT）
- **领域适配：** 医疗/法律/金融需专门化Mid-training
- **高效适配：** 15%参数调优可达全参数 tuning效果
- **多阶段顺序：** Pre-training → Mid-training → Post-training

## 📚 相关论文引用

### Pre-training Scaling
- arXiv:2203.15556 - Chinchilla (Hoffmann et al., 2022)
- arXiv:2001.08361 - Kaplan Scaling Laws (Kaplan et al., 2020)
- arXiv:2406.12907 - Reconciling Kaplan and Chinchilla
- arXiv:2401.00448 - Beyond Chinchilla-Optimal (Inference-Aware)
- arXiv:2403.08540 - Over-training Reliability (Gadre et al.)
- arXiv:2501.18107 - Scaling Inference-Efficient LLMs
- Llama 3 Technical Report (Meta, 2024)
- DeepSeek-V3 Technical Report (DeepSeek, 2024)

### Mid-training Research
- arXiv:2510.06826 - Mid-Training of Large Language Models: A Survey
- OpenReview:u7L9FOgG7t - Midtraining Bridges Pretraining and Posttraining (ICLR 2025)
- arXiv:2510.10071 - ADEPT: Continual Pretraining via Adaptive Expansion
- arXiv:2311.08545 - Efficient Continual Pre-training for Domain Specific LLMs
- arXiv:2511.02451 - Merging Continual Pretraining Models for Domain-Specialized LLMs

### Post-training Methods

- arXiv:2506.16029 - Budget Allocation
- arXiv:2506.07712 - Long CoT Degradation
- arXiv:2506.07527 - ReLIFT
- arXiv:2503.17352 - OpenVLThinker
- arXiv:2511.04285 - RLoop
- arXiv:2505.16854 - TON (NeurIPS 2025)
- arXiv:2503.21776 - Video-R1
- arXiv:2503.12937 - R1-VL (ICCV 2025)
- arXiv:2503.07365 - MM-Eureka
