# Kimi-K2.5 Technical Report

**论文**: Kimi-K2.5 Technical Report  
**arXiv**: https://arxiv.org/abs/2602.02276  
**本地文件**: `paper.pdf`

## 核心信息

### 模型规格
- **总参数量**: 1T
- **激活参数**: 32B
- **架构**: MoE (384 routed + 1 shared, Top-8)
- **稀疏比**: 31:1
- **上下文长度**: 256K
- **注意力**: MLA (Multi-head Latent Attention)
- **预训练数据**: ~15T 多模态 tokens

### 预训练阶段

| 阶段 | 数据 | 说明 |
|------|------|------|
| ViT Training | 视觉数据 | MoonViT-3D 原生处理 |
| Joint PT | ~15T MM | 文本+图像+视频联合训练 |
| Mid-training | 长文本扩展 | YaKN 渐进至 256K |

### 关键技术

#### 1. MoonViT-3D
- 原生 3D 视觉编码器
- 支持视频时序理解
- 4帧关键帧压缩表示

#### 2. Zero-Vision SFT
- 预训练足够强 → Post-training 可极简化视觉数据
- SFT 阶段主要使用纯文本数据
- 视觉能力通过预训练激活

#### 3. Joint MM RL
- 按能力(非模态)划分 domain
- GRM (Generalized Reward Model) 跨模态统一

### 数据配比
- 通用文本: ~50%
- 代码: ~20%
- STEM/科学: ~15%
- 多模态: ~15% (联合训练)

### Post-Training
- **SFT**: Zero-Vision (纯文本为主)
- **RL**: Joint MM RL，跨模态统一优化

## 对比定位

| 特性 | Kimi-K2.5 | Seed-VL | Qwen3-VL |
|------|-----------|---------|----------|
| 总参数量 | 1T (MoE) | — | — |
| 激活参数 | 32B | — | — |
| 上下文 | 256K | 128K | 256K |
| PT Tokens | ~15T | ~20T | 36T |
| 多模态占比 | ~15% | ~20% | ~10% |
| 训练策略 | 原生联合 | 视觉中心 | 分阶段 |

## 关键洞察

1. **原生联合训练**: 与后期拼接不同，K2.5 从预训练阶段就进行多模态联合
2. **Zero-Vision SFT**: 证明强预训练可以简化后训练的视觉数据需求
3. **MLA 注意力**: DeepSeek 的 MLA 被 Kimi 采用，降低推理成本
4. **4帧视频**: 时序压缩是关键，而非密集采样

## 待补充

- [ ] 详细训练超参数
- [ ] Benchmark 完整结果
- [ ] 消融实验分析
- [ ] 部署优化细节

---

*Added: 2026-04-11*
