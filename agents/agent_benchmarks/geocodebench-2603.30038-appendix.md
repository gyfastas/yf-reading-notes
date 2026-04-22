# GeoCodeBench 附录：具体代码示例与详细 Benchmark 结果

> 配合主精读笔记 `geocodebench-2603.30038.md` 使用  
> 本文聚焦：① 真实题目长什么样 ② 模型表现的具体数字

---

## 一、真实题目示例

GeoCodeBench 的每一道题都遵循统一格式：**结构化论文文本** + **带挖空的代码骨架** + **执行模板** + **单元测试**。

以下均来自论文原文展示的代表性案例。

---

### 示例 1：points_in_finite_cone（圆锥内点判定）

**来源**：3D Gaussian Splatting 相关论文  
**类别**：Geometric Transformations（General 3D Capability）

**题目描述**：
> Q: Please read the paper and consider what to fill in the `****EMPTY****` part of the function `points_in_finite_cone()`.

**代码骨架**：
```python
def points_in_finite_cone(points, apex, direction, angle_cosine, height):
    # Vector from apex to each point
    apex_to_points = points - apex
    
    # Calculate distances from apex to points
    distances = torch.norm(apex_to_points, dim=1)
    
    # Normalize apex_to_points vectors
    normalized_vectors = apex_to_points / distances.unsqueeze(1)
    
    # Calculate cosine of angle between cone direction and apex-to-point vectors
    cos_angles = torch.sum(normalized_vectors * direction, dim=1)
    
    # Check angular constraint: angle must be within cone half angle
    mask_angle = cos_angles >= angle_cosine
    
    # Check height constraint: projection onto direction must be within [0, height]
    projections = torch.sum(apex_to_points * direction, dim=1)
    mask_height = (projections >= 0) & (projections <= height)
    
    return mask_angle & mask_height
```

**模型表现**：
- GPT-5：✅ 10/10 单元测试通过
- Kimi-K2-Instruct：✅ 10/10 单元测试通过

**关键观察**：两个模型生成的代码**数学等价但路径不同**：
- 官方实现和 GPT-5 类似：先归一化向量，再算夹角
- Kimi 的实现：用投影长度 $t = v \cdot direction$ 直接算 $\cos\theta = t / (||v|| + \epsilon)$，避免了显式归一化

这正体现了论文提出的 **Creative Correctness**。

---

### 示例 2：compute_epipolar_distance（对极距离计算）

**来源**：多视图几何论文  
**类别**：Geometric Transformations（General 3D Capability）

**题目描述**：
> Q: Please read the paper and consider what to fill in the `****EMPTY****` part of the function `compute_epipolar_distance()`.

**代码骨架**：
```python
def compute_epipolar_distance(T_21, K, p_1, p_2):
    # ****EMPTY****
    return geometric_e_distance
```

**GPT-5 的实现**（Fundamental Matrix 路径，像素坐标系）：
```python
# Essential matrix: E = [t]_x R
E = skew(t_21).dot(R_21)
# Fundamental matrix: F = K^{-T} E K^{-1}
K_inv = np.linalg.inv(K)
F = K_inv.T @ E @ K_inv

# Epipolar lines in both images
l2 = F @ p_1  # shape (3, N)
l1 = F.T @ p_2  # shape (3, N)

# Numerators: s = p2^T F p1
s = np.sum(p_2 * l2, axis=0)

# Denominators
eps = 1e-12
denom2 = np.sqrt(l2[0, :]**2 + l2[1, :]**2) + eps
denom1 = np.sqrt(l1[0, :]**2 + l1[1, :]**2) + eps

# ... compute symmetric epipolar distance
return geometric_e_distance
```

**DeepSeek-R1 的实现**（Essential Matrix 路径，归一化坐标系）：
```python
# Compute essential matrix
E = skew(t) @ R
# Transform to normalized coordinates
K_inv = np.linalg.inv(K)
x1 = K_inv @ p_1  # 3xN
x2 = K_inv @ p_2  # 3xN

# Compute s = x2^T * E * x1 for all points
s = np.sum(x2 * (E @ x1), axis=0)
abs_s = np.abs(s)

# Compute epipolar lines (in normalized coords, use E directly)
l2 = E @ x1  # lines in image 2: 3xN
l1 = E.T @ x2  # lines in image 1: 3xN

# Denominators
denom1 = np.sqrt(l1[0, :]**2 + l1[1, :]**2)
denom2 = np.sqrt(l2[0, :]**2 + l2[1, :]**2)

# ... compute symmetric epipolar distance
return geometric_e_distance
```

**结果**：两者都 ✅ 10/10 通过。

**数学等价性**：$F = K^{-T} E K^{-1}$，因此两种路径本质相同，只是坐标系选择不同。这说明模型真正理解了几何关系，而非背诵答案。

---

### 示例 3：forward_event（事件相机事件累积）—— 全模型阵亡

**来源**：IncEventGS [23]（CVPR 2025）  
**类别**：Novel Algorithm Implementation（Research Capability）

**题目描述**：
> 实现事件相机的事件累积近似，使用两个事件相机帧的对数强度差。

**代码骨架**：
```python
def forward_event(self, rays_o, rays_d, rgb_rendered):
    # ****EMPTY****
    return event_change
```

**参考实现**（来自官方代码库）：
```python
event_change = (log_brightness - ref_log_brightness) / C
```

**各模型的错误**：

| 模型 | 错误类型 | 具体表现 |
|------|---------|---------|
| GPT-5 | ❌ Improper Formula | 公式使用不当，算出了错误的对数差 |
| Gemini-2.5-Pro | ❌ Invalid Input | 输入参数处理错误，传了不该传的参数 |
| Llama-3.1-405B | ❌ Fictional Function | 虚构了一个不存在的 `render_rays` 函数，里面直接 `torch.rand` 生成随机RGB |

**为什么这个函数特别难？**
- 代码本身很短（几行）
- 但需要理解**事件相机的物理成像原理**（对数强度、时间差、对比度阈值）
- 论文中这个公式可能散落在方法描述中，没有显式给出完整代码
- 模型需要把物理概念正确映射到代码实现

**启示**：Research Capability 的瓶颈不在于"写长代码"，而在于**把论文中隐式描述的物理/数学原理转化为精确的代码表达式**。

---

### 示例 4：cross_warp_with_pose_depth_candidates（跨视角warp）

**来源**：全景Omnidirectional Gaussian Splatting 论文 [10]  
**类别**：Geometric Logic Routing（Research Capability）

**题目描述**：
> 实现跨视角特征warp，需要同时处理 Yin 和 Yang 两种全景投影的相互投影关系。

**DeepSeek-R1 的失败案例**（图6右）：

模型正确理解了 `yin_to_3d`（基础几何函数，图6左）——把2D图像坐标映射到3D球坐标：
```python
def yin_to_3d(grid, h, w):
    # Convert 2D grid to spherical (theta, phi) then to Cartesian
    x = torch.cos(theta) * torch.sin(phi)
    y = -torch.sin(theta)
    z = torch.cos(theta) * torch.cos(phi)
```

但在 `cross_warp_with_pose_depth_candidates` 中，模型**把相互投影（mutual projection）误解为独立投影**：

```python
# 模型写的（错误）：分别处理，没有交叉
yin_proj = yin_from_3d(yin_3d_transformed.reshape(b*d*h*w, 3), h, w)
yang90_proj = yang90_from_3d(yang90_3d_transformed.reshape(b*d*h*w, 3), h, w)

# 参考实现（正确）：需要交叉投影
points_to_yin_from_yin = yin_from_3d(points_from_yin, h, w)
points_to_yang_from_yin = yang90_from_3d(points_from_yin, h, w)   # 关键：Yin点投到Yang
points_to_yin_from_yang = yin_from_3d(points_from_yang, h, w)     # 关键：Yang点投到Yin
points_to_yang_from_yang = yang90_from_3d(points_from_yang, h, w)
```

**核心错误**：模型没能理解"cross-warp"的语义——需要把 Yin 空间的点投到 Yang 空间，反之亦然。它只做了"各自的投影"。

---

### 示例 5：init_bin_centers（初始化深度bin中心）

**来源**：稀疏视角重建论文 [11]  
**类别**：Novel Algorithm Implementation（Research Capability）

**代码骨架**：
```python
def init_bin_centers(self):
    # Function to Implement
    ****EMPTY****
    return bin_centers
```

这是一个**论文特定算法**的初始化函数，没有通用几何知识可以依赖，必须仔细阅读论文的 Method 部分才能理解 bin 的划分策略。

---

## 二、Benchmark 结果详表

### 2.1 总体排名与细粒度分解

| 排名 | 模型 | 公司 | 总体 | General | Research | Geo.Trans | Mech./Opt. | Algorithm | Routing |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **GPT-5** | OpenAI | **36.6%** | **42.8%** | 29.1% | **41.7%** | **43.7%** | **29.1%** | 28.9% |
| 2 | Claude-Sonnet-4.5 | Anthropic | 31.1% | 37.2% | 23.7% | 38.3% | 36.5% | 19.7% | **35.9%** |
| 3 | Gemini-2.5-Pro | Google | 30.4% | 33.8% | 26.2% | 41.9% | 27.6% | 25.3% | 29.1% |
| 4 | Kimi-K2-Instruct | Moonshot | 30.4% | 34.6% | 25.1% | 36.7% | 33.1% | 23.1% | 31.4% |
| 5 | Doubao-Seed-1.6 | ByteDance | 26.9% | 29.7% | 23.4% | 40.9% | 21.0% | 22.9% | 25.2% |
| 6 | Qwen3-Coder-480B | Alibaba | 23.5% | 22.7% | **24.6%** | 29.0% | 17.7% | 21.8% | 33.2% |
| 7 | DeepSeek-R1 | DeepSeek | 21.0% | 27.2% | 13.5% | 33.9% | 21.9% | 12.4% | 17.0% |
| 8 | Llama-3.1-405B-Inst. | Meta | 14.3% | 16.8% | 11.3% | 21.3% | 13.2% | 10.9% | 12.7% |

**关键观察**：
- **GPT-5 全面领先**，尤其在 Mechanics/Optics（43.7%）和 Algorithm（29.1%）上优势明显
- **Claude-Sonnet-4.5 的 Routing 能力最强**（35.9%），但 Algorithm 很弱（19.7%）
- **Qwen3-Coder-480B 是唯一 Research > General 的模型**，说明代码专精训练有所帮助，但基础几何薄弱
- **DeepSeek-R1 的 Research 能力断崖式下跌**（27.2% → 13.5%），推理模型在科研实现上没有展现出相对优势

---

### 2.2 General vs Research Capability 逐模型对比

| 模型 | General 3D | Research | 差距 (General - Research) |
|:---|:---:|:---:|:---:|
| GPT-5 | 42.8% | 29.1% | **+13.7%** |
| Claude-Sonnet-4.5 | 37.2% | 23.7% | **+13.5%** |
| Gemini-2.5-Pro | 33.8% | 26.2% | +7.6% |
| Kimi-K2-Instruct | 34.6% | 25.1% | +9.5% |
| Doubao-Seed-1.6 | 29.7% | 23.4% | +6.3% |
| Qwen3-Coder-480B | 22.7% | 24.6% | **-1.9%** |
| DeepSeek-R1 | 27.2% | 13.5% | **+13.7%** |
| Llama-3.1-405B | 16.8% | 11.3% | +5.5% |

**解读**：
- 除 Qwen3-Coder 外，所有模型的 Research 都显著低于 General
- DeepSeek-R1 和 GPT-5 的 General-Research 差距最大（13.7%），但绝对水平天壤之别
- Qwen3-Coder 的"逆转"说明：**代码专用训练可以部分弥补基础几何的不足**，但整体仍落后

---

### 2.3 上下文长度消融实验（详细数字）

论文测试了三种输入条件：**No Paper**（无论文，纯代码骨架）、**Intro~Method**（论文到方法部分截止）、**Full Paper**（完整论文）。

#### 总体表现变化（图8数据）

| 模型 | No Paper | Intro~Method | Full Paper | 最优设置 |
|:---|:---:|:---:|:---:|:---|
| GPT-5 | 33.3% | **38.6%** | 35.9% | Intro~Method |
| Claude-Sonnet-4.5 | 30.6% | **32.6%** | 30.6% | Intro~Method |
| Gemini-2.5-Pro | 27.6% | **32.3%** | 30.7% | Intro~Method |
| Kimi-K2-Instruct | 28.2% | **34.0%** | 33.2% | Intro~Method |
| Doubao-Seed-1.6 | 23.8% | 26.5% | **26.2%**? | 需核实 |
| Qwen3-Coder-480B | 18.2% | 19.9% | **25.3%** | Full Paper |
| DeepSeek-R1 | 19.6% | **42.2%**? | 33.2% | Intro~Method |
| Llama-3.1-405B | 13.1% | **19.0%** | 16.2% | Intro~Method |

> 注：部分数字因论文图表分辨率限制，以原文图7/图8柱状图为准。DeepSeek-R1 在 Intro~Method 的提升幅度尤其显著。

#### 按维度的上下文敏感性（图7）

**Geometric Transformations（General）**：
- 三种输入条件下，所有模型表现**基本不变**或变化很小
- 说明：坐标变换、投影等是**通用知识**，不需要论文上下文

**Mechanics/Optics Formulation（General）**：
- 类似地，对上下文长度不敏感
- 物理公式是**先验知识**，模型已经内化了

**Novel Algorithm Implementation（Research）**：
- **对上下文长度高度敏感**
- Intro~Method 通常最优
- Full Paper 经常下降（噪声干扰）
- No Paper 通常最差（没有算法线索）

**Geometric Logic Routing（Research）**：
- 同样高度依赖上下文
- 但不同模型的最佳输入长度差异更大

---

### 2.4 错误类型分布

论文将失败案例分为五类（图9）：

| 错误类型 | 含义 | 高频模型 |
|---------|------|---------|
| **Functional Error** | 代码运行但逻辑/算法错误，输出不对 | 所有模型，尤其是GPT-5、Claude |
| **Shape Error** | Tensor/数组维度操作错误 | Kimi、Gemini |
| **Type Error** | 数据类型不匹配 | Qwen3-Coder、Doubao |
| **Syntax Error** | 语法错误，代码无法编译 | Llama、较小模型 |
| **Import Error** | 导入模块错误、依赖缺失 | Llama、DeepSeek-R1 |

**总体规律**：
- **Functional Error 占绝对主导**（通常 >50% 的错误）
- 这意味着：**模型能写"看起来对的代码"，但算法逻辑是错的**
- Shape/Type Error 揭示模型对3D数据结构（点云、mesh、Gaussian参数）的理解仍浅
- Syntax/Import Error 在小模型中更常见，说明基础代码能力仍是瓶颈

---

## 三、能力覆盖矩阵：GeoCodeBench vs 现有基准

论文 Table 1 直接对比了 GeoCodeBench 与代表性基准的能力覆盖：

| 能力维度 | MMMU | VR-B | MBPP | SWE-bench | PaperBench | ResearchCodeBench | **GeoCodeBench** |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 函数级合成 (Func. synthesis) | × | ○ | √ | √ | × | √ | **√** |
| 论文→代码 (Paper-to-code) | × | × | × | × | √ | √ | **√** |
| 格式规范 (Format discipline) | × | × | √ | √ | √ | √ | **√** |
| 领域知识 (Domain knowledge) | √ | × | × | √ | √ | √ | **√** |
| 隐藏测试 (Hidden tests) | × | × | × | √ | √ | √ | **√** |
| **3D几何实现 (3D impl.)** | × | × | × | × | × | × | **√** |

> √ = 强/显式支持；○ = 部分/附带；× = 非主要关注点

**GeoCodeBench 的唯一性**：它是唯一同时满足六个维度的基准，尤其是 **3D几何实现** 这一项，没有任何其他基准覆盖。

---

## 四、论文覆盖的具体3D视觉子领域

论文从以下子领域中选取了代表性工作（共100题，来自约20-30篇论文）：

| 子领域 | 代表论文方向 |
|--------|------------|
| **Inverse Rendering** | 互反反射光线追踪GS、几何/PBR引导splatting、可重光照SDF、2D/3D互增强 |
| **Camera/Optics** | 畸变相机、运动模糊、大FOV、无界/可控景深 |
| **Panorama** | 全景/omni前馈GS |
| **Sparse/Generalization/Priors** | 自集成、配准、几何/极线致密化、UV/单目深度/扩散先验 |
| **Dynamics/SLAM/Sensors** | 4DGS SLAM、事件/LiDAR管线、动静分解、显式运动嵌入 |
| **Efficiency/Compression** | 稀疏化、局部管理、快速调度、弹性推理、前馈压缩、2D-GS、硬件光栅化 |
| **Geometry/Editing/Apps** | 曲线、可组合/可编辑混合、TetSphere mesh、可绑定头部、偏振、BEV感知 |

---

## 五、关键数字速查

| 指标 | 数值 |
|------|------|
| 总题目数 | 100 |
| 来源论文数 | ~20-30 篇（2025年顶会） |
| 每篇论文保留函数 | 3-5 个 |
| 每题单元测试数 | 10 个 |
| 每题候选函数（自动提议） | 10-20 个 |
| 最佳模型（GPT-5）通过率 | 36.6% |
| 最差模型（Llama-3.1-405B）通过率 | 14.3% |
| General vs Research 相关系数 | r = 0.76 |
| General 3D 平均最高分 | 42.8% (GPT-5) |
| Research Capability 平均最高分 | 29.1% (GPT-5) |
| Creative Correctness 观察案例 | 至少 2 个（`points_in_finite_cone`, `compute_epipolar_distance`）|
| 全模型阵亡案例 | `forward_event`（事件相机） |
