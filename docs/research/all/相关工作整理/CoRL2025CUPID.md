# [CUPID: Curating Data your Robot Loves with Influence Functions](https://cupid-curation.github.io/)

## 相关工作要点
- 问题与动机：模仿学习性能高度依赖演示数据，但传统“质量”启发式与真实闭环表现不一致，缺少对“哪条演示导致成功/失败”的因果解释。
- 任务与场景：单任务模仿学习的数据策划，关注混合质量数据、分布移位下的多策略数据、以及训练数据中的虚假相关。
- 数据来源与质量：RoboMimic 多人数据集（Lift/Square/Transport MH，含质量分级）；Franka FR3 实机任务（Figure-8、TuckBox、Bookshelf）。
- 方法路线：基于 influence function 估计每条演示对期望回报的影响，排序后用于过滤或选择新数据；可与动作影响力质量度量组合。
- 训练与评估：使用 diffusion policy（仿真/实机），部分实机还使用 π0；设置 filter-k 与 select-k；有限 rollout 预算。
- 对比基线：DemInf（互信息质量）、Demo-SCORE（成功/失败分类器）、Success Similarity、Random、Oracle。
- 局限与适用性：依赖 policy rollouts 估计影响力，样本少时方差较大；质量指标并非总是有益；数据/任务性质决定是否适合强过滤。

## 相关工作框架

### 1. 背景与问题定义
- 机器人模仿学习中，监督训练目标与闭环任务回报不一致，导致“高质量演示”不一定带来高成功率。
- 现有方法多为离线启发式（质量/多样性/不确定性）或依赖成功失败标签，缺少对单条演示与闭环表现的直接因果连接。
- 目标：以“对期望回报的影响”为核心信号，对演示进行因果价值评估与排序。

### 2. 任务与数据设置
- 任务类型：
- RoboMimic MH：Lift、Square、Transport（含多级质量标签，Transport 为双臂任务，难度更高）。
- Franka FR3：Figure-8（系绳结），TuckBox（策略在分布移位下鲁棒性不同），Bookshelf（背景-动作虚假相关）。
- 数据规模：RoboMimic 每任务约 300 条演示；Franka 每任务约 120-160 条。
- 评价方式：filter-k（移除 k 条有害演示）与 select-k（从候选池中选 k 条提升性能）。

### 3. 方法路线

#### 3.1 影响力估计（Influence on Return）
- 将“单条演示对期望回报的影响”形式化为 influence function，并分解为 state-action 转移级别可累加项。
- 通过 policy rollouts 结合 REINFORCE 风格估计器近似计算影响力；无需额外任务标注。

#### 3.2 数据选择与采样
- filter-k：剔除负向影响最大的演示。
- select-k：从新采集/预采集数据中选取高正向影响的演示。
- 通过影响力排序统一支持“过滤已有数据”和“选择新数据”。

#### 3.3 质量信号融合（CUPID-QUALITY）
- 提出基于 action influence 的质量度量，用于惩罚异常/噪声影响分数。
- 与性能影响力按权重混合（例如 50/50），在混合质量数据上可降方差、提性能，但在部分场景会引入偏差。

### 4. 训练与评估
- 架构：官方 diffusion policy（仿真/实机），实机还测试 π0。
- rollout 预算：仿真 m=100，实机 m=25。
- 成功率：仿真用 500 次 rollouts（最后 10 个 checkpoint），实机用最后 checkpoint 的 25 次 rollouts。

### 5. 对比与消融
- Baselines：DemInf（互信息质量）、Demo-SCORE、Success Similarity、Random、Oracle。
- 结果要点（混合质量）：
- CUPID 在 RoboMimic 上的成功率稳定优于或接近 DemInf，尽管 DemInf 能筛出更“高质量”的演示。
- CUPID-QUALITY 在混合质量设置中通常最强，部分任务超过 Oracle，Transport MH 上用不到 33% 演示即可超官方 diffusion policy。
- 结果要点（多策略/虚假相关）：
- TuckBox：分布移位下滑动策略失效，CUPID 通过失败 rollouts 关联到不鲁棒策略，成功率达 84–88%，DemInf 与 CUPID-QUALITY 因“低方差=高质量”而误筛。
- Bookshelf：背景颜色与动作策略虚假相关，基线易误判；CUPID 识别“白底+横向拉”的失败因果，成功率由 44% 提升至 84%。
- Figure-8：CUPID 相对基线平均提升 38%，CUPID-QUALITY 进一步提升。

### 6. 局限与讨论
- 影响力估计依赖 rollouts，少量 rollouts 时噪声较大。
- 质量度量在混合质量任务有效，但在“质量不是主瓶颈”的任务（如策略偏差、虚假相关）可能削弱性能。
- 数据属性决定收益：如 Lift MH 高冗余可强过滤，Square MH 几乎所有演示都重要，不宜过度筛除。

## 相关工作评价与本文定位（CUPID-ACT）
- 定位：CUPID 代表“闭环回报影响力”导向的演示价值评估范式，区别于仅衡量质量或相似性的离线指标。我们的工作在其基础上，面向 ACT 策略做可复现、可落地的工程化改进（CUPID-ACT）。
- 相对 DemInf：DemInf 以互信息衡量“动作可预测性+多样性”，可在无 rollouts 条件下做质量筛选，但该指标与闭环回报并不一致，难以处理策略鲁棒性与虚假相关。CUPID-ACT 直接用回报影响力排序演示，更贴近部署成功率。
- 相对 Demo-SCORE：Demo-SCORE 通过成功/失败分类器过滤演示，信号更粗粒度，无法给出“每条演示的正负影响”。CUPID-ACT 产生带符号的影响值，可用于过滤、重加权或选择新数据。
- 相对原始 CUPID：
- 将影响函数落到 ACT 策略：影子训练 + 梯度缓存；用伪 Huber 统一训练/评估损失，避免 ACT 结构不兼容。
- 通过随机投影压缩梯度缓存，降低存储与 I/O 成本。
- 采用 LiSSA 估计 H^{-1}g，使 influence 可在可接受的计算预算内落地。
- 通过混合评估 rollouts 与回报中心化，减小估计方差；允许在小 rollout 预算下运行。
- 数据策划上同时支持过滤与重加权，强调“提高闭环成功率”而非仅提升静态质量。
