# [Curating Demonstrations using Online Experience](https://arxiv.org/abs/2503.03707)

## 相关工作要点
- 问题与动机：演示数据常包含多种策略与质量差异，人工筛选困难，可靠策略往往在闭环执行中才暴露。
- 核心思路：用 policy rollout 的成功/失败来训练分类器，反向筛掉“导致失败的演示”。
- 方法：训练初始策略 → 多 checkpoint rollouts → 训练并交叉验证成功/失败分类器 → 用分类器给演示打分并过滤 → 重新训练。
- 结果：仿真任务提升 15–35% 绝对成功率；真实 ALOHA 任务提升 15–40%，长时序 Jellybean 任务完整成功率从 0 提升到 20。

## 方法概述
- 输入：单任务、全部“成功”的异质演示集合 + 少量在线 rollouts（带成功/失败标签）。
- 分类器：小型 MLP（二层、每层 8 单元），用 BCE 训练；跨 checkpoint 交叉验证防止过拟合 rollout 分布。
- 过滤策略：对演示内所有状态预测成功概率并求均值，超过阈值 γ 保留。
- 成本：实验证明每个 checkpoint 约 10 次 rollout 即可（总计 <100）。

## 相关工作框架

### 1. 背景与问题定义
- 异质演示包含可靠/不可靠策略，纯 BC 容易学到脆弱策略。
- 可靠性难以由表面质量判断，但可通过 policy rollouts 的成功与失败来区分。

### 2. 任务与数据设置
- 仿真：
- ALOHA 双臂 peg insertion（MuJoCo），演示由 PegA/B/C 三种策略脚本生成。
- Robosuite square peg（Panda），包含 Human 与 SquareA/B 两种脚本策略混合。
- 实机 ALOHA：Spoon、Chocolate、Sharpie、Strawberry、Jellybean。
- 演示量：Spoon 40，Chocolate/Sharpie 各 50，Strawberry 120；Jellybean 使用 124 众包 + 100 专家演示。

### 3. 方法路线

#### 3.1 在线经验驱动筛选
- 从训练过程的多个 checkpoint 采样 rollouts，标记成功/失败。
- 训练多组分类器并在最后 checkpoint 的 rollouts 上交叉验证，选最稳健的分类器。

#### 3.2 演示过滤
- 逐状态预测成功概率，轨迹均值超过阈值则保留。
- 支持轨迹级或 chunk 级过滤（论文主结果为轨迹级）。

#### 3.3 训练策略
- 基础策略使用 Diffusion Policy（Robosuite）与 ACT（ALOHA），以覆盖异质演示分布。
- 过滤后重新训练得到最终策略。

### 4. 训练与评估
- 仿真：每种数据混合比例评估多次，报告成功率与置信区间。
- 实机：多任务真实场景验证，Jellybean 额外用 5 个子任务聚合得分。

### 5. 对比与消融
- Baselines：Base Policy（全量演示）、Training Loss 重加权、Auto-IL、Return-Conditioned Policy。
- 结果要点：
- 仿真：平均提升 15–35% 绝对成功率；square peg 平均 >94%，peg insertion >90%。
- 实机：四个 ALOHA 任务提升 15–40% 绝对成功率；Jellybean 聚合分数 27→49，完整任务成功率 0→20。

### 6. 局限与适用性
- 需要在线 rollouts 与成功/失败判别；若任务难以自动判定成功，成本上升。
- 若初始策略几乎全成功/全失败，分类器学习信号不足。
- 过滤阈值与 rollout 预算会影响结果，需在任务上调参。

## 相关工作评价与本文定位（CUPID-ACT）
- Demo-SCORE 优势：实现简单、信号直接，适合“有成功/失败判定但不便做梯度/HVP”的场景。
- Demo-SCORE 局限：输出为成功概率的粗粒度过滤，难以给出“每条演示的正负影响”；当成功/失败极不均衡时信号不足。
- 我们的位置：CUPID-ACT 以影响函数归因，输出带符号的影响值，可做过滤、重加权与选择；能够把失败 rollouts 追溯到具体演示策略与片段。
- 代价与互补：CUPID-ACT 计算成本高于 Demo-SCORE，但更接近因果归因；若只需要快速筛选可用 Demo-SCORE，若追求精细归因与稳健提升则使用 CUPID-ACT。
