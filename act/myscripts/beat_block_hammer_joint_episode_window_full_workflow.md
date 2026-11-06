# `beat_block_hammer_ur5_randomized_500_act_joint_episode_window` 全流程说明

本文档专门总结下面这条后台命令的实际计算链路、关键公式、输出产物和工程含义：

```bash
CUDA_VISIBLE_DEVICES=4 \
PYTHON_BIN=python \
DATASET_ROOT=/workspace/liushuo/dataset/beat_block_hammer_ur5_randomized_500 \
DATASET_REPO_ID=local/beat_block_hammer_ur5_randomized_500 \
PRETRAINED_DIR=/workspace/liushuo/cupid/act/outputs/robotwin/beat_block_hammer_ur5_randomized_500_act \
EVAL_ROLLOUT_DIR='/workspace/liushuo/RoboTwin/eval_result/beat_block_hammer/ACT/demo_randomized_ur5_dual/beat_block_hammer_ur5_randomized_500_act/2026-04-01 11:29:13/episodes' \
OUTPUT_BASE=/workspace/liushuo/cupid/act/outputs/robotwin_joint_episode_window_new \
RUN_NAME=beat_block_hammer_ur5_randomized_500_act_joint_episode_window \
JOB_NAME=beat_block_hammer_ur5_randomized_500_act_joint_episode_window \
QUALITY_SOURCE=trajectory_eval \
RECOMPUTE_SCORE=1 \
INFLUENCE_DEVICE=cuda:0 \
EPISODE_DROP_RATIO=0.10 \
WINDOW_SIZE=50 \
WINDOW_STRIDE=50 \
WINDOW_LOCAL_EPISODE_SCORE_WEIGHT=0.4 \
WINDOW_LOCAL_QUALITY_WEIGHT=0.4 \
WINDOW_LOCAL_EPISODE_JOINT_WEIGHT=0.2 \
MIN_COVERAGE_RATIO=0.5 \
MARGINAL_GAIN_THRESHOLD=0.15 \
RELATIVE_GAIN_THRESHOLD=0.5 \
nohup bash /workspace/liushuo/cupid/act/myscripts/run_filter_and_train_joint_episode_window.sh \
> /workspace/liushuo/cupid/act/logs/beat_block_hammer_joint_episode_window_new.log 2>&1 &
```

## 1. 这条命令最终做了什么

它会顺序完成 3 件事：

1. 用 `cupid_act_influence.py` 重新计算训练集 episode / window 的 influence 分数。
2. 用 `filter_joint_episode_window.py` 做两阶段筛选：
   - 先删掉联合分最低的 10% episode。
   - 再对剩余 episode 做 joint episode-window 贪心窗口选择。
3. 用 `train_act_cupid_reweight_joint.py` 基于筛选结果训练 ACT，并把窗口选择结果映射成 frame 级采样 mask / weight。

由于末尾有 `nohup ... &`，所以整个流程在后台运行，所有 stdout/stderr 都写入：

```text
/workspace/liushuo/cupid/act/logs/beat_block_hammer_joint_episode_window_new.log
```

## 2. 这组参数在脚本里的真实展开结果

`run_filter_and_train_joint_episode_window.sh` 会把这组环境变量解析成下面这些实际路径：

### 2.1 基础输入

- 训练数据集根目录：
  `/workspace/liushuo/dataset/beat_block_hammer_ur5_randomized_500`
- LeRobot repo_id：
  `local/beat_block_hammer_ur5_randomized_500`
- 预训练 ACT 模型目录：
  `/workspace/liushuo/cupid/act/outputs/robotwin/beat_block_hammer_ur5_randomized_500_act`
- rollout 评估轨迹目录：
  `/workspace/liushuo/RoboTwin/eval_result/beat_block_hammer/ACT/demo_randomized_ur5_dual/beat_block_hammer_ur5_randomized_500_act/2026-04-01 11:29:13/episodes`

### 2.2 自动推导出的输出目录

- influence 输出目录：
  `/workspace/liushuo/cupid/act/outputs/robotwin_joint_episode_window_new/beat_block_hammer_ur5_randomized_500_act_joint_episode_window/score`
- selection 输出目录：
  `/workspace/liushuo/cupid/act/outputs/robotwin_joint_episode_window_new/beat_block_hammer_ur5_randomized_500_act_joint_episode_window/selection`
- 训练输出目录：
  `/workspace/liushuo/cupid/act/outputs/robotwin_joint_episode_window_new/beat_block_hammer_ur5_randomized_500_act_joint_episode_window/train`

### 2.3 自动推导出的关键文件

- influence 主结果：
  `/workspace/liushuo/cupid/act/outputs/robotwin_joint_episode_window_new/beat_block_hammer_ur5_randomized_500_act_joint_episode_window/score/score_episode_window_ws50_st50.pkl`
- influence 分阶段 cache 前缀：
  `/workspace/liushuo/cupid/act/outputs/robotwin_joint_episode_window_new/beat_block_hammer_ur5_randomized_500_act_joint_episode_window/score/score`
- selection 核心产物：
  `/workspace/liushuo/cupid/act/outputs/robotwin_joint_episode_window_new/beat_block_hammer_ur5_randomized_500_act_joint_episode_window/selection/joint_episode_window_selection.npz`

### 2.4 默认分支判断

这条命令没有显式设置的脚本变量，会走默认值：

- `MODE=single`
- `SKIP_TRAIN=0`
- `MIN_WINDOWS_PER_EPISODE=1`
- `QUALITY_SOURCE=trajectory_eval`
- `RECOMPUTE_SCORE=1`

因此本次运行不是 sweep，不会跳过训练，而且一定会强制重算 influence 分数。

## 3. GPU 语义

这里有一个很容易混淆的点：

- Shell 层设置的是 `CUDA_VISIBLE_DEVICES=4`
- influence 脚本收到的是 `INFLUENCE_DEVICE=cuda:0`

这两个并不冲突。因为在当前进程可见设备里只暴露了物理 GPU 4，所以脚本内的 `cuda:0` 实际上映射到物理 GPU 4。

换句话说，本次 influence 和训练实际都跑在物理 GPU 4 上，除非后续子进程显式覆写了可见卡。

## 4. Shell 脚本执行链路

主入口脚本是：

```text
/workspace/liushuo/cupid/act/myscripts/run_filter_and_train_joint_episode_window.sh
```

单次运行分 3 个阶段。

### 4.1 Stage A: 先确保 `score.pkl` 存在

脚本会先执行 `ensure_window_score()`。

当前因为：

- `SCORE_PKL` 没有手工指定
- `RECOMPUTE_SCORE=1`

所以即使目标 `score.pkl` 已经存在，也不会复用，而是重新执行：

```bash
python /workspace/liushuo/cupid/act/myscripts/cupid_act_influence.py \
  --pretrained_dir /workspace/liushuo/cupid/act/outputs/robotwin/beat_block_hammer_ur5_randomized_500_act \
  --train_dataset_path /workspace/liushuo/dataset/beat_block_hammer_ur5_randomized_500 \
  --eval_rollout_dir '/workspace/liushuo/RoboTwin/eval_result/beat_block_hammer/ACT/demo_randomized_ur5_dual/beat_block_hammer_ur5_randomized_500_act/2026-04-01 11:29:13/episodes' \
  --output_path /workspace/liushuo/cupid/act/outputs/robotwin_joint_episode_window_new/beat_block_hammer_ur5_randomized_500_act_joint_episode_window/score/score_episode_window_ws50_st50.pkl \
  --stage_cache_prefix /workspace/liushuo/cupid/act/outputs/robotwin_joint_episode_window_new/beat_block_hammer_ur5_randomized_500_act_joint_episode_window/score/score \
  --aggregate_mode episode \
  --window_size 50 \
  --window_stride 50 \
  --device cuda:0
```

### 4.2 Stage B: joint episode-window 筛选

接着执行：

```bash
python /workspace/liushuo/cupid/act/myscripts/filter_joint_episode_window.py \
  --score-pkl .../score_episode_window_ws50_st50.pkl \
  --dataset-root /workspace/liushuo/dataset/beat_block_hammer_ur5_randomized_500 \
  --output-dir .../selection \
  --quality-source trajectory_eval \
  --episode-influence-weight 0.5 \
  --episode-quality-weight 0.5 \
  --episode-drop-ratio 0.10 \
  --window-size 50 \
  --window-stride 50 \
  --window-local-episode-score-weight 0.4 \
  --window-local-quality-weight 0.4 \
  --window-local-episode-joint-weight 0.2 \
  --budget-mode auto_marginal \
  --min-windows-per-episode 1 \
  --min-coverage-ratio 0.5 \
  --marginal-gain-threshold 0.15 \
  --relative-gain-threshold 0.5
```

### 4.3 Stage C: 用 selection 结果训练 ACT

最后执行：

```bash
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/dataset/beat_block_hammer_ur5_randomized_500 \
  --score-pkl .../score_episode_window_ws50_st50.pkl \
  --selection-npz .../selection/joint_episode_window_selection.npz \
  --selection-override-episodes \
  --output_dir .../train \
  --job_name beat_block_hammer_ur5_randomized_500_act_joint_episode_window \
  --policy.type act \
  --filter-score-mode joint \
  --joint-drop-quantile 0 \
  --drop-weight-quantile 0 \
  --drop-bad-ratio 1.1 \
  --joint-quality-source trajectory_eval \
  --weight-mode influence \
  --use-weighted-sampler \
  --segment-mask \
  --segment-weight \
  --dataset.repo_id local/beat_block_hammer_ur5_randomized_500 \
  --policy.repo_id local/beat_block_hammer_ur5_randomized_500 \
  --pretrained-dir /workspace/liushuo/cupid/act/outputs/robotwin/beat_block_hammer_ur5_randomized_500_act
```

## 5. Influence 计算全流程

`cupid_act_influence.py` 的核心目标是：估计训练集中的每个样本、每个 episode、每个 window 对 rollout 回报的正负贡献。

### 5.1 输入各自负责什么

- `PRETRAINED_DIR`
  负责提供当前 ACT 参数 `theta`。
- `DATASET_ROOT`
  负责提供训练数据，用来计算每个训练样本的梯度投影。
- `EVAL_ROLLOUT_DIR`
  负责提供 rollout episode，用来构造回报 `R_e`，再反推出“验证目标方向”。

这 3 个输入缺一不可。没有 rollout，就没有 influence 的目标向量；没有训练集，就没有被评估的候选样本；没有预训练模型，就没有当前参数点下的局部影响分析。

### 5.2 四阶段计算图

代码实现上它走 4 个 stage。

#### Stage 1: 训练集梯度缓存

对训练集每个样本 `z_i` 计算梯度：

```text
g_i = ∇_theta l(z_i, theta)
```

然后把梯度投影到低维空间：

```text
g'_i = P g_i
```

其中 `P` 是随机投影矩阵，默认维度 `proj_dim=4096`。如果 dense projection 显存不够，会自动退化到 `CountSketch`。

这个阶段的 cache 会写到类似下面的路径：

```text
.../score/score.stage1_train_projections.npy
```

#### Stage 2: rollout return gradient

脚本会从 `EVAL_ROLLOUT_DIR` 读取所有 `*.pkl` episode，默认 reward 提取逻辑如下：

1. 若存在 `reward` 列，则对 episode 全步求和。
2. 否则看 `next.reward`。
3. 再否则取最后一步 `success` 或 `next.success`。
4. 如果都没有，退化到最后一行里能找到的 reward/success 标量。

得到每个 rollout episode 的回报 `R_e` 后，先做中心化：

```text
R'_e = R_e - mean(R)
```

再把同一 rollout episode 内的梯度累加，构成 return gradient：

```text
g_return = (1 / N) * Σ_e [ R'_e * Σ_t ∈ e ∇_theta l(x_{e,t}, theta) ]
```

注意这里 eval batch 默认会加 action noise（`noise_sigma=0.1`），用来让影响分析更贴近 rollout 扰动下的稳健性。

这个阶段的 cache 默认写到：

```text
.../score/score.stage2_return_gradient.pt
```

#### Stage 3: LiSSA 近似逆 Hessian-Vector Product

脚本不会显式构造 Hessian，而是用 LiSSA 递归近似：

```text
v ≈ H^{-1} g_return
```

默认超参数：

- `lissa_depth=8`
- `lissa_samples=4`
- `lissa_scale=10.0`
- `lissa_damping=0.01`
- `hvp_batch_size=4`

这个阶段的 cache 默认写到：

```text
.../score/score.stage3_inverse_hvp.pt
```

#### Stage 4: 训练样本 influence 打分

最终 influence 近似为：

```text
I_i = - < g'_i, v' >
```

其中 `v'` 是 inverse-HVP 在同一投影空间中的表示。

样本级分数得到后，脚本会继续做两层聚合：

1. episode 聚合：
   对同一 episode 的所有样本 influence 求均值，得到 `episode_influence_scores`。
2. window 聚合：
   对每个 episode 内滑窗 `[s, e)` 的样本 influence 求均值，得到 `window_influence_scores`。

因为本次传入了：

- `aggregate_mode=episode`
- `window_size=50`
- `window_stride=50`

所以：

- 顶层 `influence_scores` 返回的是 episode 级分数。
- metadata 里仍然会额外保存 window 级分数，供后续筛选使用。

### 5.3 `score.pkl` 里和本次流程直接相关的字段

核心字段包括：

- `influence_scores`
- `metadata["episode_rewards"]`
- `metadata["episode_lengths"]`
- `metadata["episode_influence_scores"]`
- `metadata["window_influence_scores"]`
- `metadata["window_episode_index"]`
- `metadata["window_start"]`
- `metadata["window_end"]`
- `metadata["window_weights"]`
- `metadata["window_drop_mask"]`

其中后 5 个数组是后续 window 级筛选的关键输入。

### 5.4 `score` 的物理含义到底是什么

这里的 `score.pkl` 不应该被理解成“轨迹质量分”，它本质上是一个 **influence 归因结果容器**。

更准确地说，它回答的问题是：

```text
在当前预训练 ACT 参数点附近，
如果我更重视某个训练样本 / 某个 episode / 某个局部窗口，
它会让 rollout 回报变好还是变坏，影响有多大？
```

因此：

- influence 高，不代表这段轨迹一定平滑或一定是“标准示范”
- influence 高，只代表它对当前策略修正 rollout 目标更有帮助
- influence 低甚至为负，说明它可能在当前模型状态下带来误导或负迁移

这也是为什么后续还要再叠加 `quality`。因为：

- influence 解决“有没有用”
- quality 解决“干不干净”

这两个维度不能互相替代。

### 5.5 为什么 rollout reward 要先中心化

代码里在构造 return gradient 时，先做了：

```text
centered_rewards = rewards - rewards.mean()
```

这样做的目的不是数值好看，而是避免所有 rollout 都朝同一个常数偏置方向推梯度。

如果不中心化，会出现两个问题：

1. 所有 rollout 即使差异很小，也会共同推一个平均方向。
2. influence 更像是在拟合“平均 reward 基线”，而不是区分“哪些训练样本更能解释好 rollout 和坏 rollout 的差别”。

中心化后，真正起作用的是：

- 高于平均 reward 的 rollout，推动正向归因
- 低于平均 reward 的 rollout，推动负向归因

这更接近“判别式 influence”，也更适合做数据优选。

### 5.6 为什么最终要从 sample 聚合到 episode / window

单个 sample 的 influence 颗粒度太细，直接用来删数据会有两个问题：

1. 抖动太大。相邻 frame 的 influence 容易高频震荡。
2. 工程上不稳。直接逐帧删样本会破坏 ACT 所依赖的局部时序连续性。

所以代码里专门做了两层聚合：

1. episode 聚合
   解决“这整条 demonstration 值不值得信任”
2. window 聚合
   解决“这条 demonstration 里的哪一段最值得保留”

从具身智能数据优选角度，这一步非常关键。因为很多轨迹的问题不是“全坏”或“全好”，而是：

- 前半段接近专家
- 接触瞬间有抖动
- 回撤阶段又恢复稳定

只做 episode 级平均，会把这种局部结构冲淡掉。

## 6. `trajectory_eval` 质量分计算全流程

本次 `QUALITY_SOURCE=trajectory_eval`，因此 `filter_joint_episode_window.py` 不会去读 rollout 结果来当质量分，而是直接在训练集 parquet episode 上计算 episode 质量。

这点非常重要：

- `EVAL_ROLLOUT_DIR` 只服务 influence。
- `trajectory_eval` 质量分只服务筛选阶段。
- 二者来自不同数据来源，承担不同角色。

### 6.1 每个 episode 计算哪些分量

对每个训练 episode，脚本会尽量读取：

- `action`
- `observation.state`

并计算以下指标：

1. `terminal_score`
   终态一致性。用 final state 的 kNN 距离衡量，距离越小越好。
2. `efficiency_score`
   状态位移 / 状态路径长度，越接近直达越好。
3. `progress_score`
   从首状态到末状态的净位移，越大越好。
4. `monotonic_score`
   沿主方向前进的占比，越大越好。
5. `backtrack_score`
   沿主方向回退的占比，越小越好。
6. `stall_score`
   停滞步占比，越小越好。
7. `state_smooth_score`
   相邻状态差分相对幅值，越小越好。
8. `action_smooth_score`
   相邻动作差分相对幅值，越小越好。
9. `length_score`
   episode 长度接近全体中位数则更高。

### 6.2 本次实际生效的默认权重

因为命令没有覆写 quality 相关细参，所以会采用 `filter_joint_episode_window.py` 里的默认值：

```text
terminal_weight      = 0.25
efficiency_weight    = 0.20
progress_weight      = 0.25
monotonic_weight     = 0.15
backtrack_weight     = 0.10
stall_weight         = 0.10
state_smooth_weight  = 0.15
action_smooth_weight = 0.10
length_weight        = 0.05
fusion               = weighted_sum
terminal_knn_k       = 5
terminal_distance_metric = l2
terminal_normalize   = none
terminal_agg         = mean
```

最终 episode 质量分是加权平均：

```text
quality_i =
    Σ_k [ w_k * component_k(i) ] / Σ_k [ w_k ]
```

所有 component 在融合前都会先被归一到 `[0, 1]`。

### 6.3 本次没有启用的 gate

虽然 `trajectory_eval` 支持对 terminal / progress 太差的轨迹做 gate cap，但本次命令没有设置：

- `traj_eval_min_terminal_score`
- `traj_eval_min_progress_score`
- `traj_eval_gate_quality_cap`

在 `filter_joint_episode_window.py` 中，这三个默认都是 `-1`，因此本次不会额外做质量上限门控。

### 6.4 用 0 基础能看懂的方式理解三类质量指标

为了让质量分更容易理解，可以把 episode 级轨迹质量统一拆成 3 个最直观的大类：

```text
I1: 任务结果一致性
I2: 运动过程平滑性
I3: 轨迹时长合理性
```

它们分别回答 3 个问题：

1. `I1`
   这条轨迹最后有没有把任务做对，而且是不是一直朝正确方向在推进。
2. `I2`
   这条轨迹执行过程稳不稳，有没有乱抖、乱晃、动作突变。
3. `I3`
   这条轨迹总时长正不正常，是不是明显过短或过长。

更简单地说：

- `I1` 看结果像不像成功轨迹
- `I2` 看过程像不像稳定专家示范
- `I3` 看节奏是否符合任务正常完成区间

### 6.5 `I1`: 任务结果一致性用了哪些技术点

`I1` 用于衡量轨迹是否真正朝任务目标有效推进，并最终收敛到合理的完成状态。

它在代码里不是单一量，而是由下面这些小项共同组成：

1. 终端状态一致性
   看轨迹最后停在哪里。实现上使用终态 `kNN` 距离，判断该终态是否落在合理终态簇附近。
2. 轨迹进展程度
   看起点到终点的净位移是否足够大，判断轨迹有没有真正推动任务前进。
3. 执行效率
   看“净位移 / 实际路径长度”，判断轨迹是否存在大量无效绕行。
4. 单调推进性
   看轨迹是否整体沿主运动方向持续前进。
5. 回退惩罚
   看轨迹是否反复往回走，抑制频繁后退的示范。
6. 停滞惩罚
   看轨迹中是否有大量几乎不动的片段，抑制长时间无效停顿。

从 0 基础角度，可以把 `I1` 理解成：

```text
这条轨迹最后有没有完成任务，
而且是不是沿着比较合理的方向稳步完成的。
```

形式上，可以写成：

\[
I_1=
\frac{
\alpha_1 S^{\mathrm{term}}
+\alpha_2 S^{\mathrm{prog}}
+\alpha_3 S^{\mathrm{eff}}
+\alpha_4 S^{\mathrm{mono}}
+\alpha_5 S^{\mathrm{back}}
+\alpha_6 S^{\mathrm{stall}}
}{
\alpha_1+\alpha_2+\alpha_3+\alpha_4+\alpha_5+\alpha_6
}
\]

其中：

- \(S^{\mathrm{term}}\)：终端状态一致性分数
- \(S^{\mathrm{prog}}\)：轨迹进展分数
- \(S^{\mathrm{eff}}\)：执行效率分数
- \(S^{\mathrm{mono}}\)：单调推进分数
- \(S^{\mathrm{back}}\)：回退惩罚对应分数
- \(S^{\mathrm{stall}}\)：停滞惩罚对应分数

### 6.6 `I2`: 运动过程平滑性用了哪些技术点

`I2` 用于衡量轨迹在执行过程中的运动学稳定性，避免高频抖动、突变和不连续控制。

它在代码里主要由 2 个部分组成：

1. 状态平滑性
   看相邻时刻状态变化是否平稳，避免状态轨迹忽大忽小、来回震荡。
2. 动作平滑性
   看相邻时刻动作指令是否平稳，避免控制信号大幅突变或高频抖动。

为了减少量纲和幅值差异带来的偏置，代码实际使用的是一种“相对抖动”：

- 状态差分平均值 / 状态幅值平均值
- 动作差分平均值 / 动作幅值平均值

然后再做“越小越好”的归一化。

对 0 基础来说，可以把 `I2` 理解成：

```text
这条轨迹执行起来像不像一个稳定、顺滑的专家动作，
而不是抖抖索索、忽快忽慢的控制。
```

形式上，可以写成：

\[
I_2=
\frac{
\beta_1 S^{\mathrm{state\_smooth}}
+\beta_2 S^{\mathrm{action\_smooth}}
}{
\beta_1+\beta_2
}
\]

其中：

- \(S^{\mathrm{state\_smooth}}\)：状态平滑性分数
- \(S^{\mathrm{action\_smooth}}\)：动作平滑性分数

### 6.7 `I3`: 轨迹时长合理性用了哪些技术点

`I3` 用于衡量轨迹持续时间是否处于该任务的正常完成区间，从而抑制异常过短或异常过长的示范。

它在代码里主要用了 2 个技术点：

1. 长度分布统计建模
   先统计整个数据集所有 episode 的长度分布。
2. 长度一致性评分
   用当前轨迹长度相对于全体轨迹长度中位数的偏离程度来打分。

代码里使用的是：

- 中位数 `median`
- 中位绝对偏差 `MAD`

然后构造一个指数衰减分数：

\[
I_3=
\exp\left(
-\frac{|T-T_{\mathrm{med}}|}
{\lambda \cdot \mathrm{MAD}(T)+\epsilon}
\right)
\]

这表示：

- 轨迹长度越接近任务主流完成时长，分数越高
- 偏离越大，分数越低

对 0 基础来说，可以把 `I3` 理解成：

```text
这条轨迹做任务的节奏正不正常。
太快可能没做完整，太慢可能夹杂了很多无效动作。
```

### 6.8 Episode 级别三类质量指标如何融合

在 episode 级别，三类指标一起构成全局质量分：

\[
I^{\mathrm{ep}}
=
\frac{
W_1 I_1 + W_2 I_2 + W_3 I_3
}{
W_1+W_2+W_3
}
\]

如果要给一组直观、稳妥、也便于论文表达的默认权重，可以写成：

\[
W_1=0.4,\qquad W_2=0.4,\qquad W_3=0.2
\]

也就是：

- 结果正确性和过程平滑性同样重要
- 时长合理性是辅助约束，因此权重稍低

### 6.9 Window 级别保留哪些、舍弃哪些

window 只是完整轨迹中的一个局部片段，因此很多 episode 级全局指标不再适合直接重算。

#### 6.9.1 Window 级别不再单独使用 `I3`

因为窗口长度由 `window_size` 人为固定，所以：

- 每个窗口本来就差不多长
- “时长是不是合理”在 window 级别已经没有区分意义

因此可以认为：

\[
I_3^{\mathrm{win}} = 0
\]

或者更准确地说，`I3` 不参与 window 局部质量打分。

#### 6.9.2 Window 级别也不重算完整的 `I1`

原因很简单：

- 局部窗口通常不包含完整任务终点
- 窗口最后一帧不一定有“任务完成状态”的语义
- 局部停顿也不一定是坏事，有时正对应抓取、对齐或接触准备

所以在当前实现里，window 级别不会重新计算一个完整 episode 式的 `I1`。

#### 6.9.3 Window 级别保留的核心是局部平滑性

因此，window 级别真正显式计算的局部质量主要就是：

1. 局部状态平滑性
2. 局部动作平滑性

形式上可写为：

\[
I^{\mathrm{win}}
=
\frac{
\gamma_1 S^{\mathrm{state\_smooth}}_{\mathrm{win}}
+\gamma_2 S^{\mathrm{action\_smooth}}_{\mathrm{win}}
}{
\gamma_1+\gamma_2
}
\]

这也和代码完全一致：window 质量分本质上就是窗口内部 `state jitter` 和 `action jitter` 的融合。

### 6.10 Window 最终价值并不只由质量决定

需要强调的是，window 最终被保留与否，并不是只看 `I^{\mathrm{win}}`。

当前实现会把 3 部分结合起来：

1. 窗口 influence 分数
2. 窗口局部质量分
3. 所属 episode 的全局先验

形式上可以写成：

\[
V^{\mathrm{win}}
=
\eta_1 I_{\mathrm{inf}}^{\mathrm{win}}
+\eta_2 I^{\mathrm{win}}
+\eta_3 I^{\mathrm{ep}}
\]

在你当前脚本默认设置下，对应权重大致就是：

\[
V^{\mathrm{win}}
=
0.4\, I_{\mathrm{inf}}^{\mathrm{win}}
+0.4\, I^{\mathrm{win}}
+0.2\, I^{\mathrm{ep}}
\]

也就是说：

- 局部窗口本身有没有用
- 局部窗口本身稳不稳
- 它所属的整条轨迹靠不靠谱

这三者会一起决定窗口的最终保留价值。

## 7. Joint Episode-Window 筛选全流程

### 7.1 Episode 级第一阶段：删掉最差 10%

先从 `score.pkl` 里取出：

- `episode_influence`
- `episode_quality`

然后分别做 `min-max` 归一化到 `[0, 1]`：

```text
norm_influence_i = normalize(episode_influence_i)
norm_quality_i   = normalize(episode_quality_i)
```

再计算 episode 联合分：

```text
episode_joint_score_i =
    (0.5 * norm_influence_i + 0.5 * norm_quality_i) / (0.5 + 0.5)
```

接着按升序排序，删除最底部：

```text
floor(0.10 * N_episode)
```

个 episode。

这一步的含义是：

- influence 保证“对 rollout 目标有用”
- quality 保证“轨迹本身结构合理”
- 两者各占一半，先做一次粗过滤

### 7.2 Window 构造

对每个保留下来的 episode，脚本会重新从 parquet 读出数据，按：

- `window_size=50`
- `window_stride=50`

生成候选窗口。

这意味着本次窗口基本是“不重叠切块”。只有在 episode 长度不能整除窗口大小时，末尾可能补一个对齐到尾部的窗口。

### 7.3 Window 局部质量分

每个候选窗口会算两个原始抖动量：

- `raw_action_jitter`
- `raw_state_jitter`

具体就是窗口内相邻帧差分范数的均值，越小越好。

再把它们分别转成“低抖动高得分”的分数，并融合成：

```text
window_quality_score
```

默认 action / state 权重各为 `0.5`。

### 7.4 Window 局部 influence 分

脚本会优先尝试从 `score.pkl.metadata` 里查找与 `(episode, start, end)` 对齐的：

```text
window_influence_scores
```

因为本次 influence 阶段已经显式计算了 `window_size=50, window_stride=50` 的窗口分数，所以这里会命中真实的 window influence，而不是退化到 episode influence。

该值在筛选脚本中会记为：

```text
window_episode_score
```

命名看起来像“episode score”，但本次运行里它实际上是 window-level influence score。

### 7.5 Window 局部综合值

对每个窗口，脚本会融合三部分：

1. `window_episode_score`
   局部 influence。
2. `window_quality_score`
   局部平滑质量。
3. `episode_joint_score`
   所属 episode 的全局先验。

本次权重是：

```text
0.4 : 0.4 : 0.2
```

因此：

```text
window_local_value =
    (0.4 * window_episode_score
   + 0.4 * window_quality_score
   + 0.2 * episode_joint_score) / 1.0
```

随后脚本把：

```text
window_contribution_score = window_local_value
```

也就是说，本次没有再额外区分“贡献分”和“局部分”。

### 7.5.1 为什么是 `0.4 / 0.4 / 0.2`

这组权重的直觉可以概括成一句话：

```text
局部有效性和局部质量并重，episode 全局先验只做弱约束。
```

对应解释如下：

1. `0.4 * window_episode_score`
   让真正对 rollout 目标有帮助的局部动作段优先保留。
2. `0.4 * window_quality_score`
   抑制高 influence 但高抖动、时序不稳定、可能是偶然对齐的片段。
3. `0.2 * episode_joint_score`
   给整体更可靠的 episode 一个先验加成，但不让它压过局部证据。

如果把 `episode_joint_score` 权重设得太高，筛选会退化成“好 episode 里的窗口大多都容易进”；如果把它设为 0，又容易丢掉全局上下文，导致局部窗口被孤立地过拟合打分。

### 7.6 每个 episode 内的贪心窗口选择

窗口不是按固定比例删，而是对每个 episode 独立做 greedy 选择。

每次从剩余候选窗口里选出边际收益最大的一个：

```text
marginal_gain =
    0.7 * contribution_score + 0.3 * coverage_novelty
```

其中：

- `contribution_score = window_contribution_score`
- `coverage_novelty` 表示该窗口相对已选窗口集合的新颖度

### 7.7 Coverage 是怎么定义的

本次 `coverage_feature=auto`，代码会优先尝试：

- 用窗口内 `observation.state` 的均值向量作为 feature

如果状态不可用，才会退化到：

- 用时间中心 `temporal_center / episode_length`

新颖度计算方式是：

```text
novelty(candidate) =
    min_distance(candidate, selected_set) / max_pairwise_distance
```

再裁剪到 `[0, 1]`。

这相当于在每个 episode 内抑制大量相似窗口重复入选。

### 7.7.1 `coverage_novelty` 的真正作用

如果没有 `coverage_novelty`，只按 `window_contribution_score` 排序，模型很容易出现一种常见偏置：

- 反复保留同一个动作阶段的大量相似窗口
- 把看似“没那么高分”但实际上覆盖关键状态迁移的窗口漏掉

对机器人 manipulation 数据来说，这会直接导致训练分布塌缩到某几个高频动作片段，比如：

- 反复靠近目标但不覆盖真正接触瞬间
- 反复覆盖静止持有段，但不覆盖姿态修正段

所以 `coverage_novelty` 本质上不是“为了好看而做 diversity”，而是在强行阻止数据优选退化成局部模式坍缩。

### 7.7.2 为什么优先用 `state_mean` 而不是只用时间位置

代码会优先用窗口内 `observation.state` 的均值作为 feature，只有状态不可用时才退化到 `temporal_center`。

这个选择非常合理，因为：

- 时间位置只知道“这段在前面还是后面”
- `state_mean` 更接近“机器人此时在什么物理状态区域”

对于同一个任务，不同 demonstration 里的“关键接触段”未必出现在完全相同的时间索引上，但它们往往会落在相近的状态区域。因此用 `state_mean` 做 coverage，更接近具身任务的真实几何结构。

### 7.8 自动停止条件

本次使用 `budget_mode=auto_marginal`，因此不会预先规定“保留多少窗口”，而是边选边停。

停止逻辑是：

1. 至少已经选了 `min_windows_per_episode=1` 个窗口。
2. 当前 episode 的时间覆盖率已经达到 `min_coverage_ratio=0.5`。
3. 当前最佳边际收益低于自适应阈值。

自适应阈值为：

```text
adaptive_threshold =
    max(0.15, 0.5 * first_selected_gain)
```

其中：

- `0.15` 来自 `MARGINAL_GAIN_THRESHOLD`
- `0.5` 来自 `RELATIVE_GAIN_THRESHOLD`

一个细节是：代码是在“选中本轮最佳窗口之后”才检查停止条件，所以最后一个被选中的窗口有可能已经处于收益下降区间，但它仍会被保留。这是当前实现的真实行为。

### 7.8.1 为什么停止条件要同时看 coverage 和 gain

如果只看 `gain`，容易过早停止，结果是：

- 只保留少数几个高分窗口
- 覆盖不足
- 训练时 segment mask 过稀，模型只见到局部动作片段，时序上下文被截断

如果只看 `coverage`，又容易过度保留，结果是：

- 大量低价值窗口被一并带回训练
- joint episode-window 选择失去压缩意义

所以当前设计强制要求两件事同时成立：

1. 覆盖已经足够
2. 继续增加窗口的边际收益已经明显下降

这本质上是在逼近一个更符合工程直觉的停止点：

```text
“已经保住了足够多的时序骨架，再往里加窗口就主要是在重复。”
```

### 7.8.2 当前实现的一个边界行为

代码逻辑是：

1. 先选出当前最佳窗口
2. 把它标记为 keep
3. 再检查是否应该停止

因此最后一个被保留的窗口，可能正是“刚刚进入收益衰减区”的那个窗口。

这不是错误，但你在分析 `window_scores.csv` 时要知道：

- 最后一个 `keep_rank` 较大的窗口，不一定是高价值窗口
- 它有时更像“满足 coverage 后的缓冲边界窗口”

如果后面做 ablation，可以专门测试“停止判断提前到选窗前”这一变体。

### 7.9 Window 过滤输出到底如何影响训练

`joint_episode_window_selection.npz` 不只是分析报告，而是训练分布的直接控制器。

它会影响两层采样：

1. episode 层
   `episode_keep_mask` 决定哪些 episode 还留在训练集。
2. frame 层
   `window_keep` 和 `window_contribution` 决定每帧是否还能被采到，以及被采到的相对概率。

也就是说，window 过滤不是简单地“给每个窗口打标签”，而是在真实改变 ACT 的经验分布：

```text
哪些轨迹能被看到
+
轨迹中的哪些时间段更容易被看到
```

这也是为什么 window 级过滤的设计必须足够稳健。它一旦失真，训练分布会被直接拉偏。

## 8. Selection 阶段的输出产物

`filter_joint_episode_window.py` 会在 selection 目录下输出：

- `kept_episodes.txt`
- `dropped_episodes.txt`
- `episode_scores.csv`
- `window_scores.csv`
- `selection_summary.json`
- `joint_episode_window_selection.npz`

其中最关键的是 `joint_episode_window_selection.npz`，它至少包含：

- `episode_keep_mask`
- `episode_joint_scores`
- `window_episode`
- `window_start`
- `window_end`
- `window_keep`
- `window_episode_score`
- `window_quality_score`
- `window_local_value`
- `window_contribution`

## 9. 训练阶段如何消费筛选结果

这一步最容易被误解。`train_act_cupid_reweight_joint.py` 并不是简单把 selection 文件当作统计报告，而是会真正把它合并进训练采样逻辑。

### 9.1 episode 过滤是谁说了算

命令里传了：

- `--selection-npz`
- `--selection-override-episodes`
- `--joint-drop-quantile 0`
- `--drop-weight-quantile 0`
- `--drop-bad-ratio 1.1`

这组组合的真实效果是：

1. 训练脚本内部的 joint quantile 过滤被关闭。
2. weight quantile 过滤也被关闭。
3. `drop-bad-ratio=1.1` 几乎不可能再删 episode，因为坏窗口占比最多只有 `1.0`。
4. 因为有 `selection-override-episodes`，最终 episode 保留集合直接由 `selection.npz` 的 `episode_keep_mask` 覆盖。

也就是说，本次训练阶段的 episode 过滤主导权已经完全交给 joint episode-window 筛选结果。

### 9.2 窗口结果怎样转成 frame 级 mask / weight

训练脚本会先把 `selection.npz` 合并进 metadata：

- `window_keep=0` 的窗口变成 `window_drop_mask=1`
- `window_contribution` 被归一化后写成 `window_weights`

归一化逻辑是：

```text
weights = max(window_contribution, 0)
weights = weights / mean(weights_of_kept_windows)
weights[dropped] = 0
```

然后再把每个 episode 的所有 kept window 投影到 frame 级：

```text
frame_weight[t] = 平均覆盖到该帧的 kept window 权重
keep_mask[t]    = 该帧是否被至少一个 kept window 覆盖
```

### 9.3 最终采样权重的组成

本次命令使用：

- `--weight-mode influence`
- `--use-weighted-sampler`
- `--segment-mask`
- `--segment-weight`

因此最终训练样本权重大致是：

```text
sample_weight(frame) =
    normalized_episode_influence(ep)
    * frame_weight(frame)
```

并且如果某帧不在任何 kept window 内，则：

```text
sample_weight(frame) = 0
```

这意味着训练分布被分成两层：

1. episode 层：高 influence episode 更常被采到。
2. frame / segment 层：episode 内只有被选中的窗口区域拥有正采样概率，而且高贡献窗口会进一步放大。

## 10. 这组参数的实现思路与设计意图

从数据优选角度看，这组参数背后的策略是相当清晰的。

### 10.1 先粗后细

- `EPISODE_DROP_RATIO=0.10`
  只先去掉最差 10% episode，属于温和粗筛。
- 后续真正的压缩动作主要发生在 window 级。

这比“一上来按 episode 丢 30%-50%”更稳，因为不会过早把“整体一般但局部关键”的 demonstration 整段删除。

### 10.2 本次 window 几乎是不重叠分块

- `WINDOW_SIZE=50`
- `WINDOW_STRIDE=50`

这意味着每个窗口更像连续的动作 chunk，而不是高度重叠的滑窗。

优点：

- 计算量小
- 重复窗口少
- segment mask 更稀疏、更干净

代价：

- 对边界关键片段不够柔和
- 一个关键动作如果刚好跨块，可能被局部质量稀释

### 10.3 0.4 / 0.4 / 0.2 的含义

本次局部值融合为：

- 40% window influence
- 40% window quality
- 20% episode global prior

这是一种典型的“局部主导，全局兜底”配置。

优点：

- 不会只看 global episode label
- 也不会只看短窗平滑度
- 让真正对模型泛化有帮助、同时局部执行稳定的片段优先进入训练

### 10.4 覆盖阈值和收益阈值的配合

- `MIN_COVERAGE_RATIO=0.5`
  要求每个保留 episode 至少覆盖一半时间轴后，才允许因收益下降而停止。
- `MARGINAL_GAIN_THRESHOLD=0.15`
  绝对收益底线。
- `RELATIVE_GAIN_THRESHOLD=0.5`
  相对首个最佳窗口的收益比例底线。

这组组合的含义是：

- 不允许只拿最强的一个窗口就停
- 但也不允许把大量低收益冗余窗口继续塞回训练

## 11. 本次运行的隐含默认值与潜在风险

### 11.1 `RECOMPUTE_SCORE=1` 代价较高

这会强制重做 influence 四阶段，通常是全流程中最耗时、最吃 GPU 的部分。即使之前已有同参数 `score.pkl`，本次也不会复用。

### 11.2 Influence 阶段的默认 batch 非常保守

本次没有显式设置：

- `batch_size`
- `hvp_batch_size`
- `proj_dim`
- `num_workers`

因此 influence 默认是：

- `batch_size=1`
- `hvp_batch_size=4`
- `proj_dim=4096`
- `num_workers=0`

这是偏稳定、偏保守的配置，适合先跑通，但吞吐未必最优。

### 11.3 `trajectory_eval` 依赖 parquet 中的状态字段质量

如果某些 episode 缺失 `observation.state`，那么：

- terminal/progress/monotonic/backtrack/stall/state_smooth 的可靠性会下降
- coverage feature 也可能回退为 `temporal_center`

在这种情况下，window 选择会更偏“时间位置均匀覆盖”，而不是“状态空间均匀覆盖”。

### 11.4 训练阶段不是“只训练 kept episode”

更准确地说，它是：

1. 只保留 `selection.npz` 指定的 kept episode。
2. 这些 episode 内又只保留 kept window 覆盖到的帧拥有正采样概率。

所以这是 episode 过滤 + segment 采样重加权的组合，不是简单 dataset subset。

## 12. 建议重点检查的结果文件

如果你要判断这次优选是否有效，优先看下面这些文件：

1. `selection/selection_summary.json`
   看最终保留了多少 episode、多少 window。
2. `selection/episode_scores.csv`
   看 influence / quality / joint score 的排序是否符合直觉。
3. `selection/window_scores.csv`
   看 kept window 是否集中在真正关键的动作阶段。
4. `train/cupid_reweight_report.json` 或训练日志
   看 sample weight 分布、loss 曲线、checkpoint 保存和 eval 指标。
5. `score/score_episode_window_ws50_st50.pkl`
   必要时回查 metadata 中的 `window_influence_scores` 与 `episode_rewards`。

## 13. 结合 `beat_block_hammer` 任务的动作阶段解释

下面这部分是基于本数据集本地 metadata 做的任务绑定解释。

已确认的事实有两点：

1. `meta/episodes.jsonl` 中的任务描述都是“抓起锤子并击打方块”的变体，且存在左臂和右臂两种执行方式。
2. `meta/info.json` 显示数据集 `fps=15`，总共 500 条 episode、33459 帧，平均每条约 66.9 帧；样例 episode 长度大多在 `63-69` 帧之间。

这两个事实会直接改变你对 `window_size=50, window_stride=50` 的理解。

### 13.1 这个任务下，一个 episode 大致包含哪些动作阶段

对 `beat_block_hammer` 这类任务，单条 demonstration 通常可以粗分为以下阶段：

1. 接近与定位
   机械臂朝锤子移动，调整 wrist / gripper 姿态。
2. 抓取锤子
   gripper 闭合，完成抓取接触。
3. 提起并转运
   锤子被带离桌面，机械臂朝方块方向运动。
4. 对齐与预击打
   hammer head 朝目标 block 对位，姿态修正较多。
5. 击打接触
   末端速度、加速度和关节变化通常达到峰值。
6. 击打后稳定 / 回撤
   动作逐渐减速，系统恢复稳定。

这里要注意一个事实：由于该数据集有左臂执行和右臂执行两种版本，所以“关键动作阶段”的时间语义相同，但具体状态轨迹在 joint space 上会呈现明显双模态。

这也是为什么：

- `trajectory_eval` 里不能只看单一终态距离
- `coverage_feature=state_mean` 比只看时间索引更重要

### 13.2 在你这组参数下，绝大多数 episode 实际上只有 2 个候选窗口

因为：

- 大多数 episode 长度约 `64-69`
- `window_size=50`
- `window_stride=50`

结合当前 `_window_ranges()` 的实现，长度在这个区间时，通常只会得到 2 个窗口：

1. `window A = [0, 50)`
2. `window B = [length-50, length)`

例如长度 `66` 的 episode，窗口通常是：

```text
[0, 50)
[16, 66)
```

这意味着：

- 它们不是两个完全独立的不重叠阶段
- 而是两个 **大幅重叠** 的长窗口
- 重叠区域通常在 30 多帧

这个结论很重要。因为在 `beat_block_hammer` 上，当前 window 选择并不是在做很细的“接近 / 抓取 / 击打”多段切分，而更像是在做：

1. 偏前半程的大窗口
2. 偏后半程的大窗口

### 13.3 这两个窗口在物理意义上通常分别代表什么

在这个任务长度分布下，可以把两个主窗口近似理解成：

#### 窗口 A：前中段主导窗口

通常覆盖：

- 接近锤子
- 姿态对齐
- 抓取闭合
- 初步提起
- 一部分转运

如果某条 demonstration 在这部分有问题，常见表现是：

- 接近路径抖动大
- wrist 姿态反复修正
- 抓取前来回试探
- 抓取后锤子没有稳定抬起

此时这个窗口的：

- `window_quality_score` 往往会下降
- `window_episode_score` 也可能下降，因为这种局部会让后续击打成功率变差

#### 窗口 B：后中段主导窗口

通常覆盖：

- 末段转运
- 对齐 block
- 击打瞬间
- 接触后的稳定或回撤

这是最容易出现任务关键差异的窗口，因为：

- 真正决定“有没有把 block 打到位”的动作多发生在这里
- 击打接触会带来更大的 state / action 变化
- 好 demonstration 和坏 demonstration 在这段通常分化最大

因此在很多情况下：

- `window_episode_score` 的区分度会主要集中在窗口 B
- `window_quality_score` 也容易在这里两极分化，因为有效击打和无效乱挥在抖动统计上差异很大

### 13.4 为什么当前配置容易偏爱“后半窗口”

对 `beat_block_hammer` 来说，后半窗口往往更容易被选中，原因通常有 3 个：

1. 它更接近真正的任务成败时刻
   击打接触和 block 位移相关信息主要在后段。
2. rollout influence 更敏感
   如果当前预训练 ACT 的瓶颈在“最后对齐与击打”，那么 influence 会更偏向后段。
3. `trajectory_eval` 的 progress / terminal 结构分量通常也会在后段体现差异
   因为真正完成任务的轨迹，其终态与进展信息要到后段才充分展开。

但这里有一个反直觉点：

- 后半窗口动作变化更大，不一定天然有更高 `window_quality_score`
- 如果 demonstration 的击打阶段存在高频抖动、无效回摆，后半窗口也可能被 quality 压下来

所以本方法不是简单“偏爱击打段”，而是更偏爱：

```text
既能推动任务成功、又不是乱挥的击打相关片段
```

### 13.5 `coverage_novelty` 在这个任务里主要防止什么

在 `beat_block_hammer` 上，`coverage_novelty` 的主要价值不是“让时间段均匀”，而是防止以下几类模式坍缩：

1. 只保留大量右臂版本，忽略左臂版本
2. 只保留某一种抓锤姿态，忽略另一种可行接近路径
3. 只保留“击打前刚好对齐”的状态，忽略“抓取后转运”的必要过渡状态

因为本数据集 `observation.state` 是 14 维双臂 joint state，所以 `state_mean` feature 会自然把左臂执行和右臂执行区分开，也会把不同姿态簇分开。

这意味着当前 coverage 机制在这个任务上的真实作用更接近：

```text
尽量覆盖不同的可行操控姿态簇，而不只是覆盖不同时间位置
```

### 13.6 看 `window_scores.csv` 时，如何把数值映射回动作阶段

对这个任务，推荐这样读表：

1. 如果某个 episode 的后半窗口 `window_episode_score` 明显高于前半窗口
   往往说明这条 demonstration 的“击打相关局部”对 rollout 更有帮助。
2. 如果后半窗口 influence 高但 `window_quality_score` 很低
   往往说明该片段虽然和任务成败强相关，但执行抖动大，可能是“高价值但高风险片段”。
3. 如果前半窗口 `window_quality_score` 高、后半窗口低
   这通常对应“抓取和转运很稳，但击打执行不干净”。
4. 如果两个窗口都低
   通常是整条轨迹都不理想，episode 级也大概率会先被粗筛掉。
5. 如果两个窗口都高，但只保留了其中一个
   往往是 coverage / 边际收益触发了停止，而不是另一个窗口绝对差。

### 13.7 这组窗口参数在本任务上的局限

对 `beat_block_hammer` 而言，`window_size=50` 在 15 FPS 下约等于 `3.33s`，而单条 episode 只有约 `4.2-4.6s`。

这意味着当前 window 粒度偏粗，会带来几个局限：

1. 难以把“抓取段”和“击打段”完全分离。
2. 击打瞬间会被大量上下文帧稀释。
3. 两个候选窗口高度重叠，greedy 选择的自由度并不高。

因此如果后续你想更精细地做 `beat_block_hammer` 数据优选，一个很自然的 ablation 方向是：

- 降低 `window_size`，例如 `24/32/40`
- 保持一定 overlap，例如 `stride=12/16/20`

这样更容易把以下局部拆开：

- 接近抓取段
- 转运对齐段
- 击打接触段
- 回撤稳定段

也更容易验证到底是哪个物理子阶段在决定 ACT 的泛化提升。

## 14. 一段贴近代码的伪代码总结

```python
# 1) 计算 influence
score_pkl = run_cupid_act_influence(
    pretrained_dir=PRETRAINED_DIR,
    train_dataset=DATASET_ROOT,
    eval_rollouts=EVAL_ROLLOUT_DIR,
    window_size=50,
    window_stride=50,
)

# 2) 计算 episode 质量
episode_quality = trajectory_eval(DATASET_ROOT)

# 3) 先做 episode 粗筛
episode_joint = 0.5 * norm(episode_influence) + 0.5 * norm(episode_quality)
keep_episode = drop_bottom_ratio(episode_joint, ratio=0.10)

# 4) 对每个 kept episode 生成候选窗口
for episode in kept_episodes:
    windows = build_windows(size=50, stride=50)

    # 局部值 = influence + quality + global prior
    for w in windows:
        w.value = (
            0.4 * w.window_influence
            + 0.4 * w.window_quality
            + 0.2 * episode_joint[episode]
        )

    # greedy 选窗，兼顾贡献和覆盖
    selected = []
    while True:
        best = argmax_w(0.7 * w.value + 0.3 * novelty(w, selected))
        selected.append(best)
        if coverage(selected) >= 0.5 and gain(best) < max(0.15, 0.5 * first_gain):
            break

# 5) 训练 ACT
# episode 过滤交给 selection_keep_mask
# frame 采样概率 = episode influence * kept-window frame weight
train_act_with_segment_reweight(selection_npz, score_pkl)
```

## 15. 一句话结论

这条命令本质上是在做一套“先用 rollout influence 找对模型真正有帮助的轨迹，再用 trajectory quality 和 coverage-aware window selection 把轨迹内部高价值动作片段提纯出来，最后把结果落到 ACT 的 frame 级采样分布上”的联合数据优选训练流水线。
