# ACT 两阶段 Episode-Window 联合筛选方法

本文档说明 `myscripts/filter_joint_episode_window.py` 与
`myscripts/run_filter_and_train_joint_episode_window.sh`
对应的方法、筛选准则、输出产物和训练接入方式。

## 1. 方法目标

目标是在 ACT 数据筛选里同时利用：

1. episode 级全局先验
2. window 级局部价值
3. 覆盖保持与边际收益自动预算

与只按 episode 做硬筛相比，这个方法会先去掉全局低质量 episode，
再在保留 episode 内做更细粒度的 window 选择。

## 2. 方法流程

### 2.1 Episode 级联合过滤

对每个 episode 计算联合分：

```text
episode_score_i =
    a * norm(influence_i) + b * norm(quality_i)
```

默认：

- `a = 0.5`
- `b = 0.5`
- `episode_drop_ratio = 0.10`

即先删除联合分最低的 10% episode。

其中：

- `influence_i` 来自 `score.pkl` 中的 episode influence
- `quality_i` 可来自 `quality_episodes.csv`，也可在线计算
- `norm(.)` 为 min-max 归一化到 `[0, 1]`

### 2.2 Window 级局部评分

对保留的每个 episode，按滑窗生成候选 window：

- `window_size` 默认 `50`
- `window_stride` 默认 `10`

每个 window 的局部价值 `window_value` 默认由局部运动平滑性构成：

- action jitter 越小越好
- state jitter 越小越好

两个分量分别归一化后线性融合，得到 `window_value_score`。

### 2.3 Episode 先验与 Window 局部价值融合

对每个 window 再计算贡献分：

```text
window_contribution =
    alpha * episode_prior + beta * window_local_value
```

默认：

- `alpha = 0.5`
- `beta = 0.5`

其中：

- `episode_prior` 就是该 window 所属 episode 的 `episode_score`
- `window_local_value` 是该 window 的局部价值分

## 3. Window 保留依据

window 不再按固定比例删除，而是按边际收益贪心选择。

每一步对候选 window 计算：

```text
marginal_gain =
    w1 * contribution_score + w2 * coverage_novelty
```

默认：

- `w1 = 0.7`
- `w2 = 0.3`

其中：

- `contribution_score`：该 window 的联合贡献度
- `coverage_novelty`：该 window 相对已选 window 的新增覆盖价值

### 3.1 覆盖项的含义

覆盖项用于防止只保留大量相似 window。

默认优先使用：

- `state_mean` 特征：window 内 `observation.state` 的均值向量

如果状态不可用，则退化为：

- `temporal_center`：window 在轨迹时间轴上的位置

随后用“候选 window 到已选 window 集合的最小距离”表示其新颖度。

### 3.2 自动预算停止

默认使用 `auto_marginal` 模式，而不是手工指定 window 保留比例。

停止条件是：

1. 已满足最少保留窗口数 `min_windows_per_episode`
2. 已满足最小覆盖率 `min_coverage_ratio`
3. 当前最佳边际收益低于阈值

阈值采用双重约束：

```text
stop_threshold =
    max(marginal_gain_threshold,
        relative_gain_threshold * first_selected_gain)
```

默认：

- `marginal_gain_threshold = 0.15`
- `relative_gain_threshold = 0.5`
- `min_coverage_ratio = 0.4`

因此，window 保留比例是自动决定的，不需要手工给定最终 keep ratio。

## 4. 输出产物

筛选脚本默认输出到一个目录，核心产物包括：

- `selection_summary.json`
- `episode_scores.csv`
- `window_scores.csv`
- `kept_episodes.txt`
- `dropped_episodes.txt`
- `joint_episode_window_selection.npz`

其中最关键的是：

- `joint_episode_window_selection.npz`

它包含：

- `episode_keep_mask`
- `episode_joint_scores`
- `window_episode`
- `window_start`
- `window_end`
- `window_keep`
- `window_contribution`

## 5. 训练阶段如何使用

训练脚本 `train_act_cupid_reweight_joint.py` 已支持：

- `--selection-npz`
- `--selection-override-episodes`

训练时会把筛选结果转换成：

- episode 级保留集合
- frame 级 `segment_mask`
- frame 级 `segment_weight`

具体行为：

1. `episode_keep_mask` 控制哪些 episode 进入训练集
2. `window_keep=0` 的窗口会变成 dropped window
3. `window_contribution` 会转换成窗口权重，再聚合成 frame 级采样权重
4. 若开启 `--segment-mask --segment-weight`，训练采样器只对保留窗口覆盖到的帧赋非零采样权重

## 6. 直接执行方式

### 6.1 只做筛选

```bash
python /workspace/liushuo/cupid/act/myscripts/filter_joint_episode_window.py \
  --score-pkl /path/to/score.pkl \
  --dataset-root /path/to/dataset \
  --quality-source quality_csv \
  --quality-score-csv /path/to/quality_episodes.csv \
  --output-dir /path/to/output_selection
```

### 6.2 一键筛选并训练

```bash
cd /workspace/liushuo/cupid/act

DATASET_ROOT=/path/to/dataset \
DATASET_REPO_ID=ls/your_dataset \
SCORE_PKL=/path/to/score.pkl \
QUALITY_CSV=/path/to/quality_episodes.csv \
OUTPUT_BASE=/path/to/output_root \
bash /workspace/liushuo/cupid/act/myscripts/run_filter_and_train_joint_episode_window.sh
```

## 7. 当前默认策略总结

默认配置可以概括为：

1. 先按 `0.5 * norm(influence) + 0.5 * norm(quality)` 去掉最低 10% episode
2. 再在剩余 episode 内，按 `贡献度 + 覆盖新颖度` 的边际收益做 greedy window 选择
3. 一旦覆盖足够且继续保留带来的收益明显下降，就自动停止
4. 将筛选结果直接接到训练采样权重中，而不是只做离线分析

## 8. 适用场景

这个方法更适合下面两类情况：

- episode 级筛选太粗，容易把“整体一般但局部关键”的数据全部删掉
- 数据高度重复，只按 influence 或 quality 排序会保留过多相似片段

它的核心优势不是单纯“删更多数据”，而是：

- 先去掉全局低价值 episode
- 再保留每个好 episode 中最有训练价值、且覆盖更好的局部片段
