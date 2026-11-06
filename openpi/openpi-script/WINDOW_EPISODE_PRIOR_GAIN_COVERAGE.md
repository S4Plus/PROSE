# Window Episode-Prior Gain+Coverage

这套脚本实现的是一个独立于原有 `keep_ratio` 手动网格搜索的启发式筛选方案，核心目标是自动确定数据保留比例。

## 目录

- `openpi-script/select_window_joint_episode_prior_gain_coverage.py`
  只做筛选、产物落盘、`norm_stats` 重算，不启动训练。
- `openpi-script/retrain_window_joint_episode_prior_gain_coverage.py`
  先用同样的启发式确定保留比例，再直接启动训练。
- `openpi-script/window_episode_prior_gain_coverage_common.py`
  两个入口共享的通用逻辑。

## 启发式规则

设按最终窗口分数从高到低排序后为：

`s1 >= s2 >= ... >= sN`

定义：

- `gain(i) = mean(s1..si) - mean(s1..s{i+delta})`
- `coverage(i) = kept_episodes(i) / total_episodes`

其中：

- `gain(i)` 表示当保留窗口数从 `i` 扩大到 `i + delta` 时，前缀平均质量下降了多少。
- `coverage(i)` 表示在当前保留比例下，最终被选中的窗口覆盖了多少 episode。

脚本的决策规则是：

1. 对每个候选保留数 `i = 1..N`，先按当前 `episode_budget_rule` 和 `min_windows_per_episode` 跑真实的窗口选择逻辑。
2. 用真实保留集合的平均分去计算 `gain(i)`，而不是用全局排序前缀替代。
3. 统计该选择结果的：
   - `covered_episode_ratio`
   - `episodes_meeting_min_windows_ratio`
4. 选择第一个同时满足以下条件的 `i`：
   - `covered_episode_ratio >= min_episode_coverage_ratio`
   - `episodes_meeting_min_windows_ratio >= min_episode_coverage_ratio`
   - `relative_gain(i) <= max_relative_gain`

如果没有任何 `i` 同时满足三项约束，则退化为：

- 先找第一个满足覆盖约束的 `i`
- 如果还没有，则退化为全保留

## 为什么这样做

这比固定扫 `0.9 / 0.8 / 0.6` 更贴近你当前任务的目标：

- `gain` 约束控制“继续放宽筛选，平均质量还能下降多少”
- `coverage` 约束控制“不要为了高分只保留少数 episode”
- `min_windows_per_episode` 约束控制“单个 episode 至少要保留多少窗口”

因此它在“质量”与“覆盖面”之间做了一个自动平衡。

## 输入分数

脚本不会直接使用原始 influence 分数作为筛选依据，而是复用现有的 `window + episode prior` 逻辑，先得到：

- `window_joint_scores`
- `episode_joint_scores`
- `final_window_scores`

启发式搜索只负责自动找 `keep_ratio`，并不改变原有打分公式。

## 常用命令

### 1. 只做筛选

```bash
uv run python openpi-script/select_window_joint_episode_prior_gain_coverage.py \
  --score-pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/influence_grass_window.pkl \
  --train-config pi05_can_mh_lora \
  --dataset-dir /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --joint-rule sum \
  --window-score-weight 0.8 \
  --episode-prior-weight 0.2 \
  --adaptive-prior-mode window_std \
  --adaptive-prior-strength 0.5 \
  --interaction-weight 0.1 \
  --fusion-normalizer minmax \
  --episode-quality-aggregation mean \
  --min-windows-per-episode 1 \
  --episode-budget-rule episode_prior \
  --quality-source rank_fusion \
  --min-episode-coverage-ratio 0.9 \
  --max-relative-gain 0.02 \
  --overwrite
```

### 2. 筛选后直接重训练

```bash
uv run python openpi-script/retrain_window_joint_episode_prior_gain_coverage.py \
  --score-pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/influence_grass_window.pkl \
  --train-config pi05_can_mh_lora \
  --dataset-dir /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --joint-rule sum \
  --window-score-weight 0.8 \
  --episode-prior-weight 0.2 \
  --adaptive-prior-mode window_std \
  --adaptive-prior-strength 0.5 \
  --interaction-weight 0.1 \
  --fusion-normalizer minmax \
  --episode-quality-aggregation mean \
  --min-windows-per-episode 1 \
  --episode-budget-rule episode_prior \
  --quality-source rank_fusion \
  --min-episode-coverage-ratio 0.9 \
  --max-relative-gain 0.02 \
  --num-train-steps 10000 \
  --overwrite
```

### 3. 只准备数据，不启动训练

```bash
uv run python openpi-script/retrain_window_joint_episode_prior_gain_coverage.py \
  --score-pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/influence_grass_window.pkl \
  --train-config pi05_can_mh_lora \
  --dataset-dir /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --prepare-only \
  --overwrite
```

## 关键参数

### 启发式参数

- `--min-episode-coverage-ratio`
  至少要覆盖多少比例的 episode。

- `--max-relative-gain`
  允许的最大相对边际收益。越小越保守，筛选越严格。

`gain` 的窗口跨度 `delta` 不再暴露成 CLI 超参数，脚本内部固定按窗口总数的 `2%` 自动确定。

### 原有筛选参数

- `--window-score-weight`
- `--episode-prior-weight`
- `--adaptive-prior-mode`
- `--adaptive-prior-strength`
- `--interaction-weight`
- `--fusion-normalizer`
- `--min-windows-per-episode`
- `--episode-budget-rule`
- `--quality-source`

这些参数仍然决定最终打分和真实选择逻辑。
启发式只是自动找保留比例，不替代这些参数。

## 输出文件

在 `output_dir` 下会生成：

- `kept_sample_indices.npy`
  最终保留的 sample 索引。

- `joint_window_episode_prior_scores.npz`
  选择前后的核心分数和索引。

- `joint_window_episode_prior_summary.json`
  筛选摘要，包含最终 `keep_ratio` 和启发式元信息。

- `gain_coverage_selection_report.json`
  启发式推荐结果摘要。

- `gain_coverage_selection_curves.npz`
  完整的 `gain`、coverage、满足最小窗口数的 episode 比例、真实保留集平均分曲线。

- `window_quality_components.npz`
  窗口质量打分分量。

- `norm_stats.json`
  用筛选后样本重算的归一化统计量。

## 调参建议

### 想更保守

- 降低 `max-relative-gain`
- 提高 `min-episode-coverage-ratio`
### 想更激进

- 提高 `max-relative-gain`
- 降低 `min-episode-coverage-ratio`

## 约束不可满足时

如果设置了较大的 `min_windows_per_episode`，但某些 episode 在原始窗口集合里本来就没有那么多窗口，脚本会直接报错，提示该约束不可行，而不是悄悄退化成错误结果。
