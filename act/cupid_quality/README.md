# CUPID-QUALITY

这个目录提供一个独立的 `cupid_quality` 方法实现。

它直接从已经计算好的 state-action pair 分数
`Ψa-inf((s',a'),(s,a))` 出发，按下面的定义计算 demo 级质量：

```text
Ψqual(ξ; Dτ) =
  1 / m * Σ_{τ ∈ Dτ}
    [ max_{(s',a') ∈ τ} min_{(s,a) ∈ ξ} Ψa-inf((s',a'),(s,a))
    - min_{(s',a') ∈ τ} max_{(s,a) ∈ ξ} Ψa-inf((s',a'),(s,a)) ]
```

其中：

- `ξ` 是一个训练 demo 轨迹
- `τ` 是一个 rollout 轨迹
- `Ψa-inf((s',a'),(s,a))` 是 state-action 对之间的分数
- 如果使用 `--normalize-by-trajectory-length`，每条 rollout 的贡献会除以其轨迹长度 `m`

## 输入格式

`compute_cupid_quality.py` 支持三种输入：

1. `npz`
2. `pkl`
3. `csv`

### 1) NPZ / PKL

需要至少包含以下键：

- `pair_scores`: shape `[num_eval_state_actions, num_train_state_actions]`
- `eval_episode_ids`: shape `[num_eval_state_actions]`
- `train_episode_ids`: shape `[num_train_state_actions]`

可选键：

- `eval_step_ids`
- `train_step_ids`

也支持这些别名：

- `pair_scores` 的别名：`psi_a_inf` / `scores` / `influence_scores`
- `eval_episode_ids` 的别名：`eval_episode_index` / `eval_episode`
- `train_episode_ids` 的别名：`train_episode_index` / `train_episode`

### 2) CSV

需要以下列：

- `eval_episode`
- `eval_step`
- `train_episode`
- `train_step`
- `psi_a_inf`

也兼容这些分数字段名：

- `score`
- `influence`
- `pair_score`

## 运行方式

pair-score 文件输入：

```bash
python /workspace/liushuo/cupid/act/cupid_quality/compute_cupid_quality.py \
  --input-path /path/to/pair_scores.npz \
  --output-dir /workspace/liushuo/cupid/act/cupid_quality/outputs/can_mh \
  --normalize-by-trajectory-length \
  --normalizer quantile \
  --quantile-low 0.05 \
  --quantile-high 0.95
```

如果你手头只有标准的 `score.pkl` 和 stage cache，则应该使用：

```bash
python /workspace/liushuo/cupid/act/cupid_quality/compute_cupid_quality_from_score.py \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --stage1-train-projections /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.stage1_train_projections.npy \
  --stage2-return-gradient /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.stage2_return_gradient.pt \
  --stage3-inverse-hvp /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.stage3_inverse_hvp.pt \
  --output-dir /workspace/liushuo/cupid/act/cupid_quality/outputs/can_mh \
  --normalize-by-trajectory-length \
  --normalizer quantile \
  --quantile-low 0.05 \
  --quantile-high 0.95
```

注意：

- `stage2_return_gradient.pt` 和 `stage3_inverse_hvp.pt` 是 policy-level 聚合缓存，不能直接推出精确的 `Ψa-inf`
- `compute_cupid_quality_from_score.py` 会复用 `score.pkl` 里的配置和 `stage1` 投影缓存，逐步重建 quality 所需的 rollout-vs-demo 聚合

## 输出文件

- `quality_episodes.csv`
- `quality_report.json`
- `quality_keep.txt`
- `quality_drop.txt`
- `quality_top_50.txt`
- `quality_bottom_50.txt`

其中 `quality_episodes.csv` 字段为：

- `episode`
- `quality_score`
- `quality_raw`
- `rollout_count`
- `rank_desc`
- `drop`

`quality_score` 是把 `quality_raw` 映射到 `[0, 1]` 之后的结果，便于直接作为
下游训练脚本的质量 CSV 输入。

## 一键筛选和再训练

```bash
python /workspace/liushuo/cupid/act/cupid_quality/run_cupid_quality_train.py \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --stage1-train-projections /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.stage1_train_projections.npy \
  --stage2-return-gradient /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.stage2_return_gradient.pt \
  --stage3-inverse-hvp /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.stage3_inverse_hvp.pt \
  --quality-output-dir /workspace/liushuo/cupid/act/cupid_quality/outputs/can_mh \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --train-output-dir /workspace/liushuo/cupid/3_model/can_mh_cupid_quality \
  --job-name can_mh_cupid_quality \
  --normalize-by-trajectory-length \
  --joint-drop-quantile 0.2 \
  --joint-quality-power 1.0 \
  --joint-influence-power 0.0 \
  --max-steps 100000
```

如果想做 quality + influence 联合筛选和再训练，把 `--joint-influence-power`
设为 `1.0`，并保留 `--score-pkl`：

```bash
python /workspace/liushuo/cupid/act/cupid_quality/run_cupid_quality_train.py \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --stage1-train-projections /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.stage1_train_projections.npy \
  --quality-output-dir /workspace/liushuo/cupid/act/cupid_quality/outputs/can_mh_joint \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --train-output-dir /workspace/liushuo/cupid/3_model/can_mh_cupid_quality_joint \
  --job-name can_mh_cupid_quality_joint \
  --normalize-by-trajectory-length \
  --joint-drop-quantile 0.2 \
  --joint-quality-power 1.0 \
  --joint-influence-power 1.0 \
  --weight-mode influence \
  --use-weighted-sampler
```
