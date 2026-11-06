# SCIZOR / Demo-SCORE Workflow

这个目录把 `SCIZOR` 和 `Demo-SCORE` 的筛选结果接到 `OpenVLA` 的重训练链路里。

两条 workflow 的共同逻辑是：

1. 读取外部方法产出的筛选结果或样本分数。
2. 转成 `OpenVLA` 可直接消费的 `kept_sample_indices.npy`。
3. 基于保留子集重算 `norm_stats.json`。
4. 用 `sample_indices_path` 启动 `OpenVLA` 重训练。

## 目录说明

- `retrain_scizor_openvla.py`
  将 `SCIZOR` 的 `window_drop_mask` 映射成 `OpenVLA` 训练样本子集。
- `retrain_demo_score_openvla.py`
  将 `Demo-SCORE` 的逐样本分数聚合成 episode 分数，再按比例筛选 episode 后重训练。
- `run_can_mh_scizor.sh`
- `run_square_mh_scizor.sh`
- `run_can_mh_demo_score.sh`
- `run_square_mh_demo_score.sh`

## SCIZOR 复现

`SCIZOR` 默认读取逐帧 `window_drop_mask`，这比只按 episode 级 `influence_scores` 裁剪更接近原始方法。

示例：

```bash
bash /workspace/liushuo/cupid/openvla-oft/workflow/scizor_demo_score/run_can_mh_scizor.sh \
  --overwrite \
  --exp-name can_mh_scizor \
  --num-train-steps 10000
```

只准备子集和 `norm_stats`，不启动训练：

```bash
bash /workspace/liushuo/cupid/openvla-oft/workflow/scizor_demo_score/run_can_mh_scizor.sh \
  --prepare-only \
  --overwrite
```

## Demo-SCORE 复现

`Demo-SCORE` 这条 workflow 不在这里重新训练 classifier，而是直接消费已经算好的训练集逐样本分数。

支持输入：

- `train_set_demo_scores.mmap`
- `.npy`
- `.npz`
- `.json`
- `.csv`

默认策略是：

- 先把逐样本分数按 episode 做 `mean` 聚合。
- 再保留分数最高的 `90%` episode。

示例：

```bash
SCORE_PATH=/path/to/train_set_demo_scores.mmap \
bash /workspace/liushuo/cupid/openvla-oft/workflow/scizor_demo_score/run_can_mh_demo_score.sh \
  --overwrite \
  --exp-name can_mh_demo_score \
  --keep-ratio 0.9 \
  --num-train-steps 10000
```

## 关键产物

每条 workflow 的输出目录里会生成：

- `kept_sample_indices.npy`
- `norm_stats.json`
- `*_selection_summary.json`
- `*_selection_artifacts.npz`

训练启动后，新的 `OpenVLA` run 会写到 `runs/<exp-name>`。
