# SCIZOR -> OpenPI Retrain

这个目录把 `/workspace/liushuo/SCIZOR/outputs/act_scizor/can_mh/scizor_joint_filter.pkl`
接到 OpenPI 的训练链路里。

默认复现方式不是“按 episode 分数裁掉整条轨迹”，而是更接近 SCIZOR 原始语义的：

1. 读取 `metadata.window_drop_mask`
2. 把保留帧转换成 OpenPI 的 `sample_indices`
3. 基于这个 sample 子集重算 `norm_stats.json`
4. 用 `data.sample_indices_path` 启动 OpenPI 训练

原因：

- `scizor_joint_filter.pkl` 里的 `influence_scores` 实际上就是每个 episode 的 `keep_ratio`
- 但同一个 pickle 已经带了逐帧的 `window_drop_mask`
- OpenPI 支持 sample 子集训练，所以直接复用逐帧 mask 更忠实

## 一键复现

只准备子集和 `norm_stats`，不启动训练：

```bash
bash scizor/retrain/run_can_mh_joint_filter.sh --prepare-only --overwrite
```

准备完成后直接训练：

```bash
bash scizor/retrain/run_can_mh_joint_filter.sh \
  --overwrite \
  --exp-name can_mh_scizor_joint_filter \
  --num-train-steps 10000
```

默认会设置 `WANDB_MODE=disabled`，避免 `nohup`/无 TTY 场景下因为 `wandb login` 中断。
如果你需要启用 wandb，先在 shell 里配置好 `WANDB_API_KEY`，再显式覆盖：

```bash
WANDB_MODE=online bash scizor/retrain/run_can_mh_joint_filter.sh ...
```

## 关键产物

默认输出目录：

`./scizor/retrain/output/pi05_can_mh_lora_scizor_joint_filter`

里面会生成：

- `kept_sample_indices.npy`
- `scizor_selection_summary.json`
- `scizor_selection_artifacts.npz`
- `norm_stats.json`

其中：

- `kept_sample_indices.npy` 直接给 OpenPI 的 `data.sample_indices_path`
- `scizor_selection_summary.json` 会记录保留样本数、全保留/部分保留/全丢弃 episode 数

## 当前数据对齐结论

这个复现假设已经在当前环境核对过：

- OpenPI `can_mh` 数据集有 `300` 条 episode
- 总帧数是 `62756`
- `scizor_joint_filter.pkl` 里同样是 `300` 个 episode 分数、`62756` 个 frame-level windows

因此这个 pickle 可以直接映射到 OpenPI 的 `can_mh` 训练集。
