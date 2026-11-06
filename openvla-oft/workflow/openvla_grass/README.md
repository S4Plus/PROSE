# OpenVLA GraSS Influence

这个目录用于计算 `OpenVLA` 的 influence score，并使用 `GraSS` 的两阶段压缩近似加速。

底层实现复用：

- `vla-scripts/influence_openvla_lora_grass_window.py`

输出是窗口级 influence pickle，默认文件名是 `influence_grass.pkl`。

## 流程

1. 准备一个训练好的 `OpenVLA` LoRA checkpoint。
2. 准备评测 rollout 的 `pkl` 目录。
3. 运行 `GraSS` influence 计算脚本。

输出内容包括：

- `sample_influence_scores`
- `window_influence_scores`
- `episode_influence_scores`
- 窗口与样本索引映射元数据

## 常用脚本

- `run_openvla_grass_influence.sh`
  通用入口。
- `run_can_mh_openvla_grass.sh`
  `can_mh` 默认参数。
- `run_square_mh_openvla_grass.sh`
  `square_mh` 默认参数。

## 示例

`can_mh`:

```bash
bash /workspace/liushuo/cupid/openvla-oft/workflow/openvla_grass/run_can_mh_openvla_grass.sh
```

`square_mh`:

```bash
bash /workspace/liushuo/cupid/openvla-oft/workflow/openvla_grass/run_square_mh_openvla_grass.sh
```

## 默认加速配置

默认启用的 `GraSS` 近似参数是：

```text
sparsification = random_mask-32*32
projection     = sjlt-1024
hessian        = eFIM
damping        = 1e-4
stage1_microbatch_size = 1
stage2_batch_size      = 16
```

如果显存紧张，可以调小：

```bash
STAGE2_BATCH_SIZE=8
```

如果要更快但更粗糙，可以减小：

```bash
PROJECTION=sjlt-512
```

## 输出目录

默认输出在：

- `tmp/<dataset>_openvla_window_grass/influence_grass.pkl`
- `tmp/<dataset>_openvla_window_grass/cache/`

其中 `cache/` 会缓存训练梯度、preconditioner 和 eval 梯度，重复运行会直接复用。
