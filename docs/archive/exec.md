# OpenPI05 can_ph / square_ph 补充实验执行命令

## 1. 构建数据集与 sample 索引

```bash
bash /workspace/liushuo/cupid/openpi/scripts/build_canph_openpi05_ablation.sh
```

## 2. 检查 can_ph 两组补充 episode 数是否一致

```bash
wc -l /workspace/liushuo/cupid/output/openpi05_canph_ablation/manifests/can_ph_preference_keep.txt
wc -l /workspace/liushuo/cupid/output/openpi05_canph_ablation/manifests/can_ph_random_keep.txt
```

## 3. 检查 can_mh window 过滤后的 OpenPI sample 索引摘要

```bash
cat /workspace/liushuo/cupid/output/openpi05_canph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices_summary.json
cat /workspace/liushuo/cupid/output/openpi05_canph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices_summary.json
```

## 4. 归一化

### 4.1 偏好补齐组

```bash
CUDA_VISIBLE_DEVICES=7 bash /workspace/liushuo/cupid/openpi/scripts/norm_stats_canph_pref.sh
```

### 4.2 随机补齐组

```bash
CUDA_VISIBLE_DEVICES=6 bash /workspace/liushuo/cupid/openpi/scripts/norm_stats_canph_random.sh
```

## 5. 训练

### 5.1 偏好补齐组

```bash
CUDA_VISIBLE_DEVICES=7 nohup bash /workspace/liushuo/cupid/openpi/scripts/train_canph_pref.sh > logs/train_canph_pref.log 2>&1 &
```

### 5.2 随机补齐组

```bash
CUDA_VISIBLE_DEVICES=6 nohup bash /workspace/liushuo/cupid/openpi/scripts/train_canph_random.sh > logs/train_canph_random.log 2>&1 &
```

## 6. 后台运行命令

### 6.1 后台归一化

```bash
nohup bash /workspace/liushuo/cupid/openpi/scripts/norm_stats_canph_pref.sh > /workspace/liushuo/cupid/openpi/logs/norm_stats_canph_pref.log 2>&1 &
nohup bash /workspace/liushuo/cupid/openpi/scripts/norm_stats_canph_random.sh > /workspace/liushuo/cupid/openpi/logs/norm_stats_canph_random.log 2>&1 &
```

### 6.2 后台训练

```bash
nohup bash /workspace/liushuo/cupid/openpi/scripts/train_canph_pref.sh > /workspace/liushuo/cupid/openpi/logs/train_canph_pref.log 2>&1 &
nohup bash /workspace/liushuo/cupid/openpi/scripts/train_canph_random.sh > /workspace/liushuo/cupid/openpi/logs/train_canph_random.log 2>&1 &
```

## 7. 自定义补充 episode 数

如果你想把 `can_ph` 补充数从默认的 `54` 改成别的值，例如 `40`，先执行：

```bash
cd /workspace/liushuo/cupid

python /workspace/liushuo/cupid/scripts/build_openpi05_canph_ablation.py \
  --supplement-count 40 \
  --overwrite
```

然后重新执行：

```bash
bash /workspace/liushuo/cupid/openpi/scripts/build_canph_openpi05_ablation.sh
```

---

# OpenPI05 square_ph 补充实验执行命令

说明：

- 当前 `square_mh` 的基础筛选结果使用 `openpi05` 自己的 joint episode-window 输出，而不是 `act` 侧产物。
- 当前 `square_ph` 仍复用 `can_ph` 的构建与训练脚本，因此输出目录中的数据集名称仍保留 `can_mh_twostage_plus_can_ph_*` 字样。
- 只要下面的路径参数和 `TRAIN_CONFIG=pi05_square_mh_lora` 设置正确，实验逻辑就是 `square_mh + square_ph`。

## 1. 构建 square_mh + square_ph 的 pref / random 两组 merged dataset

```bash
cd /workspace/liushuo/cupid

python /workspace/liushuo/cupid/scripts/build_openpi05_canph_ablation.py \
  --can-mh-dataset /workspace/liushuo/cupid/datasets/lerobot/square_mh \
  --can-mh-keep-list /workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/square_mh_joint_episode_window/selection/kept_episodes.txt \
  --can-ph-dataset /workspace/liushuo/robomimic/robomimic/datasets/lerobot/square_ph \
  --compare-csv /workspace/liushuo/cupid/output/compare_act_openpi_square_mh/episode_comparison.csv \
  --output-root /workspace/liushuo/cupid/output/openpi05_squareph_ablation \
  --supplement-count 54 \
  --random-seed 42 \
  --overwrite
```

## 2. 构建 pref 组的 OpenPI sample_indices

```bash
cd /workspace/liushuo/cupid/openpi

UV_CACHE_DIR=/workspace/liushuo/cupid/openpi/tmp/uv-cache \
uv run python /workspace/liushuo/cupid/openpi/scripts/build_canph_openpi05_sample_indices.py \
  --dataset-dir /workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054 \
  --selection-npz /workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/square_mh_joint_episode_window/selection/joint_episode_window_selection.npz \
  --can-mh-keep-list /workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/square_mh_joint_episode_window/selection/kept_episodes.txt \
  --train-config pi05_square_mh_lora \
  --output-indices /workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices.npy \
  --output-summary /workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices_summary.json
```

## 3. 构建 random 组的 OpenPI sample_indices

```bash
cd /workspace/liushuo/cupid/openpi

UV_CACHE_DIR=/workspace/liushuo/cupid/openpi/tmp/uv-cache \
uv run python /workspace/liushuo/cupid/openpi/scripts/build_canph_openpi05_sample_indices.py \
  --dataset-dir /workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42 \
  --selection-npz /workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/square_mh_joint_episode_window/selection/joint_episode_window_selection.npz \
  --can-mh-keep-list /workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/square_mh_joint_episode_window/selection/kept_episodes.txt \
  --train-config pi05_square_mh_lora \
  --output-indices /workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices.npy \
  --output-summary /workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices_summary.json
```

## 4. 归一化

### 4.1 pref 组

```bash
cd /workspace/liushuo/cupid

CUDA_VISIBLE_DEVICES=6 DATASET_DIR=/workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054 \
TRAIN_CONFIG=pi05_square_mh_lora \
SAMPLE_INDICES_PATH=/workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices.npy \
bash /workspace/liushuo/cupid/openpi/scripts/norm_stats_canph_pref.sh
```

### 4.2 random 组

```bash
cd /workspace/liushuo/cupid

CUDA_VISIBLE_DEVICES=5 DATASET_DIR=/workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42 \
TRAIN_CONFIG=pi05_square_mh_lora \
SAMPLE_INDICES_PATH=/workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices.npy \
bash /workspace/liushuo/cupid/openpi/scripts/norm_stats_canph_random.sh
```

## 5. 训练

### 5.1 pref 组

```bash
cd /workspace/liushuo/cupid

CUDA_VISIBLE_DEVICES=7 \
DATASET_DIR=/workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054 \
ASSETS_DIR=/workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets \
ASSET_ID=can_mh_twostage_plus_can_ph_pref_k054 \
TRAIN_CONFIG=pi05_square_mh_lora \
EXP_NAME=square_mh_plus_square_ph_pref_k054 \
SAMPLE_INDICES_PATH=/workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices.npy \
nohup bash /workspace/liushuo/cupid/openpi/scripts/train_canph_pref.sh \
  > /workspace/liushuo/cupid/openpi/logs/train_squareph_pref.log 2>&1 &
```

### 5.2 random 组

```bash
cd /workspace/liushuo/cupid

CUDA_VISIBLE_DEVICES=6 \
DATASET_DIR=/workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42 \
ASSETS_DIR=/workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets \
ASSET_ID=can_mh_twostage_plus_can_ph_random_k054_seed42 \
TRAIN_CONFIG=pi05_square_mh_lora \
EXP_NAME=square_mh_plus_square_ph_random_k054_seed42 \
SAMPLE_INDICES_PATH=/workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices.npy \
nohup bash /workspace/liushuo/cupid/openpi/scripts/train_canph_random.sh \
  > /workspace/liushuo/cupid/openpi/logs/train_squareph_random.log 2>&1 &
```

## 6. 检查 sample_indices 摘要

```bash
cat /workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices_summary.json
cat /workspace/liushuo/cupid/output/openpi05_squareph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices_summary.json
```

---

# OpenPI05 transport_ph 补充实验执行命令

说明：

- 当前 `transport_mh` 的基础筛选结果使用 `openpi05` 自己的 joint episode-window 输出，而不是 `act` 侧产物。
- 当前 `transport_ph` 的 LeRobot 数据集路径为：`/workspace/liushuo/robomimic/robomimic/datasets/lerobot/transport_ph`
- 当前 `transport_ph` 仍复用 `can_ph` 的构建与训练脚本，因此输出目录中的数据集名称仍保留 `can_mh_twostage_plus_can_ph_*` 字样。
- 只要下面的路径参数和 `TRAIN_CONFIG=pi05_transport_mh_lora` 设置正确，实验逻辑就是 `transport_mh + transport_ph`。

## 1. 构建 transport_mh + transport_ph 的 pref / random 两组 merged dataset

```bash
cd /workspace/liushuo/cupid

python /workspace/liushuo/cupid/scripts/build_openpi05_canph_ablation.py \
  --can-mh-dataset /workspace/liushuo/cupid/datasets/lerobot/transport_mh \
  --can-mh-keep-list /workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/transport_mh_joint_episode_window/selection/kept_episodes.txt \
  --can-ph-dataset /workspace/liushuo/robomimic/robomimic/datasets/lerobot/transport_ph \
  --compare-csv /workspace/liushuo/cupid/output/compare_act_openpi_transport_mh/episode_comparison.csv \
  --output-root /workspace/liushuo/cupid/output/openpi05_transportph_ablation \
  --supplement-count 54 \
  --random-seed 42 \
  --overwrite
```

说明：

- 当前 `transport_mh` 在 `openpi05` 筛选后 keep 的 episode 数为 `270`
- 这里按 `0.2` 比例取补充数，所以默认是 `54`

## 2. 构建 pref 组的 OpenPI sample_indices

```bash
cd /workspace/liushuo/cupid/openpi

UV_CACHE_DIR=/workspace/liushuo/cupid/openpi/tmp/uv-cache \
uv run python /workspace/liushuo/cupid/openpi/scripts/build_canph_openpi05_sample_indices.py \
  --dataset-dir /workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054 \
  --selection-npz /workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/transport_mh_joint_episode_window/selection/joint_episode_window_selection.npz \
  --can-mh-keep-list /workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/transport_mh_joint_episode_window/selection/kept_episodes.txt \
  --train-config pi05_transport_mh_lora \
  --output-indices /workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices.npy \
  --output-summary /workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices_summary.json
```

## 3. 构建 random 组的 OpenPI sample_indices

```bash
cd /workspace/liushuo/cupid/openpi

UV_CACHE_DIR=/workspace/liushuo/cupid/openpi/tmp/uv-cache \
uv run python /workspace/liushuo/cupid/openpi/scripts/build_canph_openpi05_sample_indices.py \
  --dataset-dir /workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42 \
  --selection-npz /workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/transport_mh_joint_episode_window/selection/joint_episode_window_selection.npz \
  --can-mh-keep-list /workspace/liushuo/cupid/openpi/openpi-script/output/joint_episode_window_serial/transport_mh_joint_episode_window/selection/kept_episodes.txt \
  --train-config pi05_transport_mh_lora \
  --output-indices /workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices.npy \
  --output-summary /workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices_summary.json
```

## 4. 归一化

### 4.1 pref 组

```bash
cd /workspace/liushuo/cupid

DATASET_DIR=/workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054 \
TRAIN_CONFIG=pi05_transport_mh_lora \
SAMPLE_INDICES_PATH=/workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices.npy \
bash /workspace/liushuo/cupid/openpi/scripts/norm_stats_canph_pref.sh
```

### 4.2 random 组

```bash
cd /workspace/liushuo/cupid

DATASET_DIR=/workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42 \
TRAIN_CONFIG=pi05_transport_mh_lora \
SAMPLE_INDICES_PATH=/workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices.npy \
bash /workspace/liushuo/cupid/openpi/scripts/norm_stats_canph_random.sh
```

## 5. 训练

### 5.1 pref 组

```bash
cd /workspace/liushuo/cupid

CUDA_VISIBLE_DEVICES=3 \
DATASET_DIR=/workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054 \
ASSETS_DIR=/workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets \
ASSET_ID=can_mh_twostage_plus_can_ph_pref_k054 \
TRAIN_CONFIG=pi05_transport_mh_lora \
EXP_NAME=transport_mh_plus_transport_ph_pref_k054 \
SAMPLE_INDICES_PATH=/workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices.npy \
nohup bash /workspace/liushuo/cupid/openpi/scripts/train_canph_pref.sh \
  > /workspace/liushuo/cupid/openpi/logs/train_transportph_pref.log 2>&1 &
```

### 5.2 random 组

```bash
cd /workspace/liushuo/cupid

CUDA_VISIBLE_DEVICES=1 \
DATASET_DIR=/workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42 \
ASSETS_DIR=/workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets \
ASSET_ID=can_mh_twostage_plus_can_ph_random_k054_seed42 \
TRAIN_CONFIG=pi05_transport_mh_lora \
EXP_NAME=transport_mh_plus_transport_ph_random_k054_seed42 \
SAMPLE_INDICES_PATH=/workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices.npy \
nohup bash /workspace/liushuo/cupid/openpi/scripts/train_canph_random.sh \
  > /workspace/liushuo/cupid/openpi/logs/train_transportph_random.log 2>&1 &
```

## 6. 检查 sample_indices 摘要

```bash
cat /workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_pref_k054/openpi_sample_indices_summary.json
cat /workspace/liushuo/cupid/output/openpi05_transportph_ablation/datasets/can_mh_twostage_plus_can_ph_random_k054_seed42/openpi_sample_indices_summary.json
```
