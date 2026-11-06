# MimicGen Client Rollout 保存功能使用指南

## 概述

已为 MimicGen evaluation client (OpenPI05 和 ACT 版本) 添加了保存 eval rollout 数据为 pkl 格式的功能,用于后续的 influence 计算。

## 修改的文件

1. **`/workspace/liushuo/cupid/mimicgen/eval/client_mimicgen_openpi05.py`** - OpenPI05 版本
2. **`/workspace/liushuo/cupid/mimicgen/eval/client_mimicgen_act.py`** - ACT 版本

## 新增参数

两个 client 都添加了以下参数:

- `--save_rollouts_dir`: 指定保存 rollout pkl 文件的目录
- `--save_rollouts_format`: 保存格式 (dict 或 pandas, 默认 dict)

## 使用方法

### OpenPI05 Client

```bash
python /workspace/liushuo/cupid/mimicgen/eval/client_mimicgen_openpi05.py \
    --server_url ws://127.0.0.1:8000 \
    --transport ws \
    --env_name Square_D1 \
    --num_episodes 50 \
    --save_rollouts_dir ./eval_rollouts_openpi05 \
    --use_camera_obs \
    --openpi_obs
```

### ACT Client

```bash
python /workspace/liushuo/cupid/mimicgen/eval/client_mimicgen_act.py \
    --server_url http://127.0.0.1:8000/act \
    --transport http \
    --env_name Square_D1 \
    --num_episodes 50 \
    --save_rollouts_dir ./eval_rollouts_act \
    --use_camera_obs
```

## 保存的数据格式

每个 episode 保存为 `episode_XXXXXX.pkl`,包含:

```python
{
    "observations": {
        # OpenPI05 格式:
        "observation.images.agentview": np.ndarray,  # [T, H, W, C]
        "observation.images.wrist": np.ndarray,      # [T, H, W, C]
        "observation.state": np.ndarray,             # [T, state_dim]
        "state": np.ndarray,                         # [T, state_dim]
        
        # 或 ACT 格式 (原始观测):
        "agentview_image": np.ndarray,
        "robot0_eye_in_hand_image": np.ndarray,
        # ... 其他观测字段
    },
    "actions": np.ndarray,      # [T, action_dim]
    "reward": float,            # episode 总 reward
    "success": bool,            # 成功标志
    "rewards": np.ndarray,      # [T] 每步的 reward (可选)
}
```

## 与 Influence 计算集成

保存的 rollout 数据可以直接用于 influence 计算:

```bash
# 使用保存的 eval rollouts
cd /workspace/liushuo/cupid/openpi

# OpenPI LoRA influence 计算
python ./scripts/influence_pi05_lora.py \
    --config-name pi05_square_d1_lora \
    --checkpoint-dir ./checkpoints/pi05_square_d1_lora/square_d1_lora/29999 \
    --eval-rollouts-dir ./eval_rollouts_openpi05 \
    --output-path ./influence_scores.pkl \
    --proj-dim 4096 \
    --lissa-depth 10
```

## 完整工作流程示例

### 1. 启动 Policy Server

```bash
# OpenPI05
python openpi/scripts/serve_policy.py \
    --config-name pi05_square_d1_lora \
    --checkpoint-dir <checkpoint_dir>

# 或 ACT
python act/serve_act_policy.py \
    --pretrained-model <model_path>
```

### 2. 运行 Evaluation + 保存 Rollouts

# OpenPI05
cd /workspace/liushuo/cupid/mimicgen/eval
python client_mimicgen_openpi05.py \
    --video_dir ./openpi05 \
    --save_rollouts_dir ./openpi05/pkl \
    --openpi_obs

# ACT
python client_mimicgen_act.py \
    --video_dir ./act \
    --save_rollouts_dir ./act/pkl 
```

### 3. 计算 Influence Scores

```bash
cd /workspace/liushuo/cupid/openpi

python ./scripts/influence_pi05_lora.py \
    --config-name pi05_square_d1_lora \
    --checkpoint-dir ./checkpoints/pi05_square_d1_lora/square_d1_lora/29999 \
    --eval-rollouts-dir ./eval_rollouts \
    --output-path ./results/influence.pkl \
    --stage-cache-prefix ./cache
```

### 4. 分析结果

```python
import pickle
import numpy as np

# 加载结果
with open("./results/influence.pkl", "rb") as f:
    result = pickle.load(f)

influence_scores = result["influence_scores"]

# Top 100 最有影响力的训练样本
top_100 = np.argsort(influence_scores)[-100:][::-1]
print("Top 100 influential samples:", top_100)
```

## 注意事项

1. **保存路径**: 建议使用绝对路径或相对于运行目录的路径
2. **磁盘空间**: 每个 episode 根据长度和图像分辨率大小不同,建议预留足够空间
3. **OpenPI vs ACT**: OpenPI05 client 会自动格式化观测为 OpenPI 格式 (`observation.images.*`, `observation.state`)
4. **数据格式兼容**: 保存的格式与 `save_eval_rollouts.py` 和 `influence_pi05_lora.py` 完全兼容

## 示例脚本

查看 `/workspace/liushuo/cupid/openpi/scripts/influence_example.py` 获取完整的使用示例。
