# ACT Client 使用说明

## ⚠️ 重要提示

1.  **相机**: 必须添加 `--use_camera_obs` 参数。
2.  **控制器**: 如果模型输出是 **Joint Positions** (8维), 必须添加 `--controller JOINT_POSITION` 参数。
    默认是 `OSC_POSE` (7维, End-Effector)。

## 正确的使用方法

### Joint Space Policy (关节控制模型)

```bash
python client_mimicgen_act.py \
    --use_camera_obs \
    --controller JOINT_POSITION \
    --video_dir ./act_joint \
    --env_name Square_D1
```

### OSC Pose Policy (默认)

```bash
python client_mimicgen_act.py \
    --use_camera_obs \
    --video_dir ./act_osc
```

### 完整参数示例 (推荐)

```bash
python client_mimicgen_act.py \
    --server_url http://127.0.0.1:8000/act \
    --env_name Square_D1 \
    --robots Panda \
    --controller JOINT_POSITION \
    --use_camera_obs \
    --num_episodes 50 \
    --save_rollouts_dir ./eval_rollouts_act
```

## 常见错误

### ❌ ValueError: shapes (50,8) (7,) (7,)

原因:
1.  **Chunking**: Server 返回了 50 步的动作 chunk，Client 需要按步执行。
2.  **维度**: 模型输出 8 维，环境只有 7 维。

**解决方法**:
- 代码已支持 Chunking (默认按步执行 action chunk)。如果想用单步 Receding Horizon，可加 `--no_action_chunk`。
- **维度问题**: 请确保 `--controller` 设置正确。如果模型输出 8 维关节数据，必须加 `--controller JOINT_POSITION`。

### ❌ ValueError: ACT policy requires camera images

**解决方法**: 添加 `--use_camera_obs`。

## 数据格式

- **Images**: `data/observation.images.agentview` + `wrist`
- **State**: `data/observation.state` (Joint Pos + Gripper Pos)
- **Action**: Joint Positions (7) + Gripper (1) = 8 dim (如果使用 JOINT_POSITION)

## 兼容性

Client 代码已更新，自动处理：
1.  **Action Chunks**: 默认按步执行 chunk (可用 `--no_action_chunk` 关闭)。
2.  **Extra Dimensions**: 如果维度超出环境要求（例如 padding），会自动截取。但请务必设置正确的控制器以避免错误控制。
