# stack_cube_franka Remote Inference

这个目录提供一套和 `pi05_stack_cube_franka_lora` 对齐的最小 `server/client` 示例，用来验证 websocket 推理链路是否通。

## 1. 启动 server

```bash
uv run python examples/stack_cube_franka/server.py \
  --checkpoint-dir /path/to/checkpoint_dir
```

常用参数：

- `--config-name pi05_stack_cube_franka_lora`
- `--default-prompt "stack cube"`
- `--port 8003`
- `--pytorch-device cuda:0`

## 2. 发起 client 请求

```bash
uv run python examples/stack_cube_franka/client.py \
  --host 127.0.0.1 \
  --port 8003 \
  --prompt "stack cube" \
  --state "0,0,0,0,0,0,0,0" \
  --fixed-image /path/to/fixed.png \
  --wrist-image /path/to/wrist.png
```

如果不传图片，client 会自动生成随机图，仅用于联调，不代表策略会有合理输出。

## 3. 输入格式

这个示例发给 server 的 observation 是推理态结构，不是 LeRobot 原始字段：

```python
{
    "images": {
        "fixed_camera": np.ndarray,   # uint8, HWC
        "wrist_camera": np.ndarray,   # uint8, HWC
    },
    "state": np.ndarray,              # shape=(8,), 前 7 维关节角 + 最后 1 维 gripper width
    "prompt": "stack cube",
}
```

## 4. 输出格式

server 返回：

```python
{
    "actions": np.ndarray,            # shape=(action_horizon, 8)
    "policy_timing": {...},
    "server_timing": {...},
}
```

对于当前 `pi05_stack_cube_franka_lora` 配置，`action_horizon=20`，client 会打印第一步 action 和整体 shape。
