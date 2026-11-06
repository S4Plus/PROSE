# 三机双臂 OpenPI 推理部署

目标拓扑：

- `10.0.40.113`: Server / Master / Policy，负责同步观测、调用 OpenPI policy、分发左右动作。
- `10.0.54.248`: Right Host，负责 `right_wrist`、`agent_view` 和右臂 Franka。
- `10.0.54.196`: Left Host，负责 `left_wrist` 和左臂 Franka。

当前任务 `pi05_right_box_left_object_right_lid_lora` 的 policy 输入必须同时包含：

- `observation.images.agent_view`
- `observation.images.left_wrist`
- `observation.images.right_wrist`
- `observation.state`: 16 维，顺序为 `left 7 joints + left gripper + right 7 joints + right gripper`

## 0. 前置要求

三台机器先确认网络与时间同步：

```bash
ping -c 3 10.0.40.113
ping -c 3 10.0.54.248
ping -c 3 10.0.54.196
timedatectl
```

如果要执行真实双臂动作，三台机器建议开启 `chrony` 或等价 NTP/PTP。Master 会用同一个 epoch time 下发 `execute_at`，左右 Host 按本机时钟等待后启动动作；时钟不同步会直接变成双臂启动偏差。

## 1. Server / Master / Policy: 启动 OpenPI policy

在 `10.0.40.113` 上运行：

```bash
cd /workspace/liushuo/cupid/openpi

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
UV_CACHE_DIR=/tmp/uv-cache-openpi \
uv run scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config=pi05_right_box_left_object_right_lid_lora \
  --policy.dir=/workspace/liushuo/cupid/openpi/checkpoints/pi05_right_box_left_object_right_lid_lora/right_box_left_object_right_lid_pi05_30k/29999
```

健康检查：

```bash
curl --noproxy '*' http://127.0.0.1:8000/healthz
curl --noproxy '*' http://10.0.40.113:8000/healthz
```

## 2. Right Host: 启动右臂 agent

在 `10.0.54.248` 上运行。把 `RIGHT_FRANKA_FCI_IP`、`RIGHT_WRIST_SERIAL`、`AGENT_VIEW_SERIAL` 替换成真实值：

```bash
cd /workspace/liushuo/cupid/openpi

UV_CACHE_DIR=/tmp/uv-cache-openpi \
uv run python franka/distributed_host_agent.py \
  --side right \
  --host 0.0.0.0 \
  --port 9101 \
  --robot-ip RIGHT_FRANKA_FCI_IP \
  --cameras right_wrist:RIGHT_WRIST_SERIAL,agent_view:AGENT_VIEW_SERIAL \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --state-gripper-mode command \
  --gripper-action-mode command
```

先只测相机/网络、不连接机器人时可以加：

```bash
--no-robot
```

健康检查：

```bash
curl --noproxy '*' http://10.0.54.248:9101/healthz
```

## 3. Left Host: 启动左臂 agent

在 `10.0.54.196` 上运行。把 `LEFT_FRANKA_FCI_IP`、`LEFT_WRIST_SERIAL` 替换成真实值：

```bash
cd /workspace/liushuo/cupid/openpi

UV_CACHE_DIR=/tmp/uv-cache-openpi \
uv run python franka/distributed_host_agent.py \
  --side left \
  --host 0.0.0.0 \
  --port 9102 \
  --robot-ip LEFT_FRANKA_FCI_IP \
  --cameras left_wrist:LEFT_WRIST_SERIAL \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --state-gripper-mode command \
  --gripper-action-mode command
```

健康检查：

```bash
curl --noproxy '*' http://10.0.54.196:9102/healthz
```

## 4. Master: 先做字段级健康检查

在 `10.0.40.113` 上运行：

```bash
cd /workspace/liushuo/cupid/openpi

UV_CACHE_DIR=/tmp/uv-cache-openpi \
uv run python franka/distributed_master_pi05.py \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --right-host 10.0.54.248 \
  --right-port 9101 \
  --left-host 10.0.54.196 \
  --left-port 9102 \
  --healthcheck
```

这一步会确认：

- 可以连上左右 Host Agent。
- 可以连上本机 policy server。
- 可以收到 `left_wrist/right_wrist/agent_view`。
- 合成后的 `observation.state` 是 16 维。

## 5. Master: 推理但不执行

先跑 dry inference，确认 policy 输出 action shape 和 gripper command 语义：

```bash
cd /workspace/liushuo/cupid/openpi

UV_CACHE_DIR=/tmp/uv-cache-openpi \
uv run python franka/distributed_master_pi05.py \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --right-host 10.0.54.248 \
  --right-port 9101 \
  --left-host 10.0.54.196 \
  --left-port 9102 \
  --no-execute \
  --max-actions 32 \
  --success-label 0
```

## 6. Master: 真实执行

确认 dry inference 正常后再执行：

```bash
cd /workspace/liushuo/cupid/openpi

UV_CACHE_DIR=/tmp/uv-cache-openpi \
uv run python franka/distributed_master_pi05.py \
  --policy-host 127.0.0.1 \
  --policy-port 8000 \
  --right-host 10.0.54.248 \
  --right-port 9101 \
  --left-host 10.0.54.196 \
  --left-port 9102 \
  --execute \
  --chunk-exec-steps 32 \
  --action-hz 30 \
  --execute-lead-s 0.35 \
  --max-actions 200
```

运行中按 Enter 会结束当前 episode 并保存 pkl。默认保存到：

```text
/workspace/liushuo/cupid/openpi/Dataset/dual_host_pi05_rollouts
```

## 7. 关键风险点

- `10.0.54.248` 和 `10.0.54.196` 是控制主机 IP，不一定是 Franka FCI IP；`--robot-ip` 必须填各自 Franka 的 FCI IP。
- 当前 policy 的训练 prompt 是 `Use the right arm to stage the box and lid, the left arm to place the object, then close the lid.`，Master 默认使用该 prompt。
- 当前动作顺序固定为左臂 8 维在前、右臂 8 维在后，不能和训练数据顺序对调。
- 真实执行前先用 `--no-execute` 看第一帧 action，确认没有 NaN、没有大跳变、左右臂方向没有反。
