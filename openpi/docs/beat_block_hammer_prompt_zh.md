# beat_block_hammer 数据集的 Prompt 设计说明

本文说明 `/workspace/liushuo/dataset/beat_block_hammer_ur5_randomized_500` 在 `openpi05` 训练时的 prompt 构造方式，以及 `arm_id` 是如何被注入到训练输入中的。

## 目标

这份数据集的 `meta/tasks.jsonl` 固定为一条通用任务描述：

```text
Pick up the hammer and hit the block
```

同时，每条轨迹还包含 `observation.env_state.arm_id`，用于指定本条样本应由哪只手执行动作。

训练目标不是恢复 `task_index -> task` 的细粒度映射，而是采用：

1. 固定任务 prompt
2. 训练时根据 `arm_id` 动态在 prompt 前面追加左右臂条件

## 当前实现

### 1. 固定任务 prompt

数据加载阶段会读取 `tasks.jsonl` 中唯一的一条任务文本：

```text
Pick up the hammer and hit the block
```

当本地 LeRobot 数据集只有这一条 task 文本，但 `meta/info.json` 中 `total_tasks > 1` 时，`openpi` 会在内存里把这条固定 prompt 广播到所有 `task_index`。

这样做的目的是：

- 不修改磁盘上的 `tasks.jsonl`
- 不依赖真实的 `task_index -> task` 文案映射
- 仍然兼容 `LeRobotDataset` 默认会访问 `item["task"]` 的实现

相关代码在：

- `src/openpi/training/data_loader.py`

### 2. arm_id 注入

在训练配置 `pi05_beat_block_hammer_lora` 中，数据会先从 LeRobot 样本中 repack 出：

- `images`
- `state`
- `actions`
- `arm_id`
- `prompt`

然后通过 `InjectArmIdIntoPrompt` 将 `arm_id` 注入到 prompt 前缀中。

当前默认映射规则是：

- `arm_id = 0` -> `left`
- `arm_id = 1` -> `right`

因此，训练时的可读 prompt 会是下面两种之一：

```text
Use the left arm. Pick up the hammer and hit the block
```

```text
Use the right arm. Pick up the hammer and hit the block
```

相关代码在：

- `src/openpi/policies/arm_prompt.py`
- `src/openpi/training/config.py`

## openpi05 最终看到的输入字符串

对于 `pi05`，prompt 在送入 tokenizer 时，还会和离散化后的 `state` 一起拼接。

逻辑位于：

- `src/openpi/models/tokenizer.py`

最终格式是：

```text
Task: Use the left arm. Pick up the hammer and hit the block, State: <离散化状态token>;
Action:
```

或者：

```text
Task: Use the right arm. Pick up the hammer and hit the block, State: <离散化状态token>;
Action:
```

也就是说，`openpi05` 实际训练时使用的是：

1. 固定任务语义 `Pick up the hammer and hit the block`
2. 左右臂条件 `Use the left/right arm`
3. 当前时刻的离散化状态 token

## 训练配置

当前使用的训练配置名是：

```text
pi05_beat_block_hammer_lora
```

对应命令：

### 1. 计算归一化统计量

```bash
cd /workspace/liushuo/cupid/openpi
CUDA_VISIBLE_DEVICES=4 uv run scripts/compute_norm_stats.py --config-name pi05_beat_block_hammer_lora
```

### 2. 启动训练

```bash
cd /workspace/liushuo/cupid/openpi
CUDA_VISIBLE_DEVICES=4 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
WANDB_MODE=disabled \
uv run scripts/train.py pi05_beat_block_hammer_lora \
  --exp-name=beat_block_hammer_pi05_prompt_arm \
  --overwrite
```

## 注意事项

### 1. arm_id 映射方向

当前代码默认：

- `0 = left`
- `1 = right`

如果你后续确认数据集中语义相反，需要修改：

- `src/openpi/policies/arm_prompt.py`

### 2. prompt 冲突保护

如果原始 prompt 已经明确写了 `left arm` 或 `right arm`，而且它和 `arm_id` 冲突，代码会直接报错。

这样做是为了避免模型同时接收到相互矛盾的语言条件。

### 3. 为什么不直接用 episodes.jsonl 的细粒度任务文本

这条训练链的设计重点是：

- 保持任务语义稳定
- 只让 `arm_id` 作为关键可控条件进入 prompt

因此当前方案选择固定任务文本，而不是为不同 `task_index` 恢复各自的细粒度描述。
