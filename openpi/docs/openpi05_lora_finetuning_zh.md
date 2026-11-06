# OpenPI05 中的 LoRA 微调说明

本文整理了这个仓库里 OpenPI05 的 LoRA 微调实现方式、常用配置，以及“LoRA 微调”在当前代码中究竟训练了哪些参数。

## 总览

OpenPI05 的 LoRA 微调主要由三层组成：

1. LoRA 模块本身定义在 [`lora.py`](../src/openpi/models/lora.py)。
2. Gemma 的不同 variant 决定在哪些权重上注入 LoRA，以及对应的 rank / alpha。
3. 训练配置通过 `freeze_filter` 指定冻结哪些参数，再由 `trainable_filter` 决定最终参与训练的参数集合。

这个仓库里常见的 OpenPI05 LoRA 配置通常是：

- PaliGemma 分支：`gemma_2b_lora`
- action expert 分支：`gemma_300m_lora`
- policy 类型：`pi05=True`

最典型的配置是：

- [`pi05_can_mh_lora`](../src/openpi/training/config.py)

## LoRA 的实现

LoRA 的核心实现在 [`src/openpi/models/lora.py`](../src/openpi/models/lora.py)。

### `LoRAConfig`

`LoRAConfig` 里定义了这些关键字段：

- `rank`
- `alpha`
- `init_fn`
- `rslora`
- `axes`
- `label`

缩放系数规则是：

- 默认：`alpha / rank`
- 如果 `rslora=True`：`alpha / sqrt(rank)`

### 注意力上的 LoRA

`lora.Einsum` 是对普通 einsum 权重的封装，会在原始权重之外额外引入低秩参数：

- base weight：`w`
- LoRA 权重：`lora_a`、`lora_b`

前向计算形式可以理解为：

`base(x, w) + lora(x, lora_a, lora_b) * scaling`

### MLP 上的 LoRA

`lora.FeedForward` 会给 MLP 的两部分加 LoRA：

- `gating_einsum`
- `linear`

对应 LoRA 参数名是：

- `gating_einsum_lora_a`
- `gating_einsum_lora_b`
- `linear_lora_a`
- `linear_lora_b`

## LoRA 注入到了哪里

Gemma 的 LoRA variant 定义在 [`src/openpi/models/gemma.py`](../src/openpi/models/gemma.py)。

### 支持的 LoRA variant

- `gemma_2b_lora`
- `gemma_300m_lora`

### LoRA 超参数

- `gemma_2b_lora`
  - attention LoRA：`rank=16`，`alpha=16`
  - ffn LoRA：`rank=16`，`alpha=16`
- `gemma_300m_lora`
  - attention LoRA：`rank=32`，`alpha=32`
  - ffn LoRA：`rank=32`，`alpha=32`

### Attention 注入点

LoRA 会注入到 Gemma attention 里的这些投影上：

- `qkv_einsum`
- `q_einsum`
- `kv_einsum`
- `attn_vec_einsum`

### Feed-forward 注入点

LoRA 会注入到 Gemma 的 MLP：

- `mlp`

### `pi05_can_mh_lora` 中可见的 LoRA 层清单

结合当前模型结构与实际导出的 trainable leaves，`pi05_can_mh_lora` 里可见的 LoRA 层主要是下面这些：

- `PaliGemma/llm/layers/attn/attn_vec_einsum/lora_a`
- `PaliGemma/llm/layers/attn/attn_vec_einsum/lora_b`
- `PaliGemma/llm/layers/attn/attn_vec_einsum_1/lora_a`
- `PaliGemma/llm/layers/attn/attn_vec_einsum_1/lora_b`
- `PaliGemma/llm/layers/attn/kv_einsum/lora_a`
- `PaliGemma/llm/layers/attn/kv_einsum/lora_b`
- `PaliGemma/llm/layers/attn/kv_einsum_1/lora_a`
- `PaliGemma/llm/layers/attn/kv_einsum_1/lora_b`
- `PaliGemma/llm/layers/attn/q_einsum/lora_a`
- `PaliGemma/llm/layers/attn/q_einsum/lora_b`
- `PaliGemma/llm/layers/attn/q_einsum_1/lora_a`
- `PaliGemma/llm/layers/attn/q_einsum_1/lora_b`
- `PaliGemma/llm/layers/mlp/gating_einsum_lora_a`
- `PaliGemma/llm/layers/mlp/gating_einsum_lora_b`
- `PaliGemma/llm/layers/mlp/linear_lora_a`
- `PaliGemma/llm/layers/mlp/linear_lora_b`
- `PaliGemma/llm/layers/mlp_1/gating_einsum_lora_a`
- `PaliGemma/llm/layers/mlp_1/gating_einsum_lora_b`
- `PaliGemma/llm/layers/mlp_1/linear_lora_a`
- `PaliGemma/llm/layers/mlp_1/linear_lora_b`

可以粗略理解为：

- 不带 `_1` 的层偏向 PaliGemma 主分支
- 带 `_1` 的层偏向 action expert 分支

## OpenPI05 模型结构

OpenPI05 模型本体在 [`src/openpi/models/pi0.py`](../src/openpi/models/pi0.py)。

高层结构包含：

- `PaliGemma.llm`
- `PaliGemma.img`
- `action_in_proj`
- `action_out_proj`
- `time_mlp_in`
- `time_mlp_out`

对于 `pi05=True`，动作侧使用的是 `time_mlp_in/out` 这条路径，而不是非 PI05 模式下的 state/time 投影路径。

## 冻结规则与实际可训练参数

这是最关键的一点。

训练中真正参与优化的参数由下面两层决定：

- 训练配置里的 `freeze_filter`
- `trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))`

定义位置在 [`src/openpi/training/config.py`](../src/openpi/training/config.py)。

默认的 LoRA 冻结逻辑在 [`src/openpi/models/pi0_config.py`](../src/openpi/models/pi0_config.py)。

### 当前逻辑

如果 `paligemma_variant` 里包含 `"lora"`：

- 冻结 `.*llm.*`
- 但把 `.*lora.*` 从冻结集合里排除

如果 `action_expert_variant` 里包含 `"lora"`：

- 冻结 action-expert 对应的 `llm` 分支
- 同样把 `.*lora.*` 从冻结集合里排除

### 一个非常重要的结论

这个仓库里的“LoRA 微调”并不严格等于“只训练 LoRA 参数”。

对 `pi05_can_mh_lora` 这类配置，实际效果更接近：

- `llm` 里的大部分 base 权重被冻结
- `llm` 里的 LoRA 权重可训练
- `llm` 之外的参数仍然可能可训练

因此这些模块在当前实现下通常也会参与训练：

- `PaliGemma.img.*`
- `action_in_proj.*`
- `action_out_proj.*`
- `time_mlp_in.*`
- `time_mlp_out.*`

这和前面 influence / GraSS 实验里观测到的现象是一致的：可训练 leaves 不只是 `*lora*`，还包括 image 侧、action head 侧以及 PI05 的时间 MLP。

如果你希望实现“严格的 LoRA-only 训练”，当前 `freeze_filter` 逻辑还不够，需要进一步收紧。

### `pi05_can_mh_lora` 下实际会训练的参数

对 `pi05_can_mh_lora`，当前代码路径下实际会更新的参数可以分成两类。

第一类是 LoRA 参数：

- 上一节列出的所有 `PaliGemma/llm/.../lora_*`

第二类是非 LoRA、但当前没有被冻结的参数：

- 图像分支
  - `PaliGemma/img/Transformer/...`
  - `PaliGemma/img/embedding/...`
  - `PaliGemma/img/head/...`
  - `PaliGemma/img/pos_embedding`
- 动作头
  - `action_in_proj/bias`
  - `action_in_proj/kernel`
  - `action_out_proj/bias`
  - `action_out_proj/kernel`
- PI05 时间 MLP
  - `time_mlp_in/bias`
  - `time_mlp_in/kernel`
  - `time_mlp_out/bias`
  - `time_mlp_out/kernel`

所以如果只问“LoRA 微调时哪些参数在更新”，对当前这套实现，更准确的答案是：

- `LoRA 参数`
- `PaliGemma.img` 分支参数
- `action_in_proj / action_out_proj`
- `time_mlp_in / time_mlp_out`

而不是只有：

- `*lora*`

## 常用 OpenPI05 LoRA 训练配置

[`src/openpi/training/config.py`](../src/openpi/training/config.py) 里主要的 OpenPI05 LoRA 配置有：

- `pi05_square_d1_lora`
- `pi05_can_mh_lora`
- `pi05_square_d1_core_lora_top80`
- `pi05_square_d1_core_lora_1000`

### `pi05_can_mh_lora`

这是近期实验里最常用的一套。

关键设置如下：

- model
  - `pi05=True`
  - `paligemma_variant="gemma_2b_lora"`
  - `action_expert_variant="gemma_300m_lora"`
  - `action_horizon=20`
- dataset
  - repo：`/workspace/liushuo/cupid/datasets/lerobot/can_mh`
  - 开启 `use_delta_joint_actions=True`
- weight loader
  - base checkpoint：`gs://openpi-assets/checkpoints/pi05_base/params`
- optimization
  - `batch_size=32`
  - `num_train_steps=10000`
  - cosine lr schedule
  - warmup：`1000`
  - peak lr：`1e-4`
  - decay lr：`1e-5`
- EMA
  - `ema_decay=None`

## LoRA checkpoint 是怎么加载的

权重加载逻辑在 [`src/openpi/training/weight_loaders.py`](../src/openpi/training/weight_loaders.py)。

`CheckpointWeightLoader` 的行为是：

1. 先加载 checkpoint 里的参数
2. 再与当前模型参数树做 merge
3. 对缺失且匹配 `.*lora.*` 的 key，用当前模型初始化值补齐

这一步很重要，因为 base checkpoint 本身并不包含新加的 LoRA 权重。缺失的 LoRA 参数会从当前模型初始化状态补进来。

## 训练时 `trainable_filter` 是怎么起作用的

训练逻辑在 [`scripts/train.py`](../scripts/train.py)。

关键行为有：

- 冻结参数会先转成 `bfloat16`
- optimizer state 只为 `params.filter(config.trainable_filter)` 创建
- 梯度只对 `config.trainable_filter` 里的参数求
- update 也只应用在这个 trainable 子集上

所以工程上真正该关注的是：

`当前 trainable_filter 到底选中了哪些参数`

而不是只看配置名里有没有 `_lora`。

## 实验中观察到的现象

对 `pi05_can_mh_lora`，实际 trainable leaves 包括：

- `PaliGemma/llm/.../lora_*` 下的 Gemma LoRA 参数
- `PaliGemma/img/...` 下的图像侧参数
- 动作输入输出投影层
- PI05 的时间 MLP

这进一步说明当前这套训练不是严格意义上的纯 LoRA-only。

## 基于 LoRA 微调之上的 influence / 数据筛选脚本

仓库里这些分析脚本都复用了同一套 `trainable_filter` 逻辑，因此它们里的“LoRA influence”也应该按同样方式理解：

- [`scripts/influence_pi05_lora.py`](../scripts/influence_pi05_lora.py)
- [`scripts/influence_pi05_lora_grass.py`](../scripts/influence_pi05_lora_grass.py)
- [`scripts/influence_pi05_lora_grass_window.py`](../scripts/influence_pi05_lora_grass_window.py)
- [`scripts/influence_pi05_lora_grass_leaf_subset.py`](../scripts/influence_pi05_lora_grass_leaf_subset.py)

也就是说，这些分数通常对应的是：

- 当前 trainable 参数子空间上的 influence

而不一定是：

- 纯 LoRA 参数子空间上的 influence

## 更准确的理解方式

对当前仓库，我建议把它理解为：

- 模型架构：OpenPI05
- 适配器：Gemma attention / MLP 上的 LoRA
- 实际优化范围：LoRA 参数 + 所有未被 `freeze_filter` 冻结的外围模块参数

如果实验表述需要更严谨，建议写成：

- “OpenPI05 上基于 Gemma LoRA adapter 的低内存微调”

而不要默认写成：

- “严格 LoRA-only 微调”

## 推荐阅读顺序

如果你想从代码上完整追一遍，建议按下面顺序看：

1. [`src/openpi/models/lora.py`](../src/openpi/models/lora.py)
2. [`src/openpi/models/gemma.py`](../src/openpi/models/gemma.py)
3. [`src/openpi/models/pi0_config.py`](../src/openpi/models/pi0_config.py)
4. [`src/openpi/models/pi0.py`](../src/openpi/models/pi0.py)
5. [`src/openpi/training/config.py`](../src/openpi/training/config.py)
6. [`src/openpi/training/weight_loaders.py`](../src/openpi/training/weight_loaders.py)
7. [`scripts/train.py`](../scripts/train.py)
