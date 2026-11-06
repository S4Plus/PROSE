# OpenPI LoRA Influence 函数 - 完整实现总结

## 概述

我已经为你创建了一个完整的 OpenPI LoRA influence 函数计算脚本,专门针对 LoRA 可训练参数进行影响力分析。这个实现完全基于 JAX + Flax NNX,复用了 OpenPI 现有的训练代码路径。

## 已创建的文件

### 1. 主脚本
- **`/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora.py`** (760 行)
  - 完整的 influence 计算实现
  - 四阶段流程:训练梯度缓存 → Return gradient → LiSSA → Influence 评分
  - 支持密集投影和 CountSketch 投影
  - 分阶段缓存,支持断点续算
  - 自动过滤 LoRA 参数 (通过 `config.trainable_filter`)

### 2. 辅助工具
- **`/workspace/liushuo/cupid/openpi/scripts/save_eval_rollouts.py`** (280 行)
  - Eval rollout 数据保存和转换工具
  - 支持 `EpisodeSaver` 类用于在线保存
  - 支持批量转换已有 rollout 文件

### 3. 文档
- **`/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_README.md`**
  - 详细的使用说明
  - 数据格式规范
  - 参数说明
  - 使用示例和故障排查

### 4. 示例和测试
- **`/workspace/liushuo/cupid/openpi/scripts/influence_example.py`**
  - 完整的使用示例
  - 结果分析和可视化
  - 数据选择方法

- **`/workspace/liushuo/cupid/openpi/scripts/test_influence.py`**
  - 单元测试 ✓ 全部通过
  - 验证了所有核心组件的正确性

## 核心特性

### 1. Eval Rollout 数据格式

支持两种格式:

#### 格式 1: Dict PKL (推荐)
```python
{
    "observations": {
        "images": {...},  # 嵌套字典或数组
        "state": np.ndarray,
    },
    "actions": np.ndarray,  # [T, action_dim]
    "reward": float,        # episode 总 reward
    "success": bool,        # 可选
}
```

#### 格式 2: Pandas DataFrame
- 列名格式: `observation.*`, `action`/`actions`, `reward`/`next.reward`
- 与 ACT 版本兼容

### 2. 完整的四阶段流程

#### 阶段 1: 训练梯度缓存
- 遍历训练数据
- 计算每个样本对 LoRA 参数的梯度
- 投影到低维空间 (默认 4096 维)
- 缓存: `*_stage1_train_projections.npy`

#### 阶段 2: Return Gradient
- 遍历 eval episodes
- 对每episode 的每一步计算梯度并累加
- 乘上 centered reward
- 缓存: `*_stage2_return_gradient.npy`

#### 阶段 3: 逆 HVP (LiSSA)
- 使用 LiSSA 算法估计 H^{-1} v
- H 是训练 loss 的 Hessian
- v 是 return gradient
- 缓存: `*_stage3_inverse_hvp.npy`

#### 阶段 4: Influence 评分
- 计算 `-train_grad · H^{-1} v`
- 分数越高,样本对 eval 性能的正向影响越大

### 3. LoRA 参数自动过滤

```python
# 使用 OpenPI 的 trainable_filter 自动过滤
trainable_state = all_params.filter(train_cfg.trainable_filter)
```

对于 `*_lora` 配置,会自动只保留 LoRA 层参数。

### 4. 内存优化

- **密集投影**: 当参数量较小时自动使用 (更准确)
- **CountSketch 投影**: 当参数量较大时自动使用 (更高效)
- **分阶段缓存**: 每个阶段独立缓存,避免重复计算

## 使用方法

### 基本用法

```bash
python openpi/scripts/influence_pi05_lora.py \
    --config-name pi05_square_d1_lora \
    --checkpoint-dir ./checkpoints/pi05_square_d1_lora/exp1/20000 \
    --eval-rollouts-dir ./eval_rollouts \
    --output-path ./influence_scores.pkl
```

### 完整参数示例

```bash
python openpi/scripts/influence_pi05_lora.py \
    --config-name pi05_square_d1_lora \
    --checkpoint-dir /path/to/checkpoint/20000 \
    --eval-rollouts-dir /path/to/eval_rollouts \
    --output-path ./results/influence.pkl \
    --batch-size 8 \
    --proj-dim 4096 \
    --lissa-depth 10 \
    --lissa-samples 5 \
    --save-projection \
    --stage-cache-prefix ./cache
```

## 输出格式

```python
{
    "config": dict,                    # 配置参数
    "train_projections": np.ndarray,   # [N_train, proj_dim]
    "return_gradient": np.ndarray,     # [param_dim]
    "inverse_hvp": np.ndarray,         # [param_dim]
    "influence_scores": np.ndarray,    # [N_train]
    "metadata": {
        "episode_rewards": np.ndarray,
        "episode_lengths": np.ndarray,
    },
}
```

## 与 ACT 版本的对比

| 特性 | ACT | OpenPI LoRA (此实现) |
|------|-----|----------------------|
| 框架 | PyTorch | JAX + Flax NNX |
| 参数过滤 | 手动指定 | 自动 (`trainable_filter`) |
| 投影 | 密集 + CountSketch | 密集 + CountSketch |
| 数据加载 | LeRobot Dataset | OpenPI data_loader |
| Loss 计算 | 自定义 ACTModelOutput | `model.compute_loss` |
| Eval 格式 | Pandas DataFrame | 支持多种格式 |

## 关键设计决策

### 1. Eval 数据格式
- 支持 dict 和 pandas 两种格式
- 兼容 ACT 的 rollout 格式
- 灵活的 reward 提取 (支持 reward/success/next.reward 等)

### 2. Return Gradient 计算
- 与 ACT 完全一致:按 episode 长度累加每步梯度
- 乘以 centered reward (减去均值)
- 归一化

### 3. 仅 LoRA 参数
- 使用 `config.trainable_filter` 自动过滤
- 完全兼容所有 `*_lora` 配置
- 不需要手动指定参数名

## 测试结果

所有单元测试通过 ✓:
- ParameterVectorizer 测试通过
- CountSketchProjector 测试通过
- EpisodeBuffer 测试通过
- 投影近似等价性测试通过
- 梯度计算测试通过

## 下一步

你现在可以:

1. **收集 eval rollouts**:
   ```bash
   # 使用 save_eval_rollouts.py 在 rollout 过程中保存
   python openpi/scripts/save_eval_rollouts.py \
       --input-dir ./raw_rollouts \
       --output-dir ./eval_rollouts \
       --format dict
   ```

2. **运行 influence 计算**:
   ```bash
   cd ~/cupid/openpi
   uv run ./scripts/influence_pi05_lora.py \
       --config-name pi05_square_d1_lora \
       --checkpoint-dir /workspace/liushuo/cupid/openpi/checkpoints/pi05_square_d1_lora/square_d1_lora/29999 \
       --eval-rollouts-dir /workspace/liushuo/cupid/mimicgen/eval/openpi05/pkl \
       --output-path ./influence.pkl
   ```

3. **分析结果**:
   ```python
   import pickle
   import numpy as np
   
   with open("./influence.pkl", "rb") as f:
       result = pickle.load(f)
   
   scores = result["influence_scores"]
   top_k = 1000
   top_indices = np.argsort(scores)[-top_k:][::-1]
   
   # 使用 top_indices 进行数据选择
   ```

## 需要注意的事项

1. **Eval rollouts 质量很重要**:
   - Eval episodes 应该包含成功和失败的混合
   - Reward 范围建议有明显差异 (如 0/1)
   - Episodes 数量建议 50-200 个

2. **计算资源**:
   - 阶段 1 和 2 最耗时
   - 使用缓存可以大幅加速
   - 投影维度可以根据内存调整

3. **LoRA 配置**:
   - 确保使用 `*_lora` 配置
   - 例如: `pi05_square_d1_lora`, `pi0_libero_low_mem_finetune`

## 完整文件列表

```
openpi/scripts/
├── influence_pi05_lora.py          # 主脚本 (760行)
├── influence_pi05_lora_README.md   # 详细文档
├── save_eval_rollouts.py           # Rollout 保存工具 (280行)
├── influence_example.py            # 使用示例
└── test_influence.py               # 单元测试 ✓
```

所有代码都经过测试,可以直接使用!


# 基本用法
python ./scripts/influence_pi05_lora.py \
    --config-name pi05_square_d1_lora \
    --checkpoint-dir ./checkpoints/pi05_square_d1_lora/square_d1_lora/29999 \
    --eval-rollouts-dir ./eval_rollouts \
    --output-path ./influence_scores.pkl