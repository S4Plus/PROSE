# OpenPI LoRA Influence 函数计算

这个脚本用于计算 OpenPI LoRA 模型的 influence 分数,帮助识别训练数据中对模型性能影响最大的样本。

## 功能特性

1. **LoRA 参数专用**: 只针对 LoRA 可训练参数计算影响力
2. **高效投影**: 支持密集投影和 CountSketch 投影,自动选择
3. **分阶段缓存**: 每个计算阶段都有缓存,支持断点续算
4. **灵活的 Eval 格式**: 支持多种 rollout 数据格式

## 使用方法

### 基本用法

```bash
python openpi/scripts/influence_pi05_lora.py \
    --config-name pi05_square_d1_lora \
    --checkpoint-dir ./checkpoints/pi05_square_d1_lora/exp1/20000 \
    --eval-rollouts-dir ./eval_rollouts \
    --output-path ./influence_scores.pkl
```

### 完整参数

```bash
python openpi/scripts/influence_pi05_lora.py \
    --config-name pi05_square_d1_lora \
    --checkpoint-dir ./checkpoints/pi05_square_d1_lora/exp1/20000 \
    --eval-rollouts-dir ./eval_rollouts \
    --output-path ./results/influence.pkl \
    --batch-size 8 \
    --hvp-batch-size 4 \
    --proj-dim 4096 \
    --lissa-depth 8 \
    --lissa-samples 4 \
    --lissa-scale 10.0 \
    --lissa-damping 0.01 \
    --seed 42 \
    --save-projection \
    --stage-cache-prefix ./cache
```

## Eval Rollout 数据格式

脚本支持两种 eval rollout 格式:

### 格式 1: Dict PKL (推荐)

每个 episode 保存为 `episode_XXXXX.pkl`,包含:

```python
{
    "observations": {
        "images": np.ndarray,  # [T, ...] 或嵌套字典
        "state": np.ndarray,   # [T, state_dim]
        # ... 其他观测
    },
    "actions": np.ndarray,     # [T, action_dim]
    "reward": float,           # episode 总 reward (可选)
    "success": bool,           # 成功标志 (可选,当 reward 不存在时使用)
}
```

### 格式 2: Pandas DataFrame PKL

每个 episode 保存为 PKL,内容为 `pd.DataFrame`,列名遵循:

- `observation.*`: 观测数据
- `action` 或 `actions`: 动作数据  
- `reward` 或 `next.reward`: 奖励
- `success` 或 `next.success`: 成功标志

## 输出格式

输出的 PKL 文件包含:

```python
{
    "config": dict,                    # 配置参数
    "train_projections": np.ndarray,   # [N_train, proj_dim] 训练样本投影梯度
    "return_gradient": np.ndarray,     # [param_dim] Return gradient
    "inverse_hvp": np.ndarray,         # [param_dim] 逆 HVP
    "influence_scores": np.ndarray,    # [N_train] 每个训练样本的影响力分数
    "metadata": {
        "episode_rewards": np.ndarray, # [N_eval] Eval episode rewards
        "episode_lengths": np.ndarray, # [N_eval] Eval episode 长度
    },
    # 可选项 (如果 --save-projection)
    "projection_matrix": np.ndarray,   # 投影矩阵 (密集投影)
    "projection_state": dict,          # CountSketch 状态 (稀疏投影)
}
```

## 四阶段流程

### 阶段 1: 训练梯度缓存

- 遍历训练数据,计算每个样本对 LoRA 参数的梯度
- 投影到低维空间 (proj_dim)
- 缓存: `<output>_stage1_train_projections.npy`

### 阶段 2: Return Gradient 计算

- 遍历 eval episodes
- 对每个 episode 的每一步计算梯度并累加
- 乘上 centered reward
- 缓存: `<output>_stage2_return_gradient.npy`

### 阶段 3: 逆 HVP 求解 (LiSSA)

- 使用 LiSSA 算法估计 H^{-1} v
- H 是训练 loss 的 Hessian
- v 是 return gradient
- 缓存: `<output>_stage3_inverse_hvp.npy`

### 阶段 4: Influence 评分

- 计算 `-train_grad · H^{-1} v`
- 分数越高,样本对 eval 性能的正向影响越大

## 参数说明

### 数据参数

- `--config-name`: OpenPI 训练配置名 (必须是 `*_lora` 配置)
- `--checkpoint-dir`: Checkpoint 目录 (包含 `params` 子目录)
- `--eval-rollouts-dir`: Eval rollout 保存目录
- `--output-path`: 输出文件路径

### 投影参数

- `--proj-dim`: 投影维度,默认 4096
  - 设为 0 禁用投影 (内存占用极大)
  - 越大越准确,但内存占用越高

### LiSSA 参数

- `--lissa-depth`: LiSSA 迭代深度,默认 8
- `--lissa-samples`: LiSSA 样本数,默认 4
- `--lissa-scale`: 缩放因子,默认 10.0
- `--lissa-damping`: 阻尼系数,默认 0.01

### 其他参数

- `--batch-size`: 训练数据 batch size
- `--hvp-batch-size`: HVP 计算的 batch size
- `--seed`: 随机种子
- `--save-projection`: 保存投影矩阵
- `--stage-cache-prefix`: 阶段缓存保存目录

## 使用示例

### 示例 1: 从头计算

```bash
python openpi/scripts/influence_pi05_lora.py \
    --config-name pi05_square_d1_lora \
    --checkpoint-dir /path/to/checkpoint/20000 \
    --eval-rollouts-dir /path/to/eval_rollouts \
    --output-path ./results/influence.pkl \
    --proj-dim 4096 \
    --lissa-depth 10 \
    --lissa-samples 5
```

### 示例 2: 使用缓存加速

```bash
# 第一次运行
python openpi/scripts/influence_pi05_lora.py \
    --config-name pi05_square_d1_lora \
    --checkpoint-dir /path/to/checkpoint/20000 \
    --eval-rollouts-dir /path/to/eval_rollouts \
    --output-path ./results/influence.pkl \
    --stage-cache-prefix ./cache

# 如果中断,再次运行会从缓存恢复
python openpi/scripts/influence_pi05_lora.py \
    --config-name pi05_square_d1_lora \
    --checkpoint-dir /path/to/checkpoint/20000 \
    --eval-rollouts-dir /path/to/eval_rollouts \
    --output-path ./results/influence.pkl \
    --stage-cache-prefix ./cache
```

### 示例 3: 分析结果

```python
import pickle
import numpy as np

# 加载结果
with open("./results/influence.pkl", "rb") as f:
    result = pickle.load(f)

influence_scores = result["influence_scores"]

# 找出最有影响力的样本
top_k = 100
top_indices = np.argsort(influence_scores)[-top_k:][::-1]

print(f"Top {top_k} 最有影响力的训练样本:")
for idx in top_indices:
    print(f"  Sample {idx}: score = {influence_scores[idx]:.6f}")

# 找出负面影响最大的样本
bottom_k = 50
bottom_indices = np.argsort(influence_scores)[:bottom_k]

print(f"\nTop {bottom_k} 负面影响最大的训练样本:")
for idx in bottom_indices:
    print(f"  Sample {idx}: score = {influence_scores[idx]:.6f}")
```

## 注意事项

1. **内存需求**: 
   - 投影维度 4096 时,内存占用较小
   - 不使用投影 (proj_dim=0) 会导致极高内存占用

2. **计算时间**:
   - 阶段 1 和 2 是最耗时的
   - 使用缓存可以大幅加速重复实验

3. **LoRA 配置**:
   - 确保使用的是 `*_lora` 配置
   - 脚本会自动过滤只保留 LoRA 参数

4. **Eval Rollout 质量**:
   - Eval episodes 应该是失败和成功的混合
   - Reward 范围建议有明显差异 (如 0/1)

## 性能优化建议

1. 增大 `--hvp-batch-size` 可加速 LiSSA,但需要更多内存
2. 减小 `--proj-dim` 可减少内存占用,但可能影响准确度
3. 使用 `--stage-cache-prefix` 避免重复计算
4. 调整 `--lissa-depth` 和 `--lissa-samples` 平衡精度和速度

## 与 ACT 版本的对比

| 特性 | ACT (`cupid_act_influence.py`) | OpenPI LoRA (此脚本) |
|------|--------------------------------|----------------------|
| 框架 | PyTorch | JAX + Flax NNX |
| 参数过滤 | 手动指定 | 自动 (`trainable_filter`) |
| 投影 | 密集 + CountSketch | 密集 + CountSketch |
| 数据加载 | LeRobot | OpenPI data_loader |
| Loss 计算 | 自定义 ACTModelOutput | `model.compute_loss` |

## 故障排查

### 问题: OOM (内存不足)

**解决方案**:
- 减小 `--proj-dim`
- 减小 `--batch-size`
- 减小 `--hvp-batch-size`

### 问题: Checkpoint 加载失败

**解决方案**:
- 检查 `--checkpoint-dir` 路径是否包含 `params` 子目录
- 确认配置名与 checkpoint 匹配

### 问题: Eval rollouts 加载失败

**解决方案**:
- 检查 PKL 文件格式是否符合要求
- 查看日志中的详细错误信息

### 问题: LiSSA 不收敛

**解决方案**:
- 增大 `--lissa-depth`
- 增大 `--lissa-samples`
- 调整 `--lissa-damping` (通常 0.001-0.1)
