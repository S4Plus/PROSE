# MimicGen Client Rollout 保存功能 - 修改总结

## ✅ 已完成的修改

成功为 MimicGen evaluation client 添加了保存 eval rollout 数据的功能,用于 OpenPI LoRA influence 计算。

## 修改的文件

### 1. **client_mimicgen_openpi05.py** (OpenPI05 版本)
**修改内容:**
- ✅ 添加 `import pickle`
- ✅ 添加 `--save_rollouts_dir` 参数
- ✅ 添加 `--save_rollouts_format` 参数
- ✅ 在 `rollout_episode` 函数中添加数据收集:
  - 收集观测 (OpenPI 格式化后的观测)
  - 收集动作
  - 收集奖励
- ✅ 保存为 dict PKL 格式,与 `influence_pi05_lora.py` 兼容

### 2. **client_mimicgen_act.py** (ACT 版本)
**修改内容:**
- ✅ 添加 `import pickle`
- ✅ 添加 `--save_rollouts_dir` 参数
- ✅ 添加 `--save_rollouts_format` 参数
- ✅ 在 `rollout_episode` 函数中添加数据收集:
  - 收集观测 (原始观测)
  - 收集动作
  - 收集奖励
- ✅ 保存为 dict PKL 格式

## 保存的数据格式

```python
{
    "observations": {
        # 嵌套字典,根据环境和配置不同
        # OpenPI05: "observation.images.agentview", "observation.images.wrist", "observation.state", "state"
        # ACT: 原始观测字段
    },
    "actions": np.ndarray,      # [T, action_dim]
    "reward": float,            # episode 总 reward
    "success": bool,            # 成功标志
    "rewards": np.ndarray,      # [T] 每步的 reward
}
```

## 使用示例

### OpenPI05

```bash
python /workspace/liushuo/cupid/mimicgen/eval/client_mimicgen_openpi05.py \
    --num_episodes 100 \
    --save_rollouts_dir ./eval_rollouts_openpi05 \
    --openpi_obs \
    --use_camera_obs
```

### ACT

```bash
python /workspace/liushuo/cupid/mimicgen/eval/client_mimicgen_act.py \
    --num_episodes 100 \
    --save_rollouts_dir ./eval_rollouts_act \
    --use_camera_obs
```

## 完整工作流程

1. **运行 eval + 保存 rollouts**:
   ```bash
   python client_mimicgen_openpi05.py --num_episodes 100 --save_rollouts_dir ./eval_rollouts
   ```

2. **计算 influence scores**:
   ```bash
   cd /workspace/liushuo/cupid/openpi
   python ./scripts/influence_pi05_lora.py \
       --config-name pi05_square_d1_lora \
       --checkpoint-dir ./checkpoints/pi05_square_d1_lora/square_d1_lora/29999 \
       --eval-rollouts-dir ./eval_rollouts \
       --output-path ./influence_scores.pkl
   ```

3. **分析结果**:
   ```python
   import pickle
   with open("./influence_scores.pkl", "rb") as f:
       result = pickle.load(f)
   top_samples = np.argsort(result["influence_scores"])[-100:][::-1]
   ```

## 关键特性

1. **自动格式兼容**: 保存的格式完全兼容 `influence_pi05_lora.py` 的 `EvalRolloutsDataset`
2. **OpenPI 观测格式化**: OpenPI05 client 自动格式化观测为标准格式
3. **灵活配置**: 支持选择是否保存,以及保存路径
4. **内存高效**: 每个 episode 独立保存,避免内存溢出

## 文档

- **使用指南**: `/workspace/liushuo/cupid/mimicgen/eval/ROLLOUT_SAVE_README.md`
- **Influence 脚本文档**: `/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_README.md`
- **实现总结**: `/workspace/liushuo/cupid/openpi/scripts/IMPLEMENTATION_SUMMARY.md`

## 下一步

现在你可以:

1. 启动 policy server
2. 运行 client 进行 evaluation 并保存 rollouts
3. 使用保存的 rollouts 运行 influence 计算
4. 分析 influence scores 选择高质量训练数据

所有功能都已完成并可以直接使用! 🎉
