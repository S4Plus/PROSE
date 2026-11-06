# ACT 实验模块

这里集中 ACT 策略的数据影响分析、轨迹质量评估、数据筛选与再训练。完整环境和步骤以 [整体 README](../README.md)、[复现指南](../docs/reproduction.md) 为准。

## 主要入口

| 阶段 | 实现 |
| --- | --- |
| ACT 数据、模型加载和 TRAK 辅助 | [train_trak_act.py](train_trak_act.py) |
| 主流程的窗口影响打分 | [myscripts/cupid_act_influence.py](myscripts/cupid_act_influence.py) |
| episode + window 联合选择 | [myscripts/filter_joint_episode_window.py](myscripts/filter_joint_episode_window.py) |
| 联合筛选后的重加权训练 | [train_act_cupid_reweight_joint.py](train_act_cupid_reweight_joint.py) |
| 轨迹质量与 smoothness 评估 | [eval_cupid_quality.py](eval_cupid_quality.py) |
| 基于 pair scores 的 CUPID-QUALITY | [cupid_quality/README.md](cupid_quality/README.md) |
| 随机 state-action 比例对照 | [comparison_random_state_action_ratio/README.md](comparison_random_state_action_ratio/README.md) |

主流程从项目根目录使用 `python scripts/reproduce_act.py --stage <阶段>`。更细参数使用上述脚本的 `--help`。

## 实验版本与任务目录

[cupid_act_influence.py](cupid_act_influence.py) 保留早期影响实现；`*_last_layers.py` 和 `*_strict_last_layers.py` 保留参数层范围对照。主入口使用 `myscripts/cupid_act_influence.py`，运行时记录具体版本并使用独立缓存目录。

`robomimic/`、`mymimicgen/`、`libero/`、`robotwin/`、`franka/` 和 `swk/` 是任务或设备实验。其脚本中可能保留来源机器的绝对路径、模型目录和 GPU 分配。

原 `readme.md` 方法笔记已保存为 [METHOD_NOTES.md](METHOD_NOTES.md)。影响函数和联合筛选的详细方法见 [cupid_act_influence.md](cupid_act_influence.md) 与 [joint_episode_window_method.md](myscripts/joint_episode_window_method.md)。
