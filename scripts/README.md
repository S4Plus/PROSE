# 脚本索引

从仓库根目录调用脚本，或在安装 `prose-tools` 后使用对应 Python 模块。通用运行方式见 [整体 README](../README.md)。

| 位置 | 作用 | 入口举例 |
| --- | --- | --- |
| [reproduce_act.py](reproduce_act.py) | 根据 JSON 配置预览、检查或执行 ACT 一个阶段 | `python scripts/reproduce_act.py --stage score` |
| [check_repository.py](check_repository.py) | 检查语法、迁移目标、主要文档；可列出环境包 | `python scripts/check_repository.py` |
| [analysis](analysis/) | 失败聚类、影响与位置分析、采集热力图 | `cluster_failures.py`、`build_failure_heatmap.py` |
| [lerobot](lerobot/) | LeRobot diffusion TRAK、损失/示范打分、随机子集训练 | `train_trak_lerobot.py`、`random_train.py` |
| [train](train/) | ACT 子集再训练 | `retrain_act_subset.py` |
| [data](data/) | 单个 rollout PKL 的结构检查 | `inspect_pickle.py` |
| [experiments](experiments/) | 整理前的实验批处理命令 | `random_train.sh`、`act_train_0.01.sh` |
| [ros](ros/) | ROS 2 与 Isaac Sim 实验接口 | `ros_pi0_eval.py`、`ros_pi0_inference.py` |
| [maintenance](maintenance/) | 手动提交/推送辅助脚本 | `push_update.sh`；执行会修改 Git 并推送 |
| [legacy_diffusion](legacy_diffusion/) | 依赖缺失核心包的旧 Diffusion Policy 入口 | `train.py`、`launch/train/train_policies.sh` |

原有 `scripts/` 顶层数据分析与跨框架工具继续保留，包括：

- [compare_act_openpi_selection.py](compare_act_openpi_selection.py)：比较 ACT 与 OpenPI 的选择结果。
- [build_openpi05_canph_ablation.py](build_openpi05_canph_ablation.py)：构建补充数据对照。
- [visualize_lerobot_dataset.py](visualize_lerobot_dataset.py)：数据集可视化。
- [visualize_online_trak_influence.py](visualize_online_trak_influence.py)：影响分数可视化。

通用工具使用 `--help` 查看真实参数。历史工具包含原机器路径或需要特定模型与依赖，使用前核实。ROS 工具需要 ROS 2 / Isaac Sim 专用环境；它们未纳入通用 ACT 环境。
