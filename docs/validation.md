# 本次整理的验证记录

日期：2026-10-07。主机：Windows；检查解释器为应用提供的 Python 3.12.14。

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| `python scripts/check_repository.py` | 通过 | 51 个维护入口/工具的 Python 语法、迁移目标与 8 份主要文档的本地链接 |
| `python -m unittest discover -s tests/repository -v` | 10 项通过 | 配置与路径、实际 CLI 参数、缺失产物、输出覆盖检查、真实子进程日志及失败退出记录 |
| Bash `-n` | 19 个脚本通过 | 迁移后的启动脚本及 ACT/OpenPI/OpenVLA 三个主要联合启动器的语法 |
| CUPID-QUALITY 命令行 | 通过 | 从合成 pair-score NPZ 读取、分组聚合、归一化并写出 CSV/JSON/列表 |
| 文件迁移完整性 | 通过 | 克隆中的 2434 个已跟踪文件均在原位置或对应迁移目标存在 |

质量工具检查输入为 2 个 rollout episode、2 个训练 episode，每个 episode 有 2 个 state-action pair。手算的轨迹长度归一化质量为 `[0.5, 1.0]`，实际 CSV 相同；min-max 后为 `[0.0, 1.0]`。

该合成数据检查使用 NumPy 2.3.5，只验证独立质量工具；它不是 ACT 或仿真环境的兼容性证明。正式环境按 [环境配置](environments.md) 准备。

没有安装完整实验环境或执行 GPU 训练。真实数据转换、视频解码、MuJoCo rollout、各策略训练与成功率复核仍需在相应环境和数据上运行。
