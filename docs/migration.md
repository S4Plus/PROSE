# 结构整理与迁移说明

## 整理范围

结构整理基于原仓库 `cupid` 分支的本地工作区，整理后的代码发布于 [S4Plus/PROSE](https://github.com/S4Plus/PROSE)。保留框架主体、算法版本、已有研究资料和许可证；物理归类的是根目录入口、历史文档和批处理脚本。此次发布的来源与范围见 [发布说明](publication.md)。

完整旧路径到新路径的映射位于 [migration.json](migration.json)。该文件也由结构检查使用。

| 原位置 | 新位置 |
| --- | --- |
| 根目录的失败分析、聚类、热力图和位置分析脚本 | `scripts/analysis/` |
| 根目录的 LeRobot TRAK、示范/损失评估、随机训练脚本 | `scripts/lerobot/` |
| 根目录 `ros_pi0_*.py` | `scripts/ros/` |
| 根目录原版 Diffusion Policy 训练、评估、TRAK 入口 | `scripts/legacy_diffusion/` |
| 原 `scripts/train/*.sh`、`scripts/eval/*.sh` | `scripts/legacy_diffusion/launch/{train,eval}/` |
| 根目录 `hf_datasets_compat.py` 实现 | `prose/compat/hf_datasets.py`；原位置保留小型导入兼容入口 |
| 根目录 `random_train_patch.py` | `prose/compat/random_subset.py` |
| `vis_pkl.py` | `scripts/data/inspect_pickle.py` |
| `push_update.sh` | `scripts/maintenance/push_update.sh` |
| `sh/` | `scripts/experiments/` |
| `2_docs/` | `docs/research/` |
| `readme/`、旧 `README.md`、`workflow.md`、`exec.md` | `docs/archive/` |
| 临时 `test_small.py`、`test_sitecustomize_root.py`、`.txt` 样例 | `docs/archive/scratch/` |
| `conda_environment.yaml` | `environments/legacy_diffusion.yaml` |
| `act/readme.md` | `act/METHOD_NOTES.md`；新增 `act/README.md` 作为模块索引 |
| `openpi/readme.md` | `openpi/PROJECT_NOTES.md`；恢复原 `openpi/README.md` 框架说明 |

## 命令迁移举例

```bash
# 原 python random_train.py ...
python scripts/lerobot/random_train.py --help

# 原 python cluster_failures.py ...
python scripts/analysis/cluster_failures.py --help

# 原 python train_trak_lerobot.py ...
python scripts/lerobot/train_trak_lerobot.py --help

# 新增主 ACT 分阶段入口
python scripts/reproduce_act.py --stage score
```

旧根目录命令应按映射更新。本仓库内相关 Python 导入、随机训练补丁的子进程导入以及历史 shell 入口引用已经同步调整；仓库之外的作业脚本需要使用新路径。

## 路径与安装修复

- `setup.py` 原先将整个混合仓库声明为 `diffusion_policy`，但该核心目录不存在。现在安装为 `prose-tools`，仅包含项目工具和 ACT 包；策略框架按其自己的安装入口配置。
- 公共路径工具支持按仓库文件位置定位 LeRobot，并通过 `LEROBOT_REPO` 覆盖。
- ACT 主要打分、联合筛选、联合训练，以及 ACT/OpenPI/OpenVLA 三个主要联合启动脚本已移除其入口位置依赖的服务器根路径。
- 根目录迁移脚本保留独立执行时的仓库定位，更新了跨模块导入和与文件位置有关的配置路径。
- 随机训练子进程显式继承当前仓库的 Python 路径，以加载迁移后的补丁。
- OpenPI 两份只在大小写上不同的 README 从 Git 原始内容分别恢复，避免 Windows 文件系统覆盖其中一份。
- 新增数据、模型、缓存和实验输出忽略规则，避免把大体积运行产物混进源码。
- Bash 文件通过 `.gitattributes` 固定 LF 行尾；手动维护脚本改为推送当前分支。

Windows Git 默认忽略文件名大小写，可能把删除的 `readme.md` 与新的 `README.md` 显示为同一个文件修改。检查相关迁移可用 `git -c core.ignorecase=false status --short`；手动提交时也可使用 `git -c core.ignorecase=false add -A`，使大小写文件迁移被准确记录。当前发布版本已将大小写冲突文件分别保存。

## 验证范围

可重复执行的轻量检查：

```bash
python scripts/check_repository.py
python -m unittest discover -s tests/repository -v
```

结构检查只解析维护入口的语法、验证迁移目标和主要文档链接，不导入大型模型。测试覆盖复现命令的阶段参数、相对路径基准、配置错误、缺失上游产物、禁止覆盖训练输出及运行记录；质量计算另用合成 pair scores 检查实际工具输出。

未完成的运行验证包括：Linux/CUDA 环境的全量依赖安装、真实 HDF5/视频加载、ACT/OpenPI/OpenVLA 训练、MuJoCo rollout 以及论文指标复核。当前工作区不具备对应数据、权重和训练环境。

## 保留的限制

- 旧 Diffusion Policy 核心、训练配置和 curation notebook 不在该分支，历史环境与当前 TRAK 存在依赖冲突。
- 多个框架内仍有历史实验绝对路径。此轮优先修复主入口，没有批量改写任务专用配置、外部数据路径或未知协议。
- 多个影响版本和仿真第三方来源目录仍保留，它们可能用于已有实验；没有依据文件名相似就删除。
- `docs/archive/`、`docs/research/` 和旧方法笔记保留来源命令，其中路径和链接仅作为历史参考。
- ROS 2、Isaac Sim、实机 Franka、LIBERO 和 RoboTwin 需要各自运行环境与资源，不包含在通用 ACT 示例中。
