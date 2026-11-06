# 代码结构与模块职责

## 分层方式

本仓库按“公共工具 → 实验入口 → 策略实现 → 数据与仿真框架 → 文档”组织。已有框架保持原位置；项目根目录主要提供说明、安装入口和一份兼容工具。

| 目录 | 职责 | 关键入口 |
| --- | --- | --- |
| `prose/` | 仓库路径解析、Hugging Face 数据特征兼容、随机子集补丁 | [paths.py](../prose/paths.py)、[compat](../prose/compat/) |
| `configs/reproduction/` | 主复现路线的用户配置 | [act.example.json](../configs/reproduction/act.example.json) |
| `environments/` | 分路线依赖文件与历史环境 | [环境说明](environments.md) |
| `scripts/` | 项目级命令、通用分析工具和历史入口 | [脚本索引](../scripts/README.md) |
| `act/` | ACT 影响函数、质量评估、筛选、重加权训练 | [模块索引](../act/README.md) |
| `lerobot_2.1/` | 本地修改的 LeRobot 0.3.4 | [pyproject.toml](../lerobot_2.1/pyproject.toml) |
| `mymimicgen/` | 自定义数据转换 | [3_convert.py](../mymimicgen/3_convert.py) |
| `mimicgen/` | MimicGen 框架、robosuite/robomimic、HTTP 仿真评估 | [eval](../mimicgen/eval/)、[README](../mimicgen/README.md) |
| `openpi/` | OpenPI 框架、π₀.₅ LoRA 实验与再训练 | [README](../openpi/README.md)、[myscripts](../openpi/myscripts/) |
| `openvla-oft/` | OpenVLA-OFT 框架、LeRobot 微调、影响与筛选 | [README](../openvla-oft/README.md)、[vla-scripts](../openvla-oft/vla-scripts/) |
| `dropout/` | ACT 随机性、损失评估的扩展实验 | [eval_policy_loss_act.py](../dropout/eval_policy_loss_act.py) |
| `libero/` | LIBERO 评估服务与客户端 | [eval_suite_http.py](../libero/eval_suite_http.py) |
| `third_party/trak/` | 梯度投影和训练数据归因 | [README](../third_party/trak/README.md) |
| `docs/research/` | 原 `2_docs/` 研究资料和方法笔记 | [研究笔记](research/) |
| `docs/archive/` | 整理前的说明、命令、安装记录和临时样例 | [原始 README](archive/CUPID_original.md) |

## 主 ACT 流程

1. [mymimicgen/3_convert.py](../mymimicgen/3_convert.py) 读取 HDF5 并构建 LeRobot 数据。该转换器使用下一帧绝对关节位置和夹爪指令作为监督动作。
2. `lerobot.scripts.train` 训练 ACT 基线，配置对象和模型实现在 `lerobot_2.1/src/lerobot/policies/act/`。
3. [server_act.py](../mimicgen/eval/server_act.py) 加载 ACT 策略并提供 `/act` HTTP 推理；[client_robomimic_act.py](../mimicgen/eval/client_robomimic_act.py) 在仿真中执行并保存轨迹。
4. [act/myscripts/cupid_act_influence.py](../act/myscripts/cupid_act_influence.py) 计算训练梯度投影、rollout 回报梯度、逆 Hessian-vector product 和影响分数。基础 ACT 数据和策略加载工具位于 [train_trak_act.py](../act/train_trak_act.py)。
5. [filter_joint_episode_window.py](../act/myscripts/filter_joint_episode_window.py) 联合影响、轨迹质量和覆盖，先筛选 episode，再筛选窗口。
6. [train_act_cupid_reweight_joint.py](../act/train_act_cupid_reweight_joint.py) 读取选择结果，执行 episode 筛选、窗口掩码和样本重加权，再训练策略。

[scripts/reproduce_act.py](../scripts/reproduce_act.py) 负责配置、命令与产物目录，算法计算仍由上述实现负责。主入口使用 `act/myscripts/` 的窗口版本。`act/cupid_act_influence.py`、`*_last_layers.py` 和 `*_strict_last_layers.py` 保留用于旧实验及参数层选择对照；不同版本的缓存不可混用。

## 任务扩展索引

| 路线 | 位置 | 使用前需要准备 |
| --- | --- | --- |
| RoboMimic / MimicGen ACT | [act/robomimic](../act/robomimic/)、[act/mymimicgen](../act/mymimicgen/) | 对应数据、同任务 checkpoint 和 rollout |
| LIBERO ACT | [act/libero/README.md](../act/libero/README.md) | LIBERO 资源与任务 BDDL 文件 |
| Franka 实机 ACT | [act/franka/README.md](../act/franka/README.md) | 相机、机器人通信、控制配置 |
| RoboTwin ACT | [act/robotwin](../act/robotwin/) | RoboTwin 仿真和对应演示数据 |
| π₀.₅ Franka | [openpi/franka](../openpi/franka/) | OpenPI 权重、训练配置与机器人接口 |
| OpenVLA 对照方法 | [openvla-oft/workflow](../openvla-oft/workflow/) | 预训练权重、LoRA 权重和评估环境 |

上述目录包含来源机器上的实验配置，不是每个任务都可通过一条通用命令直接复现。

## 维护约定

- 公共工具放在 `prose/`；跨模块导入使用 `prose.*`、`scripts.*`、`act.*` 的完整模块路径。
- 新实验的任务、路径、种子、训练预算优先放入 `configs/reproduction/`，输出放入 `outputs/<run>/`。
- 项目级分析工具放入 `scripts/analysis/`；框架专用实现放入对应框架目录，避免复制到根目录。
- 相对数据路径以仓库根目录为基准；框架资源以框架根目录为基准。可用 `LEROBOT_REPO` 覆盖默认本地 LeRobot。
- 数据集、权重、日志、缓存不提交；运行后保存配置、版本和必要摘要。大型框架内部已有测试和风格约定继续按各自规定维护。
- 文档主入口是根 README。环境调整更新 `docs/environments.md`；路径或命令变化更新 `docs/migration.md` 和 `docs/migration.json`。
- 文件名避免仅通过大小写区分。OpenPI 原 `README.md` 和 `readme.md` 已分别恢复为框架 README 和 [PROJECT_NOTES.md](../openpi/PROJECT_NOTES.md)。
