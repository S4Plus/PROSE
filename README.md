# PROSE

面向机器人模仿学习的数据评估、影响分析、数据筛选与策略再训练代码集合。代码基于 `Cupid_new` 仓库的 `cupid` 分支整理，包含 CUPID/ACT 实验、轨迹与时间窗口联合筛选、CUPID-QUALITY，以及 OpenPI π₀.₅、OpenVLA-OFT、MimicGen、LIBERO 等扩展。

开源仓库：[S4Plus/PROSE](https://github.com/S4Plus/PROSE)。本 README 描述**当前版本实际存在的代码**。原始 CUPID 说明保存在 [历史 README](docs/archive/CUPID_original.md)，代码来源与发布记录见 [发布说明](docs/publication.md)。可用 `git rev-parse HEAD` 记录当前复现版本。

## 阅读入口

| 文档 | 内容 |
| --- | --- |
| [代码结构与模块职责](docs/architecture.md) | 目录边界、关键入口、数据流、维护约定 |
| [环境配置](docs/environments.md) | ACT、仿真、OpenPI、OpenVLA 分环境安装与检查 |
| [代码复现](docs/reproduction.md) | 数据准备、训练、闭环评估、打分、筛选、再训练与对照实验 |
| [迁移说明](docs/migration.md) | 旧路径到新路径、兼容入口、已知限制与验证范围 |
| [脚本索引](scripts/README.md) | 通用工具、分析工具、历史脚本的位置 |
| [ACT 模块](act/README.md) | ACT 方法实现、不同打分版本与任务扩展 |

## 目录结构

```text
PROSE/
├── README.md                         # 整体使用说明
├── setup.py                          # prose-tools：本仓库工具的安装入口
├── hf_datasets_compat.py              # 保留的旧导入兼容入口
├── prose/                            # 公共路径与数据集兼容工具
├── configs/reproduction/             # 可修改的复现实验配置
├── environments/                     # 环境依赖说明文件
├── docs/                             # 整体文档、研究笔记、历史归档
├── scripts/
│   ├── reproduce_act.py              # ACT 分阶段预览 / 检查 / 执行
│   ├── check_repository.py           # 结构、文档链接、环境包检查
│   ├── analysis/                     # 失败聚类、影响分析、热力图
│   ├── lerobot/                      # LeRobot 打分、损失评估、随机子集训练
│   ├── data/                         # 数据检查工具
│   ├── train/                        # ACT 子集再训练工具
│   ├── experiments/                  # 历史实验批处理命令
│   ├── ros/                          # ROS 2 / Isaac Sim 实验桥接
│   ├── maintenance/                  # 仓库维护脚本
│   └── legacy_diffusion/             # 原版 Diffusion Policy 入口及启动脚本
├── act/                              # ACT 影响函数、质量评估和筛选训练
├── dropout/                          # ACT dropout 评估实验
├── libero/                           # LIBERO HTTP 服务和评估客户端
├── mymimicgen/                       # 自定义 HDF5 → LeRobot 数据转换
├── lerobot_2.1/                      # 本地修改的 LeRobot 框架
├── mimicgen/                         # 数据生成、仿真评估与其第三方代码
├── openpi/                           # OpenPI 框架及 π₀.₅ 实验
├── openvla-oft/                      # OpenVLA-OFT 框架及实验
├── third_party/trak/                 # 本地 TRAK 实现
└── tests/                            # 原有框架测试与新增仓库入口检查
```

`datasets/`、`outputs/`、`logs/` 等运行时目录由用户准备或程序生成，不随代码提供。已有框架保持原目录位置，以保留其内部导入、资源和配置路径。

## 运行条件与复现范围

- 主实验使用 Linux、NVIDIA GPU 和 Bash。当前 Windows 工作区可以编辑代码、预览命令和运行结构检查；训练与仿真请使用 Linux 主机或已配置 GPU 的 WSL2。
- ACT / 本地 LeRobot 使用 Python ≥ 3.10；OpenPI 使用 Python ≥ 3.11。各框架的 Torch、JAX、Transformers 要求不同，按照 [环境配置](docs/environments.md) 分开安装。
- `lerobot_2.1` 表示仓库目录名称；其 `pyproject.toml` 声明的包版本是 **LeRobot 0.3.4**。数据格式应根据数据集的 `meta/info.json` 判断。
- 仓库不包含训练数据、实验权重、原服务器 Conda 环境或可直接复核的完整指标表。
- 原版 Diffusion Policy 路线缺少 `diffusion_policy/`、对应训练配置和 `notebooks/data_curation.ipynb`。`environments/legacy_diffusion.yaml` 仅作为历史记录；它的 Torch 1.12.1 也与当前本地 TRAK 的 Torch ≥ 2.0 要求冲突。
- 部分历史实验命令仍保留 `/workspace/liushuo/...`、`/root/...` 等来源机器路径。新增入口和三个主要联合筛选启动脚本已支持当前仓库位置；复用其他历史脚本前需检查参数。

## 快速开始：ACT

以下 Bash 命令从仓库根目录执行。已克隆的工作区无需再次克隆。

```bash
# 新机器上的克隆方式
git clone https://github.com/S4Plus/PROSE.git
cd PROSE

# 创建 ACT 环境
conda create -n prose-act python=3.10 -y
conda activate prose-act
conda install -c conda-forge ffmpeg -y

# 安装当前分支的框架、TRAK 和额外分析依赖
python -m pip install -r environments/act-requirements.txt \
  -e ./lerobot_2.1 -e ./third_party/trak -e .
python -m pip check

# 只检查代码结构，无需数据或 GPU
python scripts/check_repository.py
# 同时列出 ACT 所需的已安装包
python scripts/check_repository.py --profile act
```

上述依赖范围来自本地声明，尚未在新 Linux/GPU 环境中完整安装验证。详细约束、仿真安装与排错见 [环境配置](docs/environments.md)。

### 配置与启动

复制 [ACT 示例配置](configs/reproduction/act.example.json)，按实际数据和模型路径修改：

```bash
cp configs/reproduction/act.example.json configs/reproduction/act.local.json

# 默认仅打印命令，不运行训练、不创建实验输出
python scripts/reproduce_act.py --config configs/reproduction/act.local.json --stage train

# 检查训练输入是否存在
python scripts/reproduce_act.py --config configs/reproduction/act.local.json --stage train --check

# 实际运行基线训练
python scripts/reproduce_act.py --config configs/reproduction/act.local.json --stage train --execute
```

配置中的数据和输出相对路径均以仓库根目录为基准；`--config` 参数本身以当前工作目录为基准。路径也可使用绝对路径、`~` 和环境变量。`dataset_repo_id` 应与本地数据的元信息一致。

完整流程：

```text
HDF5 数据 → LeRobot 数据 → ACT 基线训练
                             ↓
                HTTP 策略服务 + 仿真客户端
                             ↓
                     保存 rollout PKL
                             ↓
              score → select → retrain → 同协议评估
```

评估需要另外配置仿真环境和两个进程，见 [复现指南](docs/reproduction.md)。保存评估轨迹后，依次运行：

```bash
python scripts/reproduce_act.py --config configs/reproduction/act.local.json --stage score --execute
python scripts/reproduce_act.py --config configs/reproduction/act.local.json --stage select --execute
python scripts/reproduce_act.py --config configs/reproduction/act.local.json --stage retrain --execute
```

入口逐阶段检查数据、权重和上游产物，实际运行时保存命令、配置、代码提交号、包版本、日志和退出状态。`--check` 只检查文件条件，不加载模型或验证 CUDA。基线训练和再训练输出目录已存在时会停止，需使用新的实验名。

## 主要输入与产物

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| 数据转换 | 含图像、关节状态、动作的 HDF5 | `datasets/lerobot/<task>/{meta,data,videos}` |
| 基线训练 | LeRobot 数据、ACT 配置 | `outputs/<run>/baseline/checkpoints/.../pretrained_model` |
| 闭环评估 | 策略服务、相同任务与控制器 | rollout `.pkl`、成功率摘要，可选视频 |
| 影响打分 | `config.json` + `model.safetensors`、训练数据、rollout | `outputs/<run>/influence/score.pkl` 和分阶段缓存 |
| 联合筛选 | 影响分数、训练数据 | `joint_episode_window_selection.npz`、保留/删除列表、CSV 与摘要 |
| 再训练 | 原数据、分数、筛选 NPZ | `outputs/<run>/retrain/` 下的策略和训练状态 |

轨迹筛选、窗口掩码和重加权有不同含义；保留窗口比例不能直接等同于删掉轨迹比例。对照实验至少记录成功率、保留轨迹数、保留窗口/样本数、随机种子和训练预算。

## 其他路线

- **CUPID-QUALITY**：从 state-action pair 分数计算轨迹质量；输入格式及工具见 [模块说明](act/cupid_quality/README.md)。
- **OpenPI π₀.₅**：归一化、LoRA 训练、影响打分、窗口筛选与再训练，见 [复现指南](docs/reproduction.md#openpi-π₀₅)。
- **OpenVLA-OFT**：LeRobot 数据上的 LoRA 微调、服务评估与联合筛选，见 [复现指南](docs/reproduction.md#openvla-oft)。
- **LIBERO / Franka / RoboTwin**：任务入口和专用环境见 [结构索引](docs/architecture.md)。

## 验证与维护

```bash
python scripts/check_repository.py
python -m unittest discover -s tests/repository -v
```

结构检查覆盖整理后的 Python 语法、迁移目标和主要文档链接；入口测试覆盖路径解析、命令构造、输入缺失和执行记录。它们不代表训练收敛、仿真成功率或论文数值已复现。整理详情见 [迁移说明](docs/migration.md)，本次实际检查见 [验证记录](docs/validation.md)。

项目层面的许可证见 [LICENSE](LICENSE)。LeRobot、OpenPI、OpenVLA、MimicGen 及其他第三方目录保留各自许可证；发布或使用相应模块时需按其许可证处理。原版 CUPID 的引用信息见 [历史 README](docs/archive/CUPID_original.md)。
