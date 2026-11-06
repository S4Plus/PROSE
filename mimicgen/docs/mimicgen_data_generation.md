MimicGen 数据生成完整流程（整理版）

本文整理了 MimicGen 的数据生成流程，包含“快速运行示例”和“典型工作流概述”。示例命令以仓库根目录为起点。

**快速数据生成运行**

**步骤 1：准备源人类数据集**
1. 进入仓库根目录。

```bash
cd /path/to/mimicgen
```

1. 下载 Square 任务的源演示数据集。

```bash
python mimicgen/scripts/download_datasets.py --dataset_type source --tasks square
```

1. 为源数据添加 MimicGen 所需的 `datagen_info`。

```bash
python mimicgen/scripts/prepare_src_dataset.py \
  --dataset datasets/source/square.hdf5 \
  --env_interface MG_Square \
  --env_interface_type robosuite
```

说明：`--env_interface` 与 `--env_interface_type` 用于选择正确的环境接口类，以提取 `DatagenInfo`（对象位姿、末端位姿、子任务终止信号等）。

本步骤生成（或更新）的源数据 HDF5 结构如下：

```text
<dataset>.hdf5
└─ data
   ├─ attrs
   │  ├─ total              # 总步数（所有 episode 累加）
   │  └─ env_args           # 环境元信息（json 字符串）
   ├─ demo_0
   │  ├─ actions            # [T, ...]
   │  ├─ states             # [T, ...] 或 states/<key>（若 state 为 dict）
   │  ├─ obs/<key>          # 观测（压缩存储）
   │  ├─ datagen_info
   │  │  ├─ eef_pose
   │  │  ├─ target_pose
   │  │  ├─ gripper_action
   │  │  ├─ object_poses/<obj_name>
   │  │  └─ subtask_term_signals/<signal_name>
   │  ├─ src_demo_inds      # 生成数据才会有（源数据通常无）
   │  ├─ src_demo_labels    # 生成数据才会有（源数据通常无）
   │  └─ attrs
   │     ├─ model_file       # 仅 robosuite 环境（可选）
   │     └─ num_samples      # 该 episode 的步数
   ├─ demo_1
   └─ ...
```

**步骤 2：准备数据生成配置**
1. MimicGen 的配置模板位于 `mimicgen/exps/templates`。模板可由脚本生成。

```bash
python mimicgen/scripts/generate_config_templates.py
```

1. 若要生成与论文一致的核心实验配置，使用 `mimicgen/scripts/generate_core_configs.py`。
1. 快速运行时请先在 `mimicgen/scripts/generate_core_configs.py` 中修改参数：
1. `NUM_TRAJ = 10`
1. `GUARANTEE = False`
1. 默认值为 `NUM_TRAJ = 400` 且 `GUARANTEE = True`，会持续生成直到拿到 400 条成功轨迹。
1. 执行生成脚本：

```bash
python mimicgen/scripts/generate_core_configs.py
```

脚本会生成配置 JSON 并打印路径，默认输出到 `./tmp/core_configs`。

**步骤 3：运行数据生成并查看输出**
1. 使用生成的配置启动数据生成：

```bash
python mimicgen/scripts/generate_dataset.py \
  --config ./tmp/core_configs/demo_src_square_task_D1.json \
  --auto-remove-exp
```

1. 如果末尾报错提示找不到 ffmpeg，可安装：

```bash
conda install ffmpeg
```

1. 默认输出目录来自配置，`generate_core_configs.py` 默认设置为 `./tmp/core_datasets`。示例输出路径为：
1. `./tmp/core_datasets/square/demo_src_square_task_D1`

该目录常见内容如下：
- `demo.hdf5`：成功轨迹数据集
- `demo_failed.hdf5`：失败轨迹数据集（有上限）
- `important_stats.json`：成功率等统计信息
- `log.txt`：终端输出日志
- `logs/`：长任务的阶段性统计
- `mg_config.json`：本次实验配置
- `playback_demo_src_square_task_D1.mp4`：成功轨迹回放视频（有上限）
- `playback_demo_src_square_task_D1_failed.mp4`：失败轨迹回放视频（有上限）

说明：生成的 `demo.hdf5` 与 robomimic 完全兼容，可直接用于训练或可视化。

**典型数据生成工作流概述**

**步骤 1：收集源演示**
1. 通过遥操作或其他方式收集人类演示。
1. 确保输出为 robomimic 兼容的 HDF5 数据集（必要时使用后处理脚本转换）。
1. 确保环境元数据指向可用于回放的仿真环境包装器。

**步骤 2：为源演示添加注释信息**
1. 使用 `mimicgen/scripts/prepare_src_dataset.py` 将 `datagen_info` 写入源数据集。
1. 每个任务通常需要一个环境接口类，用于：
1. 提供对象位姿字典与末端位姿；
1.（可选）提供子任务终止信号；
1. 定义动作与目标位姿之间的转换逻辑。
1. 若不依赖自动终止信号，可使用 `mimicgen/scripts/annotate_subtasks.py` 手动标注子任务边界。

**步骤 3：运行数据生成**
1. 准备配置文件（可参考模板或批量生成脚本）。
1. 执行 `mimicgen/scripts/generate_dataset.py` 启动生成。
1. 可在不同重置分布、不同机械臂或不同对象实例上运行生成（见论文与“复现实验”教程）。

**步骤 4：用生成数据训练策略**
1. 使用任意策略学习算法训练（常用行为克隆）。
1. 因为输出与 robomimic 兼容，可直接使用 robomimic 进行训练与评测。

---

**生成的 HDF5 文件结构**

以下结构来自 `mimicgen/utils/file_utils.py` 中的 `write_demo_to_hdf5` 与 `merge_all_hdf5` 写入逻辑。

```text
<dataset>.hdf5
└─ data
   ├─ attrs
   │  ├─ total              # 总步数（所有 episode 累加）
   │  └─ env_args           # 环境元信息（json 字符串）
   ├─ demo_0
   │  ├─ actions            # [T, ...]
   │  ├─ states             # [T, ...] 或 states/<key>（若 state 为 dict）
   │  ├─ obs/<key>          # 观测（压缩存储）
   │  ├─ datagen_info
   │  │  ├─ eef_pose
   │  │  ├─ target_pose
   │  │  ├─ gripper_action
   │  │  ├─ object_poses/<obj_name>
   │  │  └─ subtask_term_signals/<signal_name>
   │  ├─ src_demo_inds      # 可选，源演示索引（每子任务）
   │  ├─ src_demo_labels    # 可选，按时间步对齐的源演示标签
   │  └─ attrs
   │     ├─ model_file       # 仅 robosuite 环境（可选）
   │     └─ num_samples      # 该 episode 的步数
   ├─ demo_1
   └─ ...
```

说明：
- `demo_i` 为每条演示轨迹分组，合并后从 `demo_0` 依次编号。
- `obs/<key>` 是否存在取决于配置中 `collect_obs` 设置。
- 该结构不包含 `rewards` / `dones` / `next_obs`，当前写入逻辑未输出这些字段。
