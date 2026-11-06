# 代码复现

本指南将已有代码连接成可配置的复现流程。完整实验需要用户准备数据、权重、GPU 与仿真环境；本次整理只执行结构、入口和合成数据检查，未运行完整模型训练。所有相对命令从 `PROSE` 根目录执行，除非明确写出 `cd`。

## ACT：数据准备

先按照 [环境配置](environments.md) 创建 `prose-act` 和 `prose-sim`。以下以 RoboMimic Can MH 为示例；`can`、`square`、`transport` 等任务需分别匹配训练数据、仿真环境和模型。

### 获取 HDF5

本地 robomimic 提供下载脚本，可先预览，再下载：

```bash
conda activate prose-sim
python mimicgen/third_party/robomimic/robomimic/scripts/download_datasets.py \
  --tasks can --dataset_types mh --hdf5_types image \
  --download_dir datasets/robomimic --dry_run

# 确认预览中的任务与保存位置后，去掉 --dry_run 下载。
```

脚本根据其登记的 URL 下载数据；网络可用性、数据许可和实际下载文件需自行核实。也可以使用已有 MimicGen HDF5，数据生成入口在 [mimicgen/mimicgen/scripts](../mimicgen/mimicgen/scripts/) 中。

转换器 [3_convert.py](../mymimicgen/3_convert.py) 要求每条 `data/demo_*` 下包含：

```text
obs/agentview_image             (T, H, W, 3)
obs/robot0_eye_in_hand_image    (T, H, W, 3)
obs/robot0_joint_pos            (T, 7)
obs/robot0_gripper_qpos         (T, 2)
actions                        (T, A)，最后一维为夹爪指令
```

图像应为已解码的数组；如果源数据使用压缩图像、缺少关节位置或仅包含低维观测，需要先提取/解码观测，不能直接套用该转换器。

### 转换为 LeRobot

替换示例 HDF5 路径为实际文件：

```bash
conda activate prose-act
python mymimicgen/3_convert.py \
  --source datasets/robomimic/can/mh/image.hdf5 \
  --output datasets/lerobot/can_mh \
  --repo-id local/can_mh \
  --fps 20
```

该转换器生成 9 维状态（7 个关节角 + 2 个夹爪 qpos），以及 8 维动作（下一帧 7 个绝对关节角 + 下一帧夹爪指令）。它改变了原始动作的表示，所以后续评估使用 `JOINT_POSITION` 控制器。若你的任务采用末端位姿或不同机械臂维度，需使用对应转换器和控制器。

输出目录通常包含 `meta/`、`data/` 和 `videos/`。先检查 `meta/info.json` 的 features、FPS、episode 数，并实际读取至少一个 episode，确认图像、动作和索引对应。转换目标使用新的空目录。

## ACT：配置与基线训练

```bash
cp configs/reproduction/act.example.json configs/reproduction/act.local.json
```

编辑 `act.local.json`：

| 参数 | 含义 |
| --- | --- |
| `run_name` | 实验名；不同种子/对照组使用不同名称 |
| `dataset_root` / `dataset_repo_id` | 本地数据目录与元信息中的 repo id |
| `pretrained_dir` | 基线训练后实际生成的 ACT 模型目录 |
| `rollout_dir` | 直接包含评估 `.pkl` 文件的目录 |
| `output_root` | 统一实验输出根目录 |
| `seed` | 基线、影响计算与再训练的随机种子 |
| `device` | 例如 `cuda`、`cuda:0`；CUDA 可用性由框架验证 |
| `steps` / `batch_size` / `num_workers` | 训练预算、batch 和数据加载并行数 |
| `window_size` / `window_stride` | 打分和筛选共享的窗口长度与步长 |
| `episode_drop_ratio` | 第一阶段删掉的 episode 比例，不是窗口丢弃率 |

示例使用 100000 步，所以 `pretrained_dir` 写为 `.../checkpoints/100000/pretrained_model`。修改训练步数后同步修改该路径；实际目录结构以框架输出为准。

```bash
conda activate prose-act
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_act.py \
  --config configs/reproduction/act.local.json --stage train --check
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_act.py \
  --config configs/reproduction/act.local.json --stage train --execute
```

省略 `--execute` 和 `--check` 时仅预览命令，适合检查路径和预算，不要求数据已下载。

## ACT：闭环评估与 rollout

### 终端 A：策略服务

在 `prose-act` 环境中加载实际基线权重。下面的 `--checkpoint` 与配置中的 `pretrained_dir` 保持一致：

```bash
conda activate prose-act
python mimicgen/eval/server_act.py \
  --checkpoint outputs/can_mh/baseline/checkpoints/100000/pretrained_model \
  --host 127.0.0.1 --port 8000 --device cuda
```

### 终端 B：仿真客户端

在独立 `prose-sim` 环境中运行相同任务，记录视频、轨迹和成功率摘要：

```bash
conda activate prose-sim
export MUJOCO_GL=egl
python mimicgen/eval/client_robomimic_act.py \
  --server_url http://127.0.0.1:8000/act \
  --env_name can --robots Panda --controller JOINT_POSITION \
  --n_rollouts 20 --seed 42 \
  --camera_names agentview,robot0_eye_in_hand \
  --camera_height 84 --camera_width 84 \
  --save_rollouts_dir outputs/can_mh/rollouts/episodes \
  --save_images_in_pkl \
  --video_dir outputs/can_mh/rollouts/videos \
  --summary_path outputs/can_mh/rollouts/summary.json
```

`--save_images_in_pkl` 用于保留影响计算所需图像。数据分辨率不同则同步调整服务与客户端。这里的 20 条 rollout 仅为运行示例，正式评估数量应按实验协议设置。

MimicGen 自定义任务改用 [client_mimicgen_act.py](../mimicgen/eval/client_mimicgen_act.py)，它的参数与 RoboMimic 客户端不同，例如 `--num_episodes` 和 `--use_camera_obs`；参见 [客户端说明](../mimicgen/eval/ACT_CLIENT_USAGE.md)。

rollout 必须包含时间对齐的观测、动作和实际成功/回报信息。影响函数会使用它们计算回报梯度，仅有视频或 summary 无法替代轨迹。评估和训练需采用相同相机映射、状态/动作维度与归一化。

## ACT：影响打分、联合筛选、再训练

```bash
conda activate prose-act

# 训练数据 + 基线权重 + rollout → 分数与缓存
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_act.py \
  --config configs/reproduction/act.local.json --stage score --execute

# episode 与 window 联合筛选
python scripts/reproduce_act.py \
  --config configs/reproduction/act.local.json --stage select --execute

# 读取选择结果，重新训练 ACT
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_act.py \
  --config configs/reproduction/act.local.json --stage retrain --execute
```

打分产物位于 `outputs/<run>/influence/`。筛选输出位于 `outputs/<run>/selection/`：

- `kept_episodes.txt`、`dropped_episodes.txt`：轨迹保留/删除列表。
- `episode_scores.csv`、`window_scores.csv`：用于人工核验排序、质量和窗口选择。
- `joint_episode_window_selection.npz`：再训练使用的轨迹与样本选择结果。
- `selection_summary.json`：保留比例、窗口覆盖和参数摘要。

新入口采用 `trajectory_eval` 质量源和 `auto_marginal` 窗口预算，权重及覆盖阈值沿用当前筛选实现默认值。更多参数见 [方法说明](../act/myscripts/joint_episode_window_method.md) 和各脚本的 `--help`。

再训练从头初始化 ACT，使用影响分数重加权、窗口掩码和窗口权重；未传入 `--pretrained-dir`。如果需要从基线继续微调，应直接调用训练脚本并明确传入该参数，同时作为独立实验记录。再训练保存的模型位于 `outputs/<run>/retrain/`，而基线使用 LeRobot 的 checkpoint 层级，评估时不要混淆。

同一配置下重新计算 score/select 可能更新分数、选择结果或复用缓存。更换 checkpoint、数据、窗口参数或评分方法时，使用新的 `run_name` 或 `output_root`，避免混用旧缓存。

也可使用已修复仓库路径的历史联合启动器，但应查看其默认值：

```bash
DATASET_ROOT="$PWD/datasets/lerobot/can_mh" \
PRETRAINED_DIR="$PWD/outputs/can_mh/baseline/checkpoints/100000/pretrained_model" \
EVAL_ROLLOUT_DIR="$PWD/outputs/can_mh/rollouts/episodes" \
SKIP_TRAIN=1 bash act/myscripts/run_filter_and_train_joint_episode_window.sh
```

这里的 `SKIP_TRAIN=1` 会执行打分/筛选并跳过训练，仍会创建实验产物。

## 对照实验与结果复核

至少比较全量训练、同预算随机子集和方法筛选三组。随机轨迹子集入口是 [scripts/lerobot/random_train.py](../scripts/lerobot/random_train.py)，随机 state-action 比例对照见 [act/comparison_random_state_action_ratio](../act/comparison_random_state_action_ratio/README.md)。

使用相同的任务、初始状态分布、测试 seeds、控制器、相机、训练步数和 batch size。每组至少记录：

| 记录 | 用途 |
| --- | --- |
| Git 提交和本地修改 | 确定实际运行的代码 |
| 数据来源、repo id、features、episode/帧数 | 确定输入数据与动作表示 |
| seed、配置、训练预算、评分版本 | 保证组间可比 |
| 保留轨迹、窗口、有效样本数 | 区分不同筛选粒度 |
| 每个 seed 的成功率、汇总均值与波动 | 比较性能与稳定性 |
| checkpoint、评估 summary、日志 | 支撑结果复核 |

新入口在 `outputs/<run>/logs/<stage>_<UTC时间>/` 保存 `manifest.json` 和 `stdout.log`。长时间执行时可从另一终端查看日志。退出码和记录代表该命令的完成状态；还需核验策略文件与评估指标。

## CUPID-QUALITY

已获得 pair-score NPZ/PKL/CSV 时，可以单独计算质量：

```bash
conda activate prose-act
python act/cupid_quality/compute_cupid_quality.py \
  --input-path outputs/pair_scores.npz \
  --output-dir outputs/cupid_quality \
  --normalize-by-trajectory-length --normalizer quantile
```

输入需包含 `pair_scores`、`eval_episode_ids`、`train_episode_ids`；详见 [格式说明](../act/cupid_quality/README.md)。常规 `score.pkl` 中的聚合分数和 policy-level 梯度缓存不能直接当作精确的 state-action pair 分数，需使用模块中的重建工具。

## OpenPI π₀.₅

先配置独立 `uv` 环境。修改 [训练配置](../openpi/src/openpi/training/config.py) 中相应任务的 `repo_id`/数据路径、prompt、动作变换、预训练权重和输出目录。当前 `pi05_can_mh_lora` 等配置仍包含来源机器的数据绝对路径。

```bash
cd openpi
# 为实际数据计算归一化，再训练基线
uv run scripts/compute_norm_stats.py --config-name pi05_can_mh_lora
WANDB_MODE=disabled uv run scripts/train.py pi05_can_mh_lora --exp-name prose_can_mh

# checkpoint 目录应直接包含 params，替换为真实路径
uv run python scripts/influence_pi05_lora.py \
  --config-name pi05_can_mh_lora \
  --checkpoint-dir /absolute/path/to/checkpoint_step \
  --eval-rollouts-dir /absolute/path/to/openpi_rollouts \
  --output-path "$PWD/tmp/prose_can_mh/influence.pkl"

# 联合筛选 + 子集归一化，先跳过训练
DATASET_DIR="$(cd .. && pwd)/datasets/lerobot/can_mh" \
SCORE_PKL="$PWD/tmp/prose_can_mh/influence.pkl" \
TRAIN_CONFIG=pi05_can_mh_lora RUN_NAME=prose_can_mh SKIP_TRAIN=1 \
OVERWRITE=0 bash myscripts/run_filter_and_train_joint_episode_window.sh
cd ..
```

执行完整筛选再训练时，将 `SKIP_TRAIN` 改为 `0`，使用新的实验名和正确的权重配置。`SKIP_TRAIN=1` 仍会生成样本索引并重算子集归一化。OpenPI rollout 由 [client_robomimic_openpi05.py](../mimicgen/eval/client_robomimic_openpi05.py) 等客户端采集；服务配置见框架和任务 README。

这一框架还包含 GRASS、不同层参数、episode-only、window-only 等实验变体；缓存及 score 文件需与对应选择脚本匹配，不能直接用 ACT 分数替代所有 OpenPI 分数。

## OpenVLA-OFT

先按 [环境配置](environments.md) 安装独立环境，并准备 OpenVLA 预训练权重。使用 [finetune_openvla_lerobot.py](../openvla-oft/vla-scripts/finetune_openvla_lerobot.py) 或任务启动脚本训练；任务启动脚本仍包含历史路径，需核对数据、模型、run 目录和 GPU。

已有本模型的影响分数和 checkpoint 后，可使用路径已修复的联合启动器：

```bash
conda activate prose-openvla
DATASET_ROOT="$PWD/datasets/lerobot/can_mh" \
SCORE_PKL="/absolute/path/to/openvla_influence.pkl" \
CHECKPOINT_DIR="/absolute/path/to/openvla_checkpoint" \
RUN_NAME=prose_can_mh PREPARE_ONLY=1 \
bash openvla-oft/vla-scripts/run_filter_and_train_joint_episode_window.sh
```

`PREPARE_ONLY=1` 执行筛选与训练准备并跳过训练；先查看生成的索引、摘要与后续训练命令。完整运行时改为 `PREPARE_ONLY=0`，并使用新的实验名。服务入口是 [serve_policy_openvla.py](../openvla-oft/vla-scripts/serve_policy_openvla.py)，仿真客户端是 [client_robomimic_openvla.py](../mimicgen/eval/client_robomimic_openvla.py)。原有端到端流程见 [run_full_can_mh_openvla_pipeline.sh](../openvla-oft/vla-scripts/run_full_can_mh_openvla_pipeline.sh)，其中 Conda 环境与路径需要按实际主机调整。

## 历史 Diffusion Policy 路线

旧训练/评估入口集中在 [scripts/legacy_diffusion](../scripts/legacy_diffusion/)，原说明见 [历史 README](archive/CUPID_original.md)。当前分支缺少核心包、configs 和 curation notebook，且历史环境与当前 TRAK 存在版本冲突。

恢复这一条路线需要匹配版本的完整 CUPID/Diffusion Policy 源码、配置、TRAK 和相应数据。当前整理不虚构缺失组件，也不把旧脚本标为已复现。
