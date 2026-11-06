# 环境配置

## 环境划分与已知约束

以下版本来自当前代码的依赖声明，未通过本次整理验证完整的 GPU 安装与训练。环境文件提供可执行的安装起点；创建成功后应保存自己的解析结果。

| 路线 | Python | 主要依赖依据 | 说明 |
| --- | --- | --- | --- |
| ACT / LeRobot | 示例用 3.10；声明 ≥ 3.10 | [LeRobot pyproject](../lerobot_2.1/pyproject.toml)、[TRAK setup](../third_party/trak/setup.py) | Torch ≥ 2.2.1 且 < 2.8，Torchvision ≥ 0.21 且 < 0.23；交由包解析器匹配 |
| 仿真 / MimicGen | 示例用 3.10 | [robosuite setup](../mimicgen/third_party/robosuite/setup.py)、[MimicGen setup](../mimicgen/setup.py) | 本地 robosuite 1.4.1、robomimic 0.3.0；使用原生 `mujoco` |
| OpenPI | ≥ 3.11 | [pyproject](../openpi/pyproject.toml)、[uv.lock](../openpi/uv.lock) | JAX CUDA12 0.5.3、Flax 0.10.2、Torch 2.7.1、Transformers 4.53.2 |
| OpenVLA-OFT | 示例用 3.10 | [pyproject](../openvla-oft/pyproject.toml)、[SETUP](../openvla-oft/SETUP.md) | Torch 2.2.0、Torchvision 0.17.0、PEFT 0.11.1、专用 Transformers fork |
| 历史 Diffusion Policy | 历史文件为 3.10 | [legacy_diffusion.yaml](../environments/legacy_diffusion.yaml) | Torch 1.12.1 与当前 TRAK 要求冲突；核心代码与配置缺失 |

不要将以上依赖全部合并到同一个环境。尤其 OpenPI 的 `uv` 源声明使用一个固定提交的 LeRobot，ACT 使用本仓库 `lerobot_2.1`；两者不应互相替换。

## 主机准备

训练命令以 Linux Bash 为准。准备 Conda、可用的 NVIDIA 驱动以及足够的数据、视频、模型存储空间。视频编码需要 FFmpeg；编译 CUDA 扩展时还需要与所选 Torch 匹配的编译工具。

```bash
nvidia-smi
conda --version
```

无显示器的仿真可先使用 `MUJOCO_GL=egl`；不支持 EGL 的主机可尝试 OSMesa，并安装相应系统库。具体选择取决于主机驱动和显示环境。

Windows 本机适合结构检查与命令预览。当前源码包含 Bash、Linux 仿真和 CUDA 依赖，主实验安装步骤未验证原生 Windows。

## ACT / LeRobot 环境

在 `PROSE` 根目录执行：

```bash
conda create -n prose-act python=3.10 -y
conda activate prose-act
conda install -c conda-forge ffmpeg -y

python -m pip install -r environments/act-requirements.txt \
  -e ./lerobot_2.1 -e ./third_party/trak -e .

python -m pip check
python scripts/check_repository.py --profile act
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import lerobot; print(lerobot.__file__)"
```

[act-requirements.txt](../environments/act-requirements.txt) 补充分析和 HDF5 依赖，不覆盖 LeRobot 自身声明。这里的 NumPy `< 2` 是为了与已有机器人仿真及历史数据代码保持一致，仍需用真实样例核验。

LeRobot 使用视频数据；安装后检查 `ffmpeg -version`，并尝试读取自己的一个 episode。其声明按平台选择 `torchcodec`，不要把 `lerobot_2.1/torchcodec/` 的源代码目录当作已经安装的包。

TRAK 可选的快速投影扩展位于 `third_party/trak/fast_jl/`。只有实际使用 CUDA projector 且提示缺失 `fast_jl` 时，再在已配置编译环境中安装：

```bash
python -m pip install ./third_party/trak/fast_jl --no-build-isolation
```

新 ACT 入口会设置当前仓库和 LeRobot 的 Python 搜索路径。直接调用其他旧工具时，可以使用：

```bash
export PROSE_ROOT="$(pwd)"
export LEROBOT_REPO="$PROSE_ROOT/lerobot_2.1"
export PYTHONPATH="$PROSE_ROOT:$LEROBOT_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled
```

`PROSE_ROOT` 是文档中的路径变量；主代码从文件位置自动解析仓库。`LEROBOT_REPO` 是代码支持的框架覆盖变量。

## RoboMimic / MimicGen 仿真环境

单独的仿真进程通过 HTTP 调用 ACT 服务，因此不必将 LeRobot 安装进仿真环境：

```bash
conda create -n prose-sim python=3.10 -y
conda activate prose-sim
conda install -c conda-forge ffmpeg -y

python -m pip install -r environments/simulation-requirements.txt \
  -e ./mimicgen/third_party/robosuite \
  -e ./mimicgen/third_party/robomimic \
  -e ./mimicgen/third_party/robosuite-task-zoo \
  -e ./mimicgen

python -m pip check
python scripts/check_repository.py --profile simulation
python -c "import robosuite, robomimic, mimicgen; print(robosuite.__file__); print(robomimic.__file__); print(mimicgen.__file__)"
export MUJOCO_GL=egl
```

仓库同时包含 `mimicgen/third_party/` 和 `mimicgen/mimicgen/third_party/` 两套来源目录。以上统一使用第一套，避免把多个相同包装到同一环境。应通过输出的 `__file__` 核实实际导入位置。

旧 CUPID 安装笔记中的 `mujoco-py` 对应另一条历史路线，不用于这里的 robosuite 1.4.1 原生 MuJoCo 环境。

## OpenPI 环境

按照本地 [OpenPI README](../openpi/README.md) 使用其独立 `uv` 工作区。先准备 `uv`，再执行：

```bash
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --locked
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv run python -c "import jax; print(jax.devices())"
uv run python ../scripts/check_repository.py --profile openpi
cd ..
```

`--locked` 会在锁文件与声明不一致时停止。发生此类错误，应先核对 `openpi/pyproject.toml` 与 `uv.lock` 的差异，再决定更新锁文件；不要悄悄混入 ACT 环境中的依赖。

OpenPI 的 PyTorch 推理还有专用 Transformers 替换文件，见 [OpenPI README](../openpi/README.md) 的 PyTorch 部分。JAX 训练流程与 PyTorch 推理流程应分别按其要求配置。

## OpenVLA-OFT 环境

使用仓库自己的 fork 和依赖声明：

```bash
conda create -n prose-openvla python=3.10 -y
conda activate prose-openvla
python -m pip install -e ./openvla-oft
python -m pip install numpy\<2 h5py pyarrow scipy scikit-learn
python -m pip install packaging ninja
# 需要支持的 GPU 和 CUDA 编译环境
python -m pip install flash-attn==2.5.5 --no-build-isolation

python -m pip check
python scripts/check_repository.py --profile openvla
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`pyproject.toml` 指向专用 Transformers 和 `dlimp` Git 来源；这些来源未全部固定提交，所以本路线并非完整锁定环境。安装后保存包版本和 Git 来源。Flash Attention 的实际可用性需要在目标 GPU 上验证。

## 环境记录与排错

```bash
mkdir -p outputs/environment
python -m pip freeze > outputs/environment/pip-freeze.txt
conda env export --no-builds > outputs/environment/conda-environment.yaml
```

| 症状 | 优先核实 |
| --- | --- |
| `No module named lerobot` / `act` | 当前环境、editable 安装、`PYTHONPATH` 和导入文件位置 |
| `No module named diffusion_policy` | 当前分支没有该核心目录；旧脚本不能仅靠补装几个依赖运行 |
| Torch / Torchvision 算子加载失败 | 依赖解析是否匹配、CPU/CUDA wheel 是否一致 |
| 视频或 torchcodec 加载失败 | FFmpeg 动态库、Torch/torchcodec 兼容性、视频文件实际是否存在 |
| `mujoco` / OpenGL 初始化失败 | 仿真环境、驱动、`MUJOCO_GL` 和 EGL/OSMesa 系统库 |
| OpenPI 缺少 norm stats | 对当前训练配置和实际训练子集重新计算归一化 |
| 影响计算显存不足 | 先减少计算脚本的 batch、projection 维度和并发；记录改变后的参数 |
| 历史绝对路径不存在 | 优先使用新入口；历史脚本的路径、checkpoint、GPU 编号需按主机修改 |

主入口的包检查仅列出是否安装和版本；`pip check` 仅检查声明关系。两者都不能代替真实的数据加载、视频解码和 GPU 运行验证。
