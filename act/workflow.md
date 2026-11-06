# square_mh 自适应筛选与训练 Workflow

本文档使用当前主流程：

- `filter_joint_episode_window.py`
- `run_filter_and_train_joint_episode_window.sh`
- `budget_mode=auto_marginal`

## 1. lerobot数据集训练act

mkdir -p ./logs ./.cache/huggingface/home ./.cache/huggingface/datasets

HF_HOME=./.cache/huggingface/home \
HF_DATASETS_CACHE=./.cache/huggingface/datasets \
CUDA_VISIBLE_DEVICES=6 nohup lerobot-train \
    --dataset.repo_id ls/transport_mh \
    --dataset.root=/workspace/liushuo/cupid/datasets/lerobot/transport_mh \
    --output_dir /workspace/liushuo/cupid/3_model/transport_mh_full_act \
    --job_name transport_mh \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.repo_id ls/transport_mh \
    > ./logs/transport_mh_full_act.log 2>&1 &

## 2. eval性能测试
服务器端：
  cd ~/cupid/mimicgen/eval
  conda activate cupid
  python server.py --checkpoint /workspace/liushuo/cupid/3_model/square_mh/checkpoints/100000/pretrained_model

客户端：
  cd ~/cupid/mimicgen/eval
  conda activate robomimic_venv
  python client_robomimic_act.py 

## 3. 影响分数计算：window级别的

 CUDA_VISIBLE_DEVICES=4 nohup python /workspace/liushuo/cupid/act/myscripts/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/3_model/transport_mh_full_act/checkpoints/100000/pretrained_model \
    --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/transport_mh \
    --eval_rollout_dir /workspace/liushuo/robomimic/robomimic/eval/tmp/transport_mh_act_58.0/episodes \
    --output_path /workspace/liushuo/cupid/1_infulenece_score/transport_mh/score_episode_window.pkl \
    --stage_cache_prefix /workspace/liushuo/cupid/1_infulenece_score/transport_mh/score \
    --aggregate_mode episode \
    --window_size 50 \
    --window_stride 50\
    > logs/cupid_act_influence_transport_mh.log 2>&1 &



 CUDA_VISIBLE_DEVICES=0 nohup python /workspace/liushuo/cupid/act/myscripts/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/3_model/square_mh/checkpoints/100000/pretrained_model \
    --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/square_mh \
    --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/square_mh/base_46_37.0/pkl \
    --output_path /workspace/liushuo/cupid/1_infulenece_score/square_mh/score_episode_window.pkl \
    --stage_cache_prefix /workspace/liushuo/cupid/1_infulenece_score/square_mh/score \
    --aggregate_mode episode \
    --window_size 50 \
    --window_stride 50\
    > logs/cupid_act_influence_square_mh.log 2>&1 &


 CUDA_VISIBLE_DEVICES=0  python /workspace/liushuo/cupid/act/myscripts/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/3_model/can_mh/checkpoints/100000/pretrained_model \
    --train_dataset_path /workspace/liushuo/cupid/datasets/lerobot/can_mh \
    --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/tmp/can_mh_act_63_73/episodes \
    --output_path /workspace/liushuo/cupid/1_infulenece_score/can_mh/score_episode_window_50_25.pkl \
    --stage_cache_prefix /workspace/liushuo/cupid/1_infulenece_score/can_mh/score \
    --aggregate_mode episode \
    --window_size 50 \
    --window_stride 25
## 4. 数据筛选再训练
CUDA_VISIBLE_DEVICES=4 DATASET_ROOT=/workspace/liushuo/cupid/datasets/lerobot/square_mh \
  DATASET_REPO_ID=ls/square_mh \
  SCORE_PKL=/workspace/liushuo/cupid/1_infulenece_score/square_mh/score_episode_window.pkl \
  QUALITY_SOURCE=trajectory_eval \
  OUTPUT_BASE=/workspace/liushuo/cupid/3_model/joint_episode_window \
  RUN_NAME=square_mh_joint_episode_window \
  JOB_NAME=square_mh_joint_episode_window \
  EPISODE_DROP_RATIO=0.10 \
  WINDOW_SIZE=50 \
  WINDOW_STRIDE=50 \
  WINDOW_LOCAL_EPISODE_SCORE_WEIGHT=0.4 \
  WINDOW_LOCAL_QUALITY_WEIGHT=0.4 \
  WINDOW_LOCAL_EPISODE_JOINT_WEIGHT=0.2 \
  MIN_COVERAGE_RATIO=0.5 \
  MARGINAL_GAIN_THRESHOLD=0.15 \
  RELATIVE_GAIN_THRESHOLD=0.5 \
  nohup bash /workspace/liushuo/cupid/act/myscripts/run_filter_and_train_joint_episode_window.sh \
  > logs/square_mh_joint_episode_window.log 2>&1 &


### 4.1 筛选
## 1. 输入

必需：

- `score_pkl`
  - `/workspace/liushuo/cupid/1_infulenece_score/square_mh/score_episode_window.pkl`
- `dataset_root`
  - `/workspace/liushuo/cupid/datasets/lerobot/square_mh`

可选：

- `quality_episodes.csv`
  - 已有离线质量分时使用 `quality_csv`
  - 没有就用 `trajectory_eval` 在线计算


## 2. 自适应算法说明

这里的“自适应”不是 MLP 预测比例，而是 `auto_marginal`：

1. 先按 episode 联合分去掉最低一部分 episode
2. 对每个保留 episode 构建滑窗
3. 对每个 window 计算局部贡献分
4. 用 `contribution + coverage novelty` 的边际收益贪心选窗
5. 满足最小覆盖率后，当边际收益下降到阈值以下时自动停止

推荐使用的筛选超参数是这组：

- `episode_drop_ratio = 0.10`
- `window_size = 50`
- `window_stride = 50`
- `window_local_episode_score_weight = 0.4`
- `window_local_quality_weight = 0.4`
- `window_local_episode_joint_weight = 0.2`
- `min_coverage_ratio = 0.5`
- `marginal_gain_threshold = 0.15`
- `relative_gain_threshold = 0.5`


## 3. 一键执行：筛选 + 训练

### 3.1 有 `quality_episodes.csv`

```bash
DATASET_ROOT=/workspace/liushuo/cupid/datasets/lerobot/square_mh \
DATASET_REPO_ID=ls/square_mh \
SCORE_PKL=/workspace/liushuo/cupid/1_infulenece_score/square_mh/score_episode_window.pkl \
QUALITY_SOURCE=quality_csv \
QUALITY_CSV=/path/to/square_mh/quality_episodes.csv \
OUTPUT_BASE=/workspace/liushuo/cupid/3_model/joint_episode_window \
RUN_NAME=square_mh_joint_episode_window \
JOB_NAME=square_mh_joint_episode_window \
DEVICE=cuda \
BATCH_SIZE=32 \
NUM_WORKERS=4 \
MAX_STEPS=100000 \
SAVE_EVERY=20000 \
LOG_EVERY=200 \
SEED=1000 \
LEARNING_RATE=1e-5 \
BACKBONE_LR=1e-5 \
WEIGHT_DECAY=1e-4 \
DROPOUT=0.1 \
VISION_BACKBONE=resnet18 \
PRETRAINED_BACKBONE=ResNet18_Weights.IMAGENET1K_V1 \
EPISODE_DROP_RATIO=0.10 \
WINDOW_SIZE=50 \
WINDOW_STRIDE=50 \
WINDOW_LOCAL_EPISODE_SCORE_WEIGHT=0.4 \
WINDOW_LOCAL_QUALITY_WEIGHT=0.4 \
WINDOW_LOCAL_EPISODE_JOINT_WEIGHT=0.2 \
MIN_COVERAGE_RATIO=0.5 \
MARGINAL_GAIN_THRESHOLD=0.15 \
RELATIVE_GAIN_THRESHOLD=0.5 \
bash /workspace/liushuo/cupid/act/myscripts/run_filter_and_train_joint_episode_window.sh
```

### 3.2 没有 `quality_episodes.csv`

```bash
DATASET_ROOT=/workspace/liushuo/cupid/datasets/lerobot/square_mh \
DATASET_REPO_ID=ls/square_mh \
SCORE_PKL=/workspace/liushuo/cupid/1_infulenece_score/square_mh/score_episode_window.pkl \
QUALITY_SOURCE=trajectory_eval \
OUTPUT_BASE=/workspace/liushuo/cupid/3_model/joint_episode_window \
RUN_NAME=square_mh_joint_episode_window \
JOB_NAME=square_mh_joint_episode_window \
DEVICE=cuda \
BATCH_SIZE=32 \
NUM_WORKERS=4 \
MAX_STEPS=100000 \
SAVE_EVERY=20000 \
LOG_EVERY=200 \
SEED=1000 \
LEARNING_RATE=1e-5 \
BACKBONE_LR=1e-5 \
WEIGHT_DECAY=1e-4 \
DROPOUT=0.1 \
VISION_BACKBONE=resnet18 \
PRETRAINED_BACKBONE=ResNet18_Weights.IMAGENET1K_V1 \
EPISODE_DROP_RATIO=0.10 \
WINDOW_SIZE=50 \
WINDOW_STRIDE=50 \
WINDOW_LOCAL_EPISODE_SCORE_WEIGHT=0.4 \
WINDOW_LOCAL_QUALITY_WEIGHT=0.4 \
WINDOW_LOCAL_EPISODE_JOINT_WEIGHT=0.2 \
MIN_COVERAGE_RATIO=0.5 \
MARGINAL_GAIN_THRESHOLD=0.15 \
RELATIVE_GAIN_THRESHOLD=0.5 \
bash /workspace/liushuo/cupid/act/myscripts/run_filter_and_train_joint_episode_window.sh
```


## 4. 只做筛选，不启动训练

先看筛选结果时，推荐：

```bash
DATASET_ROOT=/workspace/liushuo/cupid/datasets/lerobot/square_mh \
DATASET_REPO_ID=ls/square_mh \
SCORE_PKL=/workspace/liushuo/cupid/1_infulenece_score/square_mh/score_episode_window.pkl \
QUALITY_SOURCE=trajectory_eval \
OUTPUT_BASE=/workspace/liushuo/cupid/3_model/joint_episode_window \
RUN_NAME=square_mh_joint_episode_window_debug \
JOB_NAME=square_mh_joint_episode_window_debug \
EPISODE_DROP_RATIO=0.10 \
WINDOW_SIZE=50 \
WINDOW_STRIDE=50 \
WINDOW_LOCAL_EPISODE_SCORE_WEIGHT=0.4 \
WINDOW_LOCAL_QUALITY_WEIGHT=0.4 \
WINDOW_LOCAL_EPISODE_JOINT_WEIGHT=0.2 \
MIN_COVERAGE_RATIO=0.5 \
MARGINAL_GAIN_THRESHOLD=0.15 \
RELATIVE_GAIN_THRESHOLD=0.5 \
SKIP_TRAIN=1 \
bash /workspace/liushuo/cupid/act/myscripts/run_filter_and_train_joint_episode_window.sh
```

关键输出：

- `selection/selection_summary.json`
- `selection/episode_scores.csv`
- `selection/window_scores.csv`
- `selection/joint_episode_window_selection.npz`
- `selection/kept_episodes.txt`
- `selection/dropped_episodes.txt`


## 5. 分步执行

### 5.1 只做筛选

```bash
python /workspace/liushuo/cupid/act/myscripts/filter_joint_episode_window.py \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/square_mh/score_episode_window.pkl \
  --dataset-root /workspace/liushuo/cupid/datasets/lerobot/square_mh \
  --output-dir /workspace/liushuo/cupid/3_model/square_mh_joint_episode_window_selection \
  --quality-source trajectory_eval \
  --episode-drop-ratio 0.10 \
  --window-size 50 \
  --window-stride 50 \
  --window-local-episode-score-weight 0.4 \
  --window-local-quality-weight 0.4 \
  --window-local-episode-joint-weight 0.2 \
  --budget-mode auto_marginal \
  --min-windows-per-episode 1 \
  --min-coverage-ratio 0.5 \
  --marginal-gain-threshold 0.15 \
  --relative-gain-threshold 0.5
```

如果使用质量分 CSV，则加上：

```bash
--quality-source quality_csv \
--quality-score-csv /path/to/square_mh/quality_episodes.csv
```

### 5.2 使用筛选结果训练 ACT

```bash
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/square_mh \
  --dataset.repo_id ls/square_mh \
  --policy.repo_id ls/square_mh \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/square_mh/score_episode_window.pkl \
  --selection-npz /workspace/liushuo/cupid/3_model/square_mh_joint_episode_window_selection/joint_episode_window_selection.npz \
  --selection-override-episodes \
  --output_dir /workspace/liushuo/cupid/3_model/square_mh_joint_episode_window_train \
  --job_name square_mh_joint_episode_window_train \
  --policy.type act \
  --device cuda \
  --batch-size 32 \
  --num-workers 4 \
  --max-steps 100000 \
  --save-every 20000 \
  --log-every 200 \
  --seed 1000 \
  --filter-score-mode joint \
  --joint-drop-quantile 0 \
  --drop-weight-quantile 0 \
  --drop-bad-ratio 1.1 \
  --joint-quality-source trajectory_eval \
  --weight-mode influence \
  --use-weighted-sampler \
  --segment-mask \
  --segment-weight \
  --learning-rate 1e-5 \
  --backbone-lr 1e-5 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --vision-backbone resnet18 \
  --pretrained-backbone ResNet18_Weights.IMAGENET1K_V1
```

如果使用质量分 CSV，则把：

- `--joint-quality-source trajectory_eval`

替换为：

- `--joint-quality-source quality_csv`
- `--quality-score-csv /path/to/square_mh/quality_episodes.csv`


## 6. 怎么检查筛选是否正常

优先看：

1. `selection_summary.json`
2. `episode_scores.csv`
3. `window_scores.csv`
4. `joint_episode_window_selection.npz`

建议检查：

1. `kept_episodes` 是否符合预期
2. `kept_window_ratio` 是否过低或过高
3. 是否只保留了少量相似窗口
4. 长轨迹和短轨迹是否都保留了合理数量的 window


## 7. 最短可执行命令

如果你现在只想先把 `square_mh` 的自适应筛选跑出来，最短命令是：

```bash
DATASET_ROOT=/workspace/liushuo/cupid/datasets/lerobot/square_mh \
DATASET_REPO_ID=ls/square_mh \
SCORE_PKL=/workspace/liushuo/cupid/1_infulenece_score/square_mh/score_episode_window.pkl \
QUALITY_SOURCE=trajectory_eval \
OUTPUT_BASE=/workspace/liushuo/cupid/3_model/joint_episode_window \
RUN_NAME=square_mh_joint_episode_window_debug \
JOB_NAME=square_mh_joint_episode_window_debug \
EPISODE_DROP_RATIO=0.10 \
WINDOW_SIZE=50 \
WINDOW_STRIDE=50 \
WINDOW_LOCAL_EPISODE_SCORE_WEIGHT=0.4 \
WINDOW_LOCAL_QUALITY_WEIGHT=0.4 \
WINDOW_LOCAL_EPISODE_JOINT_WEIGHT=0.2 \
MIN_COVERAGE_RATIO=0.5 \
MARGINAL_GAIN_THRESHOLD=0.15 \
RELATIVE_GAIN_THRESHOLD=0.5 \
SKIP_TRAIN=1 \
bash /workspace/liushuo/cupid/act/myscripts/run_filter_and_train_joint_episode_window.sh
```
