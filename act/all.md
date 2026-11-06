# Can-MH 再训练命令总览（Random / Quality-Only / Influence-Only / Joint）

## 0) 固定路径

```bash
# dataset
DATASET_ROOT=/workspace/liushuo/cupid/datasets/lerobot/can_mh
DATASET_REPO_ID=ls/can_mh

# quality csv
QUALITY_CSV=/workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv

# score 来源（all + last 4/8/12）
SCORE_ALL=/workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl
SCORE_L4=/workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last4.pkl
SCORE_L8=/workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last8.pkl
SCORE_L12=/workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last12.pkl
```

筛选比例统一按 `q=0.1/0.2/0.3/0.4/0.5`（表示 drop quantile）。

## 1) Random（随机子集）

随机方法是“随机保留比例”，与 `q` 对应关系：`keep_ratio = 1 - q`，所以 keep 用 `0.9/0.8/0.7/0.6/0.5`。

```bash
mkdir -p /workspace/liushuo/cupid/3_model/logs/random_retrain

for keep in 0.9 0.8 0.7 0.6 0.5; do
  tag=$(echo "$keep" | tr -d '.')
  CUDA_VISIBLE_DEVICES=7 HF_HOME=/tmp/hf_home_random_${tag} HF_DATASETS_CACHE=/tmp/hf_datasets_cache_random_${tag} nohup \
  python /workspace/liushuo/cupid/random_train.py \
    --sample-ratio "$keep" \
    --sample-seed 42 \
    --dataset-repo-id ${DATASET_REPO_ID} \
    --dataset-root ${DATASET_ROOT} \
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --policy.repo_id ${DATASET_REPO_ID} \
    --output_dir /workspace/liushuo/cupid/3_model/can_mh_random_keep${tag} \
    --job_name can_mh_random_keep${tag} \
    --policy.device cuda \
    > /workspace/liushuo/cupid/3_model/logs/random_retrain/can_mh_random_keep${tag}_gpu7.log 2>&1 &
done
```

## 2) Quality-Only（只按质量分）

关键设置：
- `--joint-quality-source quality_csv`
- `--joint-influence-power 0.0`
- `--weight-mode none`
- 可不传 `--score-pkl`

```bash
mkdir -p /workspace/liushuo/cupid/3_model/logs/quality_only

for q in 0.1 0.2 0.3 0.4 0.5; do
  qtag=$(echo "$q" | tr -d '.')
  CUDA_VISIBLE_DEVICES=7 HF_HOME=/tmp/hf_home_quality_${qtag} HF_DATASETS_CACHE=/tmp/hf_datasets_cache_quality_${qtag} nohup \
  python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
    --dataset.root ${DATASET_ROOT} \
    --dataset.repo_id ${DATASET_REPO_ID} \
    --output-dir /workspace/liushuo/cupid/3_model/can_mh_quality_only_q${qtag} \
    --job_name can_mh_quality_only_q${qtag} \
    --joint-quality-source quality_csv \
    --quality-score-csv ${QUALITY_CSV} \
    --filter-score-mode joint \
    --joint-drop-quantile "$q" \
    --joint-quality-power 1.0 \
    --joint-influence-power 0.0 \
    --drop-bad-ratio 1.1 \
    --weight-mode none \
    --num-workers 0 \
    --log-every 20 \
    --max-steps 100000 \
    > /workspace/liushuo/cupid/3_model/logs/quality_only/can_mh_quality_only_q${qtag}_gpu7.log 2>&1 &
done
```

## 3) Influence-Only（只按 influence score）

关键设置：
- 使用 `train_act_cupid_reweight.py`
- `--drop-weight-quantile q`
- `--weight-mode influence`

### 3.1 启动命令（覆盖 all + last4/8/12）

```bash
mkdir -p /workspace/liushuo/cupid/3_model/logs/influence_only

SCORES=(
  /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl
  /workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last4.pkl
  /workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last8.pkl
  /workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last12.pkl
)
RATIOS=(0.1 0.2 0.3 0.4 0.5)
GPUS=(7 6 5 4)

idx=0
for score_pkl in "${SCORES[@]}"; do
  tag=$(basename "${score_pkl}" .pkl)   # score / score_last4 / score_last8 / score_last12
  for q in "${RATIOS[@]}"; do
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    qtag=$(echo "$q" | tr -d '.')
    run_name="can_mh_influence_only_${tag}_q${qtag}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    HF_HOME="/tmp/hf_home_${run_name}" \
    HF_DATASETS_CACHE="/tmp/hf_datasets_cache_${run_name}" nohup \
    python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
      --dataset.root ${DATASET_ROOT} \
      --dataset.repo_id ${DATASET_REPO_ID} \
      --score-pkl "${score_pkl}" \
      --output-dir /workspace/liushuo/cupid/3_model/${run_name} \
      --job_name "${run_name}" \
      --policy.type act \
      --policy.repo_id ${DATASET_REPO_ID} \
      --drop-bad-ratio 1.1 \
      --drop-weight-quantile "$q" \
      --weight-mode influence \
      --use-weighted-sampler \
      --num-workers 0 \
      --log-every 20 \
      --max-steps 100000 \
      > /workspace/liushuo/cupid/3_model/logs/influence_only/${run_name}_gpu${gpu}.log 2>&1 &
    idx=$((idx + 1))
  done
done
```

## 4) Joint（质量 + 影响力联合分）

关键设置：
- 使用 `train_act_cupid_reweight_joint.py`
- `--joint-quality-source quality_csv`
- `--joint-quality-power 1.0 --joint-influence-power 1.0`
- `--filter-score-mode joint --joint-drop-quantile q`

### 4.1 启动命令（覆盖 all + last4/8/12）

```bash
mkdir -p /workspace/liushuo/cupid/3_model/logs/joint_by_score_ratio

SCORES=(
  /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl
  /workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last4.pkl
  /workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last8.pkl
  /workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last12.pkl
)
RATIOS=(0.1 0.2 0.3 0.4 0.5)
GPUS=(7 6 5 4)

idx=0
for score_pkl in "${SCORES[@]}"; do
  tag=$(basename "${score_pkl}" .pkl)
  for q in "${RATIOS[@]}"; do
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    qtag=$(echo "$q" | tr -d '.')
    run_name="can_mh_jointsr_${tag}_dq${qtag}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    HF_HOME="/tmp/hf_home_${run_name}" \
    HF_DATASETS_CACHE="/tmp/hf_datasets_cache_${run_name}" nohup \
    python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
      --dataset.root ${DATASET_ROOT} \
      --dataset.repo_id ${DATASET_REPO_ID} \
      --score-pkl "${score_pkl}" \
      --output-dir /workspace/liushuo/cupid/3_model/joint_by_score_ratio/${run_name} \
      --joint-quality-source quality_csv \
      --quality-score-csv ${QUALITY_CSV} \
      --filter-score-mode joint \
      --joint-drop-quantile "$q" \
      --joint-quality-power 1.0 \
      --joint-influence-power 1.0 \
      --weight-mode influence \
      --use-weighted-sampler \
      --num-workers 0 \
      --log-every 20 \
      --max-steps 100000 \
      > /workspace/liushuo/cupid/3_model/logs/joint_by_score_ratio/${run_name}_gpu${gpu}.log 2>&1 &
    idx=$((idx + 1))
  done
done
```

## 5) 常用查看/停止命令

```bash
# 查看训练任务
pgrep -af train_act_cupid_reweight.py
pgrep -af train_act_cupid_reweight_joint.py

# 查看 worker 日志（串行调度）
tail -f /workspace/liushuo/cupid/3_model/logs/serial_worker_gpu7.log
tail -f /workspace/liushuo/cupid/3_model/logs/serial_worker_gpu6.log
tail -f /workspace/liushuo/cupid/3_model/logs/serial_worker_gpu5.log
tail -f /workspace/liushuo/cupid/3_model/logs/serial_worker_gpu4.log

# 停止 influence-only / joint 训练
pkill -f train_act_cupid_reweight.py
pkill -f train_act_cupid_reweight_joint.py

# 停止调度脚本
pkill -f train_with_lastlayer_scores_serial_per_gpu.sh
pkill -f train_joint_with_scores_and_ratios_serial_per_gpu.sh
```
