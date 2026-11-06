# `eval_cupid_quality.py` 使用说明（LeRobot Only）
act用绝对关节角度训练
openpi05要用相对的
脚本路径：`/workspace/liushuo/cupid/act/eval_cupid_quality.py`

## 1. 目标与约束

* 目标：对轨迹做质量评估并输出可用于 drop/keep 的结果。
* 数据来源：**只依赖 LeRobot 数据集**（`data/chunk-*/episode_*.parquet`）。
* 不依赖项：不读取原始 `hdf5`、不依赖先验质量 `csv`。

## 2. 支持的质量源

1. `episode_eval`
   * 侧重运动学平滑性：action/state jitter、长度一致性、可选能量项。

2. `trajectory_eval`
   * 侧重任务轨迹结构：terminal consistency(kNN)、efficiency、progress、monotonic/backtrack/stall、smoothness、length consistency。
   * terminal kNN 现支持更稳健配置：
     * 距离度量：`--traj-eval-terminal-distance-metric {l2,l1,cosine}`
     * 终态归一化：`--traj-eval-terminal-normalize {none,zscore,robust_zscore}`
     * 邻居聚合：`--traj-eval-terminal-agg {mean,median,trimmed_mean}`
     * 截尾比例：`--traj-eval-terminal-trim-ratio`

3. `rank_fusion`
   * 先计算 `episode_eval` 与 `trajectory_eval`；
   * 各自转为百分位 rank（0~1）；
   * 按权重融合（`--rank-fusion-episode-weight`、`--rank-fusion-trajectory-weight`）。

## 3. 输出文件

输出目录下会生成：

* `quality_report.json`
* `quality_consistency_report.json`
* `quality_episodes.csv`
* `quality_drop.txt`
* `quality_keep.txt`
* `quality_top_50.txt`
* `quality_bottom_50.txt`

## 4. Can 数据集（LeRobot）推荐命令

当前 can 数据集路径：
`/workspace/liushuo/cupid/datasets/lerobot/can_mh`

推荐先用 `rank_fusion`：

```bash
python /workspace/liushuo/cupid/act/eval_cupid_quality.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --quality-source rank_fusion \
  --rank-fusion-episode-weight 0.3 \
  --rank-fusion-trajectory-weight 0.7 \
  --drop-quantile 0.2 \
  --output-dir /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion
```

## 5. 关键参数

* `--drop-quantile`：按分位数丢弃低分样本（如 `0.2` 丢弃最低 20%）。
* `--drop-threshold`：按绝对分数阈值丢弃（可与 quantile 叠加，取并集）。
* `--consistency-bootstrap-rounds`：无监督稳定性评估采样轮数（默认 200）。
* `--consistency-episode-block-size`：分段偏置评估窗口（默认 50）。
* `--traj-eval-terminal-normalize`：建议默认 `robust_zscore`，减少状态量纲差异和离群点影响。
* `--traj-eval-terminal-agg`：建议 `trimmed_mean`，避免少量异常邻居主导终态一致性分。

## 6. 当前实现评估思路（无监督）

脚本内部会额外输出一致性分析：

* 组件相关性（Spearman）
* Bootstrap drop 集稳定性（Jaccard）
* Episode 分段 drop 偏置
* 终态模式覆盖率（terminal mode coverage）
* 分数堆积检测（score pileup）

用于判断：当前质量分数是否稳定、是否被单一指标主导、是否有明显批次/收集者偏置。

## 7. 与 Influence 联合打分再训练（Joint Retrain）

思路：

1. 先用本脚本产出 `quality_episodes.csv`；
2. 在训练脚本中设置 `--joint-quality-source quality_csv`；
3. 将 `quality_score` 与 influence 归一化分数做联合分：
   `joint_score = quality_score^a * influence_score^b`；
4. 按 `--joint-drop-quantile` 丢弃低分 episode，再训练 ACT。

示例（can_mh）：

```bash
HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_joint_quality_csv \
  --joint-quality-source quality_csv \
  --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
  --filter-score-mode joint \
  --joint-drop-quantile 0.2 \
  --joint-quality-power 1.0 \
  --joint-influence-power 1.0 \
  --weight-mode influence \
  --use-weighted-sampler \
  --max-steps 100000
```

## 8. 多比例批量训练（nohup + 指定 GPU）

目标：联合分过滤比例 `q=0.1/0.2/0.3/0.4/0.5`。  
GPU 分配：`7/6/5/4/7`。

```bash
mkdir -p /workspace/liushuo/cupid/3_model/logs

# q=0.1 -> GPU 7
CUDA_VISIBLE_DEVICES=7 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_joint_quality_csv_q01 \
  --joint-quality-source quality_csv \
  --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
  --filter-score-mode joint --joint-drop-quantile 0.1 \
  --joint-quality-power 1.0 --joint-influence-power 1.0 \
  --weight-mode influence --use-weighted-sampler --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_joint_q01_gpu7.log 2>&1 &

# q=0.2 -> GPU 6
CUDA_VISIBLE_DEVICES=6 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_joint_quality_csv_q02 \
  --joint-quality-source quality_csv \
  --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
  --filter-score-mode joint --joint-drop-quantile 0.2 \
  --joint-quality-power 1.0 --joint-influence-power 1.0 \
  --weight-mode influence --use-weighted-sampler --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_joint_q02_gpu6.log 2>&1 &

# q=0.3 -> GPU 5
CUDA_VISIBLE_DEVICES=5 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_joint_quality_csv_q03 \
  --joint-quality-source quality_csv \
  --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
  --filter-score-mode joint --joint-drop-quantile 0.3 \
  --joint-quality-power 1.0 --joint-influence-power 1.0 \
  --weight-mode influence --use-weighted-sampler --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_joint_q03_gpu5.log 2>&1 &

# q=0.4 -> GPU 4
CUDA_VISIBLE_DEVICES=4 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_joint_quality_csv_q04 \
  --joint-quality-source quality_csv \
  --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
  --filter-score-mode joint --joint-drop-quantile 0.4 \
  --joint-quality-power 1.0 --joint-influence-power 1.0 \
  --weight-mode influence --use-weighted-sampler --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_joint_q04_gpu4.log 2>&1 &

# q=0.5 -> GPU 7
CUDA_VISIBLE_DEVICES=7 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_joint_quality_csv_q05 \
  --joint-quality-source quality_csv \
  --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
  --filter-score-mode joint --joint-drop-quantile 0.5 \
  --joint-quality-power 1.0 --joint-influence-power 1.0 \
  --weight-mode influence --use-weighted-sampler --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_joint_q05_gpu7.log 2>&1 &
```

查看任务：

```bash
ps -ef | rg train_act_cupid_reweight_joint.py
```

## 10. 仅按 Influence Score 筛选训练

目标：不使用质量分，只按 influence score 做筛选。  
固定 score 路径：`/workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl`

```bash
mkdir -p /workspace/liushuo/cupid/3_model/logs

# q=0.1 -> GPU 7
CUDA_VISIBLE_DEVICES=7 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_influence_only_q01 \
  --drop-bad-ratio 1.1 \
  --drop-weight-quantile 0.1 \
  --weight-mode influence \
  --use-weighted-sampler \
  --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_influence_only_q01_gpu7.log 2>&1 &

# q=0.2 -> GPU 6
CUDA_VISIBLE_DEVICES=6 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_influence_only_q02 \
  --drop-bad-ratio 1.1 \
  --drop-weight-quantile 0.2 \
  --weight-mode influence \
  --use-weighted-sampler \
  --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_influence_only_q02_gpu6.log 2>&1 &

# q=0.3 -> GPU 5
CUDA_VISIBLE_DEVICES=5 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_influence_only_q03 \
  --drop-bad-ratio 1.1 \
  --drop-weight-quantile 0.3 \
  --weight-mode influence \
  --use-weighted-sampler \
  --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_influence_only_q03_gpu5.log 2>&1 &

# q=0.4 -> GPU 4
CUDA_VISIBLE_DEVICES=4 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_influence_only_q04 \
  --drop-bad-ratio 1.1 \
  --drop-weight-quantile 0.4 \
  --weight-mode influence \
  --use-weighted-sampler \
  --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_influence_only_q04_gpu4.log 2>&1 &

# q=0.5 -> GPU 7
CUDA_VISIBLE_DEVICES=7 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_influence_only_q05 \
  --drop-bad-ratio 1.1 \
  --drop-weight-quantile 0.5 \
  --weight-mode influence \
  --use-weighted-sampler \
  --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_influence_only_q05_gpu7.log 2>&1 &
```

查看任务：

```bash
ps -ef | rg train_act_cupid_reweight.py
```

## 9. 仅按质量筛选训练（不传 `--score-pkl`）

目标：保留比例 `0.9/0.8/0.7/0.6/0.5`；GPU 分配 `7/5/6/4/7`。  
实现方式：`joint_influence_power=0` 且 `weight_mode=none`，仅使用 `quality_episodes.csv` 进行筛选。

```bash
mkdir -p /workspace/liushuo/cupid/3_model/logs

# keep=0.9 -> drop=0.1 -> GPU 7
CUDA_VISIBLE_DEVICES=7 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_quality_only_keep09 \
  --job_name can_mh_quality_only_keep09 \
  --joint-quality-source quality_csv \
  --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
  --filter-score-mode joint --joint-drop-quantile 0.1 \
  --joint-quality-power 1.0 --joint-influence-power 0.0 \
  --drop-bad-ratio 1.1 \
  --weight-mode none \
  --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_quality_only_keep09_gpu7.log 2>&1 &

# keep=0.8 -> drop=0.2 -> GPU 5
CUDA_VISIBLE_DEVICES=5 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_quality_only_keep08 \
  --job_name can_mh_quality_only_keep08 \
  --joint-quality-source quality_csv \
  --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
  --filter-score-mode joint --joint-drop-quantile 0.2 \
  --joint-quality-power 1.0 --joint-influence-power 0.0 \
  --drop-bad-ratio 1.1 \
  --weight-mode none \
  --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_quality_only_keep08_gpu5.log 2>&1 &

# keep=0.7 -> drop=0.3 -> GPU 6
CUDA_VISIBLE_DEVICES=6 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_quality_only_keep07 \
  --job_name can_mh_quality_only_keep07 \
  --joint-quality-source quality_csv \
  --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
  --filter-score-mode joint --joint-drop-quantile 0.3 \
  --joint-quality-power 1.0 --joint-influence-power 0.0 \
  --drop-bad-ratio 1.1 \
  --weight-mode none \
  --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_quality_only_keep07_gpu6.log 2>&1 &

# keep=0.6 -> drop=0.4 -> GPU 4
CUDA_VISIBLE_DEVICES=4 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_quality_only_keep06 \
  --job_name can_mh_quality_only_keep06 \
  --joint-quality-source quality_csv \
  --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
  --filter-score-mode joint --joint-drop-quantile 0.4 \
  --joint-quality-power 1.0 --joint-influence-power 0.0 \
  --drop-bad-ratio 1.1 \
  --weight-mode none \
  --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_quality_only_keep06_gpu4.log 2>&1 &

# keep=0.5 -> drop=0.5 -> GPU 7
CUDA_VISIBLE_DEVICES=7 HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets_cache nohup \
python /workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --dataset.repo_id ls/can_mh \
  --output-dir /workspace/liushuo/cupid/3_model/can_mh_quality_only_keep05 \
  --job_name can_mh_quality_only_keep05 \
  --joint-quality-source quality_csv \
  --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
  --filter-score-mode joint --joint-drop-quantile 0.5 \
  --joint-quality-power 1.0 --joint-influence-power 0.0 \
  --drop-bad-ratio 1.1 \
  --weight-mode none \
  --max-steps 100000 \
  > /workspace/liushuo/cupid/3_model/logs/can_mh_quality_only_keep05_gpu7.log 2>&1 &
```

查看任务：

```bash
ps -ef | rg train_act_cupid_reweight_joint.py
```
