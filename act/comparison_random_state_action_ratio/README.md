## Random ACT Baseline by State-Action Ratio

This folder contains an isolated random-baseline launcher that keeps a target
fraction of state-action pairs by randomly masking frames.

Unlike `/workspace/liushuo/cupid/random_train.py`, which samples by episode
count ratio, this script:

- reads episode lengths from `meta/episodes.jsonl`
- samples exact frame indices to keep
- writes the frame mask to a JSON selection file
- patches `LeRobotDataset` at runtime so ACT trains only on those kept frames

Example:

```bash
cd /workspace/liushuo/cupid/act

CUDA_VISIBLE_DEVICES=7 python comparison_random_state_action_ratio/random_train_by_state_action_ratio.py \
  --sample-ratio 0.8 \
  --sample-seed 0 \
  --dataset-repo-id ls/square_mh \
  --dataset-root /workspace/liushuo/cupid/datasets/lerobot/square_mh \
  --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
  --selection-output /workspace/liushuo/cupid/outputs/square_mh_random_frame_mask_0p8_seed0.json \
  -- \
  --output_dir /workspace/liushuo/cupid/outputs/square_mh_random_sa_0p8_seed0 \
  --job_name square_mh_random_sa_0p8_seed0 \
  --policy.type act \
  --policy.repo_id ls/square_mh
```

Sweep `0.9 -> 0.1` ratios serially on one GPU:

```bash
cd /workspace/liushuo/cupid/act

GPU=7 \
DATASET_ROOT=/workspace/liushuo/cupid/datasets/lerobot/square_mh \
DATASET_REPO_ID=ls/square_mh \
OUTPUT_ROOT=/workspace/liushuo/cupid/outputs/square_mh_random_sa_sweep \
LOG_DIR=/workspace/liushuo/cupid/logs/square_mh_random_sa_sweep \
bash comparison_random_state_action_ratio/run_random_state_action_ratio_sweep.sh
```
