# nohup python scripts/train/retrain_act_subset.py \
#     --retain-trak-pkl  /workspace/liushuo/cupid/eval_belt_pick_bottle/new_score/trajectory_scores.pkl\
#     --retain-metric mean_of_mean-succ-all \
#     --retain-split train \
#     --drop-ratio 0 \
#     --dataset-repo-id ls/Static_Pick_Bottle_400 \
#     --dataset-root /ext_workspace/liushuo/Static_Pick_Bottle_400 \
#     --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
#     --policy.type act \
#     --output_dir /workspace/liushuo/cupid/outputs/retrain_cupid/ratio2_1.0 \
#     --job_name Belt_Pick_Bottle_abs_403_drop50 \
#     --policy.device cuda \
#     --policy.repo_id ls/Static_Pick_Bottle_400 \
#     > ./logs/train2_400_act2_1.0.log 2>&1 &

nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/new_test/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0.2 \
    --dataset-repo-id ls/Static_Pick_Bottle_400 \
    --dataset-root /ext_workspace/liushuo/Static_Pick_Bottle_400 \
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/outputs_new/retrain_cupid/ratio2_0.8 \
    --job_name Belt_Pick_Bottle_abs_403_drop50 \
    --policy.device cuda \
    --policy.repo_id ls/Static_Pick_Bottle_400 \
    > ./logs/train2_400_act2_0.8.log 2>&1 &



nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/new_test/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0.5 \
    --dataset-repo-id ls/Static_Pick_Bottle_400 \
    --dataset-root /ext_workspace/liushuo/Static_Pick_Bottle_400 \
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/outputs_new/retrain_cupid/ratio2_0.5 \
    --job_name Belt_Pick_Bottle_abs_403_drop50 \
    --policy.device cuda \
    --policy.repo_id ls/Static_Pick_Bottle_400 \
    > ./logs/train2_400_act2_0.5.log 2>&1 &



nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/new_test/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0.8 \
    --dataset-repo-id ls/Static_Pick_Bottle_400 \
    --dataset-root /ext_workspace/liushuo/Static_Pick_Bottle_400 \
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/outputs_new/retrain_cupid/ratio2_0.2 \
    --job_name Belt_Pick_Bottle_abs_403_drop50 \
    --policy.device cuda \
    --policy.repo_id ls/Static_Pick_Bottle_400 \
    > ./logs/train2_400_act2_0.2.log 2>&1 &


CUDA_VISIBLE_DEVICES=7 nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/act/mimicgen/hammer_cleanup_D0/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0.2 \
    --dataset-repo-id ls/hammer_cleanup_d0_core \
    --dataset-root /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core \
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/act/mimicgen/hammer_cleanup_core/retrain_cupid/ratio_0.2 \
    --job_name hammer_cleanup_d0_core \
    --policy.device cuda \
    --policy.repo_id ls/hammer_cleanup_d0_core \
    > ./logs/train2_hammer_cleanup_d0_core_act2_0.2.log 2>&1 &





CUDA_VISIBLE_DEVICES=6 nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/act/mimicgen/hammer_cleanup_D0/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0.4 \
    --dataset-repo-id ls/hammer_cleanup_d0_core \
    --dataset-root /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core \
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/act/mimicgen/hammer_cleanup_core/retrain_cupid/ratio_0.4 \
    --job_name hammer_cleanup_d0_core \
    --policy.device cuda \
    --policy.repo_id ls/hammer_cleanup_d0_core \
    > ./logs/train2_hammer_cleanup_d0_core_act2_0.4.log 2>&1 &



CUDA_VISIBLE_DEVICES=5 nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/act/mimicgen/hammer_cleanup_D0/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0.6 \
    --dataset-repo-id ls/hammer_cleanup_d0_core \
    --dataset-root /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core \
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/act/mimicgen/hammer_cleanup_core/retrain_cupid/ratio_0.6 \
    --job_name hammer_cleanup_d0_core \
    --policy.device cuda \
    --policy.repo_id ls/hammer_cleanup_d0_core \
    > ./logs/train2_hammer_cleanup_d0_core_act2_0.6.log 2>&1 &


CUDA_VISIBLE_DEVICES=5 nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/act/mimicgen/hammer_cleanup_D0/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0.8 \
    --dataset-repo-id ls/hammer_cleanup_d0_core \
    --dataset-root /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core \
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/act/mimicgen/hammer_cleanup_core/retrain_cupid/ratio_0.8 \
    --job_name hammer_cleanup_d0_core \
    --policy.device cuda \
    --policy.repo_id ls/hammer_cleanup_d0_core \
    > ./logs/train2_hammer_cleanup_d0_core_act2_0.8.log 2>&1 &