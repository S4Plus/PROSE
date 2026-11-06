# nohup python scripts/train/retrain_act_subset.py \
#     --retain-trak-pkl  /workspace/liushuo/cupid/eval_belt_pick_bottle/new_score/score.pkl\
#     --retain-metric mean_of_mean-succ-all \
#     --retain-split train \
#     --drop-ratio 0 \
#     --dataset-repo-id ls/Static_Pick_Bottle_400 \
#     --dataset-root /ext_workspace/liushuo/Static_Pick_Bottle_400 \
#     --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
#     --policy.type act \
#     --output_dir /workspace/liushuo/cupid/outputs/retrain_cupid/ratio_1.0 \
#     --job_name Belt_Pick_Bottle_abs_403_drop50 \
#     --policy.device cuda \
#     --policy.repo_id ls/Static_Pick_Bottle_400 \
#     > ./logs/train2_400_act_1.0.log 2>&1 &

# nohup python scripts/train/retrain_act_subset.py \
#     --retain-trak-pkl  /workspace/liushuo/cupid/new_test/score.pkl\
#     --retain-metric mean_of_mean-succ-all \
#     --retain-split train \
#     --drop-ratio 0.2 \
#     --dataset-repo-id ls/Static_Pick_Bottle_400 \
#     --dataset-root /ext_workspace/liushuo/Static_Pick_Bottle_400 \
#     --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
#     --policy.type act \
#     --output_dir /workspace/liushuo/cupid/outputs_new/retrain_cupid/ratio_0.8 \
#     --job_name Belt_Pick_Bottle_abs_403_drop50 \
#     --policy.device cuda \
#     --policy.repo_id ls/Static_Pick_Bottle_400 \
#     > ./logs/train2_400_act_0.8.log 2>&1 &



# nohup python scripts/train/retrain_act_subset.py \
#     --retain-trak-pkl  /workspace/liushuo/cupid/new_test/score.pkl\
#     --retain-metric mean_of_mean-succ-all \
#     --retain-split train \
#     --drop-ratio 0.5 \
#     --dataset-repo-id ls/Static_Pick_Bottle_400 \
#     --dataset-root /ext_workspace/liushuo/Static_Pick_Bottle_400 \
#     --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
#     --policy.type act \
#     --output_dir /workspace/liushuo/cupid/outputs_new/retrain_cupid/ratio_0.5 \
#     --job_name Belt_Pick_Bottle_abs_403_drop50 \
#     --policy.device cuda \
#     --policy.repo_id ls/Static_Pick_Bottle_400 \
#     > ./logs/train2_400_act_0.5.log 2>&1 &



nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/square_d1_core/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0.2 \
    --dataset-repo-id ls/square_d1_core \
    --dataset-root /workspace/liushuo/cupid/datasets/square_d1_core\
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/outputs_new/retrain_cupid/ratio_drop_0.2 \
    --job_name square_d1_core \
    --policy.device cuda \
    --policy.repo_id ls/square_d1_core \
    > ./logs/train2_square_d1_core_act_0.2.log 2>&1 &

nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/square_d1_core/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0.5 \
    --dataset-repo-id ls/square_d1_core \
    --dataset-root /workspace/liushuo/cupid/datasets/square_d1_core\
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/outputs_new/retrain_cupid/ratio_drop_0.5 \
    --job_name square_d1_core \
    --policy.device cuda \
    --policy.repo_id ls/square_d1_core \
    > ./logs/train2_square_d1_core_act_0.5.log 2>&1 &


nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/square_d1_core/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0.8 \
    --dataset-repo-id ls/square_d1_core \
    --dataset-root /workspace/liushuo/cupid/datasets/square_d1_core\
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/outputs_new/retrain_cupid/ratio_drop_0.8 \
    --job_name square_d1_core \
    --policy.device cuda \
    --policy.repo_id ls/square_d1_core \
    > ./logs/train2_square_d1_core_act_0.8.log 2>&1 &

nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/square_d1_core/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0. \
    --dataset-repo-id ls/square_d1_core_1000\
    --dataset-root /workspace/liushuo/cupid/datasets/square_d1_core_1000\
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/outputs_new/retrain_cupid/ratio_drop_0. \
    --job_name square_d1_core_1000 \
    --policy.device cuda \
    --policy.repo_id ls/square_d1_core_1000 \
    > ./logs/train2_square_d1_core_act_0.log 2>&1 &



CUDA_VISIBLE_DEVICES=5 nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/act/mimicgen/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0.5 \
    --dataset-repo-id ls/mimicgen\
    --dataset-root /workspace/liushuo/cupid/datasets/square_d1\
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/act/mimicgen/retrain_cupid/ratio_drop_0.5 \
    --job_name mimicgen \
    --policy.device cuda \
    --policy.repo_id ls/mimicgen \
    > ./logs/train2_mimicgen_act_0.5.log 2>&1 &



CUDA_VISIBLE_DEVICES=7 nohup python scripts/train/retrain_act_subset.py \
    --retain-trak-pkl  /workspace/liushuo/cupid/act/mimicgen/score.pkl\
    --retain-metric mean_of_mean-succ-all \
    --retain-split train \
    --drop-ratio 0.2 \
    --dataset-repo-id ls/mimicgen\
    --dataset-root /workspace/liushuo/cupid/datasets/square_d1\
    --lerobot-repo /workspace/liushuo/cupid/lerobot_2.1 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/act/mimicgen/retrain_cupid/ratio_drop_0.2 \
    --job_name mimicgen \
    --policy.device cuda \
    --policy.repo_id ls/mimicgen \
    > ./logs/train2_mimicgen_act_0.2.log 2>&1 &