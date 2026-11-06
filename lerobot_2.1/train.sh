# lerobot-train --dataset.repo_id tpy/usb_insert_1004   --policy.type smolvla   --output_dir /home/mirrow/tpy/lerobot/outputs/train/smothVLA/usb_insert_real_v2   --job_name usb_insert   --policy.device cuda   --wandb.enable False   --policy.repo_id mirrow/usb_insert

# nohup lerobot-train --dataset.repo_id tpy/peg-in-hole-sim   --policy.type smolvla   --output_dir /home/mirrow/tpy/lerobot/outputs/train/smothVLA/peg-in-hole-sim   --job_name peg-in-hole-sim   --policy.device cuda   --wandb.enable False   --policy.repo_id tpy/eg-in-hole-sim & 

nohup lerobot-train --dataset.repo_id pj/Belt_Pick_Bottle_400_normal \
            --policy.type act \
            --output_dir /home/lerobot_2.1/outputs/train/act/pj/Belt_Pick_Bottle_400_normal_11_13 \
            --job_name Belt_Pick_Bottle_400_normal_11_13 \
            --policy.device cuda \
            --wandb.enable False \
            --policy.repo_id pj/Belt_Pick_Bottle_400_normal \
            > ./logs/train2_400_act_0.8.log 2>&1 &




nohup lerobot-train --dataset.repo_id ls/square_d1_core_1000 \
            --policy.type act \
            --output_dir /workspace/liushuo/cupid/outputs/square_d1_1000 \
            --job_name square_d1_core_1000 \
            --policy.device cuda \
            --policy.repo_id ls/square_d1_core_1000 \
            > ./logs/train2_square_d1_core_1000_act_0.log 2>&1 &




nohup lerobot-train \
    --dataset.root /workspace/liushuo/cupid/datasets/square_d1_core_1000 \
    --dataset.repo_id ls/square_d1_core_1000 \
    --policy.type act \
    --output_dir /workspace/liushuo/cupid/outputs/square_d1_1000 \
    --job_name square_d1_core_1000 \
    --policy.device cuda \
    --policy.repo_id ls/square_d1_core_1000 \
    --policy.push_to_hub false \
    > ./logs/train2_square_d1_core_1000_act_0.log 2>&1 &