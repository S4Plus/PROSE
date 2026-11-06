CUDA_VISIBLE_DEVICES=7 nohup lerobot-train \
    --dataset.repo_id ls/CB81 \
    --dataset.root=/ext_workspace/liushuo/swk/CB81 \
    --output_dir ./swk/outputs/CB81 \
    --job_name CB81 \
    --policy.type=act \
    --policy.repo_id ls/CB81 \
    > ./swk/logs/train2_cb81_act.log 2>&1 &



CUDA_VISIBLE_DEVICES=6 nohup lerobot-train \
    --dataset.repo_id ls/merge \
    --dataset.root=/workspace/liushuo/cupid/act/swk/merge \
    --output_dir ./swk/outputs/merage \
    --job_name merage \
    --policy.type=act \
    --policy.repo_id ls/merage \
    > ./swk/logs/train2_merage_act.log 2>&1 &

#200
CUDA_VISIBLE_DEVICES=6 nohup lerobot-train \
    --dataset.repo_id ls/six \
    --dataset.root=/workspace/liushuo/cupid/act/swk/six_object_pick_conveyor_first2\
    --output_dir ./swk/outputs/six \
    --job_name six_object_pick_conveyor_first2 \
    --policy.type=act \
    --policy.repo_id ls/six \
    --dataset.tolerance_s=0.01 \
    > ./swk/logs/train2_six_object_pick_conveyor_first2_act.log 2>&1 &


cd ~/cupid/act
CUDA_VISIBLE_DEVICES=6 nohup lerobot-train \
    --dataset.repo_id ls/120 \
    --dataset.root=/workspace/liushuo/dataset/swk/120\
    --output_dir ./swk/outputs/120 \
    --job_name 120_object_pick_conveyor_first2 \
    --policy.type=act \
    --policy.repo_id ls/120 \
    --dataset.tolerance_s=0.01 \
    > ./swk/logs/train2_120_object_pick_conveyor_first2_act.log 2>&1 &