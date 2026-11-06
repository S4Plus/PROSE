CUDA_VISIBLE_DEVICES=7 uv run scripts/compute_norm_stats.py --config-name pi05_cb81_lora

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_cb81_lora --exp-name=cb81_lora --overwrite


mkdir -p logs
CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
nohup uv run scripts/train.py pi05_cb81_lora --exp-name=cb81_lora --overwrite \
> logs/cb81_lora_$(date +%Y%m%d_%H%M%S).log 2>&1 &



#执行pi05 square_d1的任务执行命令
CUDA_VISIBLE_DEVICES=7 uv run scripts/compute_norm_stats.py --config-name pi05_square_d1_lora

CUDA_VISIBLE_DEVICES=7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
nohup uv run scripts/train.py pi05_square_d1_lora --exp-name=square_d1_lora --overwrite \
> logs/square_d1_lora.log 2>&1 &


#启动服务
CUDA_VISIBLE_DEVICES=7 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_square_d1_lora --policy.dir=/workspace/liushuo/cupid/openpi/checkpoints/pi05_square_d1_lora/square_d1_lora/29999

#客户端
cd ~/cupid/mimicgen/eval
python client_mimicgen_openpi05.py --video_dir ./openpi05



python mymimicgen/3_convert.py  \
   --source /workspace/liushuo/cupid/mimicgen/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5  \
   --output /workspace/liushuo/cupid/datasets/square_d1 \
   --repo-id ls/square_d1

CUDA_VISIBLE_DEVICES=6 nohup lerobot-train \
    --dataset.repo_id ls/square_d1 \
    --dataset.root=/workspace/liushuo/cupid/datasets/square_d1 \
    --output_dir ./outputs/square_d1 \
    --job_name square_d1 \
    --policy.type=act \
    --policy.repo_id ls/square_d1 \
    > ./train2_square_d1_act.log 2>&1 &