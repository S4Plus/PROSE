cd /workspace/liushuo/cupid/openpi

CUDA_VISIBLE_DEVICES=7 uv run scripts/compute_norm_stats.py --config-name pi05_can_mh_lora

nohup bash -lc 'CUDA_VISIBLE_DEVICES=4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
uv run scripts/train.py pi05_can_mh_lora --exp-name can_mh_delta_joint --overwrite' \
> logs/can_mh/can_mh_delta_joint.nohup.out 2>&1 &






CUDA_VISIBLE_DEVICES=3 uv run scripts/compute_norm_stats.py --config-name pi05_transport_mh_lora
CUDA_VISIBLE_DEVICES=3 nohup bash -lc 'XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
uv run scripts/train.py pi05_transport_mh_lora --exp-name transport_mh_full_openpi05 --overwrite' \
> logs/transport_mh_full_openpi05.log 2>&1 &