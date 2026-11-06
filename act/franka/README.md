我需要的执行client_act的流程是：
1. 最开始按r归位
2. 按t开始一次
3. 推理结束按0/1记录失败or成功
4. 按r归位
5. 按t开始下一次



 CUDA_VISIBLE_DEVICES=1 nohup lerobot-train \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.repo_id=local/stack_cube_franka_joint_act \
    --dataset.repo_id=local/stack_cube_franka_joint \
    --dataset.root=/workspace/liushuo/dataset/stack_cube_franka_joint_lerobot \
    --output_dir=/workspace/liushuo/cupid/outputs/stack_cube_franka_joint_act \
    --job_name=stack_cube_franka_joint_act \
    > /workspace/liushuo/cupid/act/logs/stack_cube_franka_joint_act.log 2>&1 &





 CUDA_VISIBLE_DEVICES=2 \
  WANDB_MODE=disabled \
  uv run scripts/compute_norm_stats.py \
    --config-name pi05_stack_cube_franka_lora

 CUDA_VISIBLE_DEVICES=2 \
 WANDB_MODE=disabled \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  nohup uv run scripts/train.py pi05_stack_cube_franka_lora \
    --exp-name=stack_cube_franka_joint_delta \
    --overwrite \
    > logs/stack_cube_franka_joint_delta.log 2>&1 &


nohup bash -lc 'CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled uv run scripts/compute_norm_stats.py --config-name pi05_stack_cube_franka_lora && CUDA_VISIBLE_DEVICES=2 WANDB_MODE=disabled XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_stack_cube_franka_lora --exp-name=stack_cube_franka_joint_delta --overwrite' > logs/stack_cube_franka_joint_delta.log 2>&1 &