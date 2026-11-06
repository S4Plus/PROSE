# Setup Instructions

## Set Up Conda Environment

```bash
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```



cd /workspace/liushuo/cupid/openvla-oft

  python vla-scripts/serve_policy_openvla.py \
    --checkpoint-dir /workspace/liushuo/cupid/openvla-oft/runs/can_mh_openvla_lora_fixed_stats--10000_chkpt \
    --dataset-root /workspace/liushuo/cupid/datasets/lerobot/can_mh\
    --host 0.0.0.0 \
    --port 8000 \
    --default-prompt "Place a coke can from a large bin into a smaller target bin."

  client 端：

  cd /workspace/liushuo/cupid/mimicgen/eval

  OPENVLA_SERVER_URL="ws://127.0.0.1:8000" \
  python client_robomimic_openvla.py \
    --env_name can \
    --n_rollouts 20 \
    --timeout_s 60 \
    --test_init_mode mixed \
    --video_dir ./tmp/openvla_videos \
    --save_rollouts_dir ./tmp/openvla_rollouts \
    --summary_path /tmp/openvla_summary.json