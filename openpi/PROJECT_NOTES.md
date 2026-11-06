# 需要修改的地方
config.py中的数据集路径


# 训练的命令
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_MODE=disabled \
nohup uv run scripts/train.py pi05_can_mh_lora --exp-name=pi05_can_mh_lora --overwrite \
> logs/pi05_can_mh_lora.log 2>&1 &

can替换成transport/square即可

client为测试端脚本，基于robomimic环境
robomimic_venv可直接复制conda环境