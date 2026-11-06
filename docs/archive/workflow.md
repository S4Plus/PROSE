# mimicgen采集 训练 再训练流程

## 数据采集
```bash
    cd ~/cupid/mimicgen
    conda activate mimicgen
    #下载所有的数据集
    python mimicgen/scripts/download_datasets.py --dataset_type source --tasks square
    #准备源数据
    python mimicgen/scripts/prepare_src_dataset.py --dataset datasets/source/coffee.hdf5 --env_interface MG_Coffee --env_interface_type robosuite



    python mimicgen/scripts/prepare_src_dataset.py \
        --dataset datasets/source/hammer_cleanup.hdf5 \
        --env_interface MG_HammerCleanup \
        --env_interface_type robosuite



    #robot迁移
    python mimicgen/scripts/generate_robot_transfer_configs.py

    python mimicgen/scripts/generate_core_configs.py

    python mimicgen/scripts/generate_dataset.py \
        --config ./tmp/core_configs/demo_src_square_task_D1.json \
        --auto-remove-exp

conda activate cupid
cd ~/cupid

python mymimicgen/3_convert.py  \
   --source /workspace/liushuo/cupid/mimicgen/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5  \
   --output /workspace/liushuo/cupid/datasets/square_d1_select200 \
   --repo-id ls/square_d1_select200
```

  python /workspace/liushuo/cupid/act/mimicgen/merge_lerobot_datasets.py \
    --dataset-a /workspace/liushuo/cupid/datasets/square_d1 \
    --dataset-b /workspace/liushuo/cupid/datasets/square_d1_select200\
    --output-dataset /workspace/liushuo/cupid/datasets/square_d1_1200 \
    --overwrite

cd ~/cupid
conda activate cupid


/workspace/liushuo/cupid/datasets/hammer_cleanup
/workspace/liushuo/cupid/datasets/square_d1_50
CUDA_VISIBLE_DEVICES=7 nohup lerobot-train \
    --dataset.repo_id ls/square_d1_50\
    --dataset.root=/workspace/liushuo/cupid/datasets/square_d1_50 \
    --output_dir ./outputs/square_d1_50 \
    --job_name square_d1_50\
    --policy.type=act \
    --policy.repo_id ls/square_d1_50 \
    > ./train2_square_d1_50_act.log 2>&1 &


CUDA_VISIBLE_DEVICES=7 nohup lerobot-train \
    --dataset.repo_id ls/hammer_cleanup \
    --dataset.root=/workspace/liushuo/cupid/datasets/hammer_cleanup \
    --output_dir ./outputs/hammer_cleanup \
    --job_name hammer_cleanup \
    --policy.type=act \
    --policy.repo_id ls/hammer_cleanup \
    > ./train2_hammer_cleanup_400_act.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup lerobot-train \
    --dataset.repo_id ls/square_d1 \
    --dataset.root=/workspace/liushuo/cupid/datasets/square_d1_1200 \
    --output_dir ./outputs/square_d1_1200 \
    --job_name square_d1_1200 \
    --policy.type=act \
    --policy.repo_id ls/square_d1_1200 \
    > ./train2_square_d1_1200_act.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup lerobot-train \
    --dataset.repo_id ls/square_d1_180 \
    --dataset.root=/workspace/liushuo/cupid/datasets/square_d1_180 \
    --output_dir ./outputs/square_d1_180 \
    --job_name square_d1_180 \
    --policy.type=act \
    --policy.repo_id ls/square_d1_180 \
    > ./logs/train2_square_d1_180_act.log 2>&1 &


CUDA_VISIBLE_DEVICES=7 nohup lerobot-train \
    --dataset.repo_id ls/hammer_cleanup_d1_core \
    --dataset.root=/workspace/liushuo/cupid/datasets/hammer_cleanup_d1_core \
    --output_dir ./outputs/hammer_cleanup_d1_core \
    --job_name hammer_cleanup_d1_core \
    --policy.type=act \
    --policy.repo_id ls/hammer_cleanup_d1_core \
    > ./logs/train2_hammer_cleanup_d1_core_act.log 2>&1 &

## 打分
CUDA_VISIBLE_DEVICES=7 nohup python ./act/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/act/mimicgen/retrain_cupid/ratio_drop_0.2/checkpoints/200000/pretrained_model \
    --train_dataset_path /workspace/liushuo/cupid/datasets/square_d1_filtered_0.8 \
    --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/act_0.8/episodes\
    --output_path ./act/mimicgen_0.8/score \
> logs/cupid_act_influence_0.8.log 2>&1 &
### 可视化
python scripts/visualize_online_trak_influence.py \
    --pkl-path ./act/mimicgen/score.pkl 

python plot_initial_positions_from_names.py \
    --root /workspace/liushuo/cupid/mimicgen/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5 \
    --object-key square_nut \
    --influence-file ./act/mimicgen/score.pkl \
    --train-dataset-path /workspace/liushuo/cupid/datasets/square_d1 \
    --grid-cols 2 \
    --grid-rows 10 \
    --x-range -0.25 -0.15 \
    --y-range -0.25 0.25 \
    --output /workspace/liushuo/cupid/act/mimicgen/score_grid_heatmap.png \
    --test-dir /workspace/liushuo/cupid/mimicgen/eval/act/eval_rollouts_act



### 热力图
python act/mimicgen/plot_score_heatmap.py \
    --method umap \
    --grid-size 80 \
    --sigma 1.5 \
    --output /workspace/liushuo/cupid/act/mimicgen/score_heatmap_umap.png \
    --train-dataset /workspace/liushuo/cupid/datasets/square_d1
## 再训练
```bash
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
```






  用模型计算训练集 loss（会生成 train_set_loss.mmap）

  python eval_policy_loss_act.py \
    --exp_name loss_eval \
    --eval_dir /workspace/liushuo/cupid/mimicgen/eval/act/ \
    --pretrained_dir /workspace/liushuo/cupid/mimicgen/eval/outputs/square_d1/checkpoints/200000/pretrained_model \
    --train_dataset_path /workspace/liushuo/cupid/datasets/square_d1 \
    --batch_size 4 \
    --num_timesteps 8 \
    --device cuda:0 \
    --overwrite True

  如果显存不够，先把 --batch_size 调小到 2 或 1。

  3) 运行网格分析（加入 --loss-file）

  python plot_initial_positions_from_names.py \
    --hdf5 /workspace/liushuo/cupid/mimicgen/tmp/core_datasets/square/demo_src_square_task_D1/demo.hdf5 \
    --object-key square_nut \
    --influence-file /workspace/liushuo/cupid/act/mimicgen/score.pkl \
    --loss-file /workspace/liushuo/cupid/mimicgen/eval/act/eval_rollouts_act/loss_eval/train_set_loss.mmap \
    --train-dataset-path /workspace/liushuo/cupid/datasets/square_d1 \
    --grid-cols 2 \
    --grid-rows 10 \
    --x-range -0.25 -0.15 \
    --y-range -0.25 0.25 \
    --output /workspace/liushuo/cupid/act/mimicgen/score_grid_heatmap.png \
    --test-dir /workspace/liushuo/cupid/mimicgen/eval/act/episodes



CUDA_VISIBLE_DEVICES=7 nohup python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
    --dataset.repo_id ls/hammer_cleanup \
    --dataset.root /workspace/liushuo/cupid/datasets/hammer_cleanup \
    --output_dir ./outputs/hammer_cleanup \
    --job_name hammer_cleanup \
    --policy.type=act \
    --policy.repo_id ls/hammer_cleanup \
    --score-pkl /workspace/liushuo/cupid/1_infulenece_score/hammer_cleanup_D0/score.pkl \
    --drop-weight-quantile 0.2 \
    --weight-mode window_mean \
    --use-weighted-sampler \
    --segment-mask \
    --segment-weight \
    > ./train2_hammer_cleanup_400_act.log 2>&1 &