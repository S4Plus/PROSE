# CUPID-ACT 影响函数计算流程

为保证影响函数改进策略的可复现性，本文档总结了 CUPID-ACT 的四个阶段。所有阶段都在 ACT 策略的基础上完成，并与 `act/` 目录下的工具链保持一致。

## 6.1 阶段一：影子训练与梯度缓存（Shadow Training）
- **影子训练**：正常训练 ACT 策略 $\pi_{\theta^*}$ 直到收敛，得到与主模型同构的影子权重。
- **梯度预计算**：遍历训练数据集 $\mathcal{D}_{train}=\{d_i\}$，对每个样本 $d_i=(\mathbf{o}_i,\mathbf{a}_{i:i+k})$ 计算原始 L1 训练损失对应的梯度 $\nabla_\theta \mathcal{L}_{train}(d_i)$，并将其缓存。
- **随机投影降维**：若完整梯度维度为 $D$，使用随机投影矩阵 $P\in \mathbb{R}^{D \times M}$（例如 $M=4096$）得到压缩后的梯度 $P^\top \nabla_\theta \mathcal{L}_{train}(d_i)$，显著降低梯度缓存的存储开销。

## 6.2 阶段二：混合评估与回报采集（Hybrid Evaluation）
- **Rollout 收集**：在仿真或真实机器人中运行 $N$ 次评估 rollout，记录观察 $\mathbf{o}_t$、潜变量采样 $\mathbf{z}_t$ 以及执行的动作块 $\mathbf{a}_{t:t+k}$。
- **噪声注入**：当前实现仅在评估梯度阶段对 `action` 目标加入高斯噪声（默认 $\sigma=0.1$），用于增加梯度估计的方差与探索性。若需要对潜变量 $z_t$ 加噪，需要在 ACTPolicy 前向推理中显式注入。
- **分层/子任务回报**：支持从多个回报字段组合成总回报（如 `success, subtask_reward`），可按权重线性组合；用于缓解稀疏终回报带来的估计噪声。
- **Bootstrap 重复估计**：对同一批 rollouts 进行重复采样，得到多组回报梯度与影响值，用于计算均值与方差。
- **回报记录**：为每条轨迹 $\tau$ 存储回报 $R(\tau)$，用于后续的回报梯度加权。

## 6.3 阶段三：基于伪 Huber 的影响力计算（Influence Computation）
- **回报梯度构建**：使用回报中心化加权：
  \[
    \hat{g}_{\text{return}}=\frac{1}{N}\sum_{\tau}(R(\tau)-\bar{R})\sum_{t\in\tau}
      \nabla_\theta \mathcal{L}_{\text{PH}}(\mathbf{o}_t,\mathbf{a}_{t:t+k})
  \]
  其中 $\mathcal{L}_{\text{PH}}$ 为在评估过程中使用的伪 Huber 损失。
- **LiSSA 逆海森向量积**：通过 LiSSA（Linear-time Stochastic Second-Order Algorithm）求解 $s = H^{-1}\hat{g}_{\text{return}}$。在每一步 HVP 计算中都使用伪 Huber 损失，以避免与 ACT 结构不兼容的问题。
- **影响值**：对每个训练样本 $d_i$ 计算
  \[
    \text{Influence}(d_i) = -\nabla_\theta \mathcal{L}_{train}(d_i)^\top s
  \]
  该值衡量训练样本对评估回报的正负影响。

## 6.4 阶段四：数据筛选策略（Curation Strategy）
### 6.4.1 片段重加权 + 极端负片段剔除（推荐）
- **重加权优先**：在 window/片段级别计算影响力后，先对所有片段赋权（弱化噪声、保留时序完整性）。
- **极端负片段剔除**：对影响力最低的少量片段再做剔除，避免明显坏段持续干扰。
- **理由**：单纯删片段易破坏时序一致性；单纯重加权可能保留“致命坏段”。先重加权再剔除极端负段，是稳健与清理力度的折中。

**实现要点（已落地）**
- `--window_size/--window_stride`：启用片段级影响力窗口。
- `--window_weight_temperature`：控制片段权重分布的平滑程度。
- `--window_weight_clip_min/max`：限制权重范围，避免极端放大/缩小。
- `--window_drop_quantile` 或 `--window_drop_threshold`：剔除最差片段。

输出结果中会新增：
- `window_influence_scores`（片段分数）
- `window_weights`（片段权重）
- `window_drop_mask`（片段剔除标记）
### 6.4.2 轨迹级筛选（备选）
- 将 state-action 对的分数聚合成轨迹分数，直接过滤低分演示。
- 简单高效，但容易误删“包含关键好片段”的演示。
## 6.5 阶段五：再训练
- **排序**：根据 $\text{Influence}(d_i)$ 对训练数据排序。正值代表样本提升策略表现，负值代表样本造成性能下降。
- **筛除与重加权**：
  - 过滤策略：移除排名后 10%–30% 的负影响样本，缓解死锁、碰撞等失败案例的干扰。
  - 选择策略：对排名前若干百分比的高影响样本进行过采样或加权，增强成功轨迹的贡献。
- **再训练**：在完成数据筛选后，使用更新后的数据分布重新训练 ACT 策略，并重复评估以验证收益。

以上流程为 act 目录下影响函数实现的参考标准，可直接映射到 `train_trak_act.py` 与 `act/scoring_utils.py` 中的梯度缓存、LiSSA 估计及数据筛选逻辑。



  python /workspace/liushuo/cupid/act/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/outputs/hammer_cleanup_d0_core/checkpoints/100000/pretrained_model \
    --train_dataset_path /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core \
    --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/hammer_cleanup_D0/episodes  \
    --output_path /workspace/liushuo/cupid/1_infulenece_score/hammer_cleanup_D0/score \
    --reward_columns success,subtask_reward \
    --reward_weights 1.0,0.5 \
    --reward_reduce sum \
    --bootstrap_samples 5 \
    --window_size 16 \
    --window_stride 8 \
    --window_weight_temperature 1.0 \
    --window_drop_quantile 0.05 \
    --batch_size 4 \
    --num_workers 4 \
    --proj_dim 2048

  python /workspace/liushuo/cupid/act/cupid_act_influence.py \
    --pretrained_dir /workspace/liushuo/cupid/outputs/hammer_cleanup_d0_core/checkpoints/100000/pretrained_model \
    --train_dataset_path /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core \
    --eval_rollout_dir /workspace/liushuo/cupid/mimicgen/eval/hammer_cleanup_D0/episodes  \
    --output_path /workspace/liushuo/cupid/1_infulenece_score/hammer_cleanup_D0_8/score \
    --reward_columns success,subtask_reward \
    --reward_weights 1.0,0.5 \
    --reward_reduce sum \
    --bootstrap_samples 5 \
    --window_size 16 \
    --window_stride 8 \
    --window_weight_temperature 1.0 \
    --window_drop_quantile 0.05 \
    --batch_size 8 \
    --num_workers 4 \
    --proj_dim 2048