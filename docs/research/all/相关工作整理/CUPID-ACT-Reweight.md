# CUPID-ACT 重加权与片段剔除说明

本文档说明 **CUPID-ACT 的“先重加权、再剔除极端负片段”** 策略如何落地，并给出训练脚本与输出文件含义。

## 1. 总体思路
- **重加权优先**：保留轨迹时序完整性，避免直接删片段导致断裂。
- **极端负片段驱动的整条轨迹剔除**：当前实现不直接删除单个片段，而是根据片段坏段比例 `bad_ratio` 过滤整条 episode。
- **原因**：纯剔除片段会破坏时序一致性；纯重加权可能保留“致命坏段”。因此先重加权，再用坏段比例对整条轨迹做审慎剔除。

## 2. 输入与中间量
- 输入：`cupid_act_influence.py` 生成的 `score.pkl`
- 必需字段（metadata）：
- `window_influence_scores`：片段影响力
- `window_weights`：片段权重（已归一化）
- `window_drop_mask`：片段是否被剔除
- `window_episode_index`：片段所属 episode

## 3. Episode 级统计
对每个 episode 计算：
- **bad_ratio**：该 episode 内被剔除片段比例
  - `bad_ratio = mean(window_drop_mask)`
- **episode_weight**：该 episode 内片段权重均值
  - `episode_weight = mean(window_weights)`

## 4. 过滤与重加权策略
1. **过滤**：
- 默认丢弃 `bad_ratio >= drop_bad_ratio` 的 episode。
- 可选：再按 `weight_source` 做分位数过滤（`drop_weight_quantile`）。
2. **重加权**：
- 训练时对 episode 赋予采样权重 `sampling_weight`。
- `weight_mode=window_mean`：使用 `episode_weight` 归一化。
- `weight_mode=influence`：使用 `influence_scores` 归一化。
- `weight_mode=none`：所有 episode 权重为 1。
3. **片段级 soft mask（可选）**：
- `--segment-mask`：将坏片段对应的样本权重设为 0，相当于 loss mask（不参与梯度）。
- `--segment-weight`：将 window 权重施加到样本级权重上，细粒度重加权。

## 5. 训练脚本
脚本位置：`/workspace/liushuo/cupid/act/train_act_cupid_reweight.py`

示例：
```bash
python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
  --dataset-path /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core \
  --score-pkl /workspace/liushuo/cupid/1_infulenece_score/hammer_cleanup_D0/score.pkl \
  --output-dir /workspace/liushuo/cupid/outputs/hammer_cleanup_d0_core_cupid \
  --drop-bad-ratio 0.3 \
  --weight-mode window_mean \
  --use-weighted-sampler \
  --max-steps 100000
```

关键参数：
- `--drop-bad-ratio`：剔除坏段比例阈值
- `--drop-weight-quantile`：按权重再剔除底部分位数
- `--weight-mode`：`window_mean | influence | none`
- `--use-weighted-sampler`：启用加权采样
- `--segment-mask`：启用片段级软剔除（样本权重为 0）
- `--segment-weight`：启用片段级权重（样本权重按 window 权重缩放）

## 6. 输出文件说明
脚本会在 `--output-dir` 下输出：
- `cupid_reweight_report.json`：总体统计（保留/剔除数量、分布统计、阈值）
- `cupid_reweight_episodes.csv`：每条 episode 的详细信息
- `cupid_reweight_keep.txt` / `cupid_reweight_drop.txt`
- Top/Bottom 排名：
- `cupid_top_weight_source.txt`
- `cupid_bottom_weight_source.txt`
- `cupid_top_sampling_weight.txt`
- `cupid_bottom_sampling_weight.txt`
- `cupid_top_bad_ratio.txt`
- `cupid_bottom_bad_ratio.txt`
- 直方图（CSV）：
- `cupid_hist_bad_ratio.csv`
- `cupid_hist_weight_source.csv`
- `cupid_hist_sampling_weight.csv`
- `cupid_hist_sample_weight.csv`（启用 WeightedSampler 时）

## 6.1 剔除掉的数据特点
基于当前默认策略（`drop_bad_ratio=0.3`，未启用 `drop_weight_quantile`）被剔除的 episode 具有以下共性：
- **坏片段占比高**：该 episode 的 `bad_ratio >= 0.3`，即至少 30% 片段被 `window_drop_mask` 标记为坏段。
- **片段影响力偏低**：坏段通常对应 `window_influence_scores` 处于分布的低端，意味着对回报贡献为负或接近 0。
- **权重均值偏小**：其 `episode_weight = mean(window_weights)` 往往低于总体均值，反映为训练贡献低。

> 说明：当前训练脚本**不会删除单个片段**，只会按 `bad_ratio` 删除整条 episode，并对保留 episode 进行重加权采样。

如需查看**具体被剔除的 episode 列表与分布**，请直接查看：
- `cupid_reweight_drop.txt`：被剔除的 episode 列表
- `cupid_reweight_episodes.csv`：每条 episode 的 bad_ratio/权重/是否保留
- `cupid_reweight_report.json`：整体分布统计

## 7. 经验默认值
- `drop_bad_ratio = 0.3`
- `weight_mode = window_mean`
- `drop_weight_quantile = 0.0`（先不启用）
- `use_weighted_sampler = True`

如需针对具体任务调参，可根据 `cupid_reweight_report.json` 的分布统计进行调整。



- drop-weight-quantile：按整体“低影响力”直接丢（全局排序）。
- drop-bad-ratio：按“坏片段比例”丢（局部质量）。
- 先用 drop-weight-quantile 单独控制比例，再叠加 drop-bad-ratio 做更严格筛选。

 python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
    --dataset-path /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core \
    --score-pkl /workspace/liushuo/cupid/1_infulenece_score/hammer_cleanup_D0/score.pkl \
    --output-dir /workspace/liushuo/cupid/outputs/hammer_cleanup_d0_core_cupid \
    --drop-weight-quantile 0.2 \
    --weight-mode window_mean \
    --use-weighted-sampler \
    --segment-mask \
    --segment-weight \
    --max-steps 100000


 <!-- python /workspace/liushuo/cupid/act/train_act_cupid_reweight.py \
    --dataset-path /workspace/liushuo/cupid/datasets/hammer_cleanup_d0_core \
    --score-pkl /workspace/liushuo/cupid/1_infulenece_score/hammer_cleanup_D0/score.pkl \
    --output-dir /workspace/liushuo/cupid/outputs/hammer_cleanup_d0_core_cupid_2 \
    --drop-weight-quantile 0.2 \
    --drop-bad-ratio 1.0 \
    --weight-mode window_mean \
    --use-weighted-sampler \
    --segment-mask \
    --segment-weight \
    --max-steps 100000 -->