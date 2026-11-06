# ACT Influence 资源对比（all vs last-layer=4）

生成时间：2026-03-04

## 1. 数据来源

- all: `/workspace/liushuo/cupid/1_infulenece_score/can_mh/score.pkl`
- last4: `/workspace/liushuo/cupid/1_infulenece_score/can_mh/score_last4.pkl`
- stage 缓存目录：`/workspace/liushuo/cupid/1_infulenece_score/can_mh`

## 2. 参数规模对比

| 指标 | all | last-layer=4 | 对比 |
|---|---:|---:|---:|
| 参数维度 `param_dim` | 51,576,712 | 293,384 | last4 占 all 的 0.5688% |
| 维度缩减倍数 | 1x | - | all / last4 = 175.8x |

备注：`last4` 的 `metadata.selected_param_dim=293384`，`metadata.full_param_dim=51576712`。

## 3. 资源消耗对比（实际产物）

| 项目 | all | last4 | 对比 |
|---|---:|---:|---:|
| `stage1_train_projections.npy` | 980.56 MiB | 980.56 MiB | 基本相同 |
| `stage2_return_gradient.pt` | 196.75 MiB | 1.12 MiB | 175.5x |
| `stage3_inverse_hvp.pt` | 196.75 MiB | 1.12 MiB | 175.5x |
| 最终 `score*.pkl` | 1374.07 MiB | 982.81 MiB | 1.40x |

补充（理论内存量级）：

- 单个参数向量（fp32）
  - all: 196.75 MiB
  - last4: 1.12 MiB
- `proj_dim=4096` 的 dense 投影矩阵理论大小
  - all: 约 787 GiB
  - last4: 约 4.48 GiB

## 4. 时间消耗对比（可观测区间）

按文件时间戳计算：`stage1 文件写完 -> 最终 pkl 写完`

| 区间 | all | last4 | 提速 |
|---|---:|---:|---:|
| stage1 完成后到任务结束 | 20.59 分钟 | 15.00 分钟 | 1.37x |

说明：

- 该时间不包含 Stage1 本身（这里只能从已有产物时间戳观测）。
- Stage1 主要受训练样本遍历影响，all 与 last4 差异通常小于 Stage2/3。

## 5. 结论

- 参数维度上，`last-layer=4` 相比 `all` 缩减约 **175.8x**。
- Stage2/Stage3 的张量体积与参数维度近似线性，实际也缩减到约 **1/175**。
- 但端到端（从 Stage1 完成后开始计）实测提速约 **1.37x**，说明整体耗时仍受 Stage1/数据遍历等固定开销影响明显。
