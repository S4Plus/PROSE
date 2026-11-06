# ACT Influence Stage 耗时对比（can_mh）

说明：基于 stage cache 文件时间戳估算。Stage1 本身起始时间无法从缓存直接恢复，因此这里对比的是 **Stage2 / Stage3 / 后处理**，以及 `stage1结束->结果落盘` 总时长。

| 方案 | Stage2 (`stage1->stage2`) | Stage3 (`stage2->stage3`) | 后处理(`stage3->pkl`) | `stage1->pkl` 总时长 | Stage2 相对 all | Stage3 相对 all | 总时长相对 all |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 20m30s | 0m4s | 0m1s | 20m35s | 1.00x | 1.00x | 1.00x |
| last4 | 14m56s | 0m4s | 0m1s | 15m0s | 1.37x | 1.12x | 1.37x |
| last8 | 14m50s | 0m4s | 0m1s | 14m54s | 1.38x | 1.11x | 1.38x |
| last12 | 14m40s | 0m4s | 0m1s | 14m45s | 1.40x | 1.10x | 1.40x |

文件时间戳基准：
- all: `score.stage1_train_projections.npy` / `score.stage2_return_gradient.pt` / `score.stage3_inverse_hvp.pt` / `score.pkl`
- last4: `score_last4.stage1_train_projections.npy` / `score_last4.stage2_return_gradient.pt` / `score_last4.stage3_inverse_hvp.pt` / `score_last4.pkl`
- last8: `score_last8.stage1_train_projections.npy` / `score_last8.stage2_return_gradient.pt` / `score_last8.stage3_inverse_hvp.pt` / `score_last8.pkl`
- last12: `score_last12.stage1_train_projections.npy` / `score_last12.stage2_return_gradient.pt` / `score_last12.stage3_inverse_hvp.pt` / `score_last12.pkl`