# square_mh / can_mh 训练模型整理

## 说明

- ACT 统计来源：`3_model/`
- openpi05 统计来源：`openpi/checkpoints/`
- 仅统计当前仓库中已经存在训练配置或 checkpoint 的目录。
- `latest` 表示当前目录下可见的最大 checkpoint。
- ACT 目录同时存在 `step_XXXX` 与 `checkpoints/XXXXXX` 两种保存格式，这里按磁盘原样记录。
- openpi05 中有一批目录名带 `<PropagatingMissingType ...>`，这里按当前磁盘目录名原样保留，未做重命名。

## ACT

### square_mh

基目录：`3_model/`

- `square_mh` | latest: `100000`
- `square_mh_cupid_quality` | latest: `step_80000`
- `square_mh_influence_only_q02` | latest: `step_80000`
- `joint_episode_window/square_mh_joint_episode_window/train` | latest: `step_80000`

### can_mh

基目录：`3_model/`

#### 基线 / 主实验

- `can_mh` | latest: `100000`
- `can_mh_cupid_quality_dropout` | latest: `step_60000`
- `can_mh_joint_eval_0.2-0` | latest: `step_80000`
- `joint_episode_window_mlp_ratio/can_mh_ratio_mlp_edr0p10_ws50_st50_w0p4-0p4-0p2_cov0p5_mg0p15_rg0p5/train` | latest: `step_80000`

#### baseline_random

- `can_mh_model/baseline_random/can_mh_random_0_5_seed42_gpu7` | latest: `100000`
- `can_mh_model/baseline_random/can_mh_random_0_6_seed42_gpu4` | latest: `100000`
- `can_mh_model/baseline_random/can_mh_random_0_7_seed42_gpu5` | latest: `100000`
- `can_mh_model/baseline_random/can_mh_random_0_8_seed42_gpu6` | latest: `100000`
- `can_mh_model/baseline_random/can_mh_random_0_9_seed42_gpu7` | latest: `100000`

#### influence_only

- `can_mh_model/influence_only/can_mh_influence_only_q01` | latest: `step_80000`
- `can_mh_model/influence_only/can_mh_influence_only_q02` | latest: `step_80000`
- `can_mh_model/influence_only/can_mh_influence_only_q03` | latest: `step_80000`
- `can_mh_model/influence_only/can_mh_influence_only_q04` | latest: `step_80000`
- `can_mh_model/influence_only/can_mh_influence_only_q05` | latest: `step_80000`

#### joint

- `can_mh_model/joint/can_mh_joint_quality_csv_q01` | latest: `step_80000`
- `can_mh_model/joint/can_mh_joint_quality_csv_q02` | latest: `step_80000`
- `can_mh_model/joint/can_mh_joint_quality_csv_q03` | latest: `step_80000`
- `can_mh_model/joint/can_mh_joint_quality_csv_q04` | latest: `step_80000`
- `can_mh_model/joint/can_mh_joint_quality_csv_q05` | latest: `step_80000`

#### joint_strict

- `can_mh_model/joint_strict/can_mh_joint_score_last4_q01` | latest: `step_80000`
- `can_mh_model/joint_strict/can_mh_joint_score_last4_q02` | latest: `step_80000`
- `can_mh_model/joint_strict/can_mh_joint_score_last4_q03` | latest: `step_80000`
- `can_mh_model/joint_strict/can_mh_joint_score_last4_q04` | latest: `step_80000`
- `can_mh_model/joint_strict/can_mh_joint_score_last4_q05` | latest: `step_80000`
- `can_mh_model/joint_strict/can_mh_joint_score_last8_q01` | latest: `step_80000`
- `can_mh_model/joint_strict/can_mh_joint_score_last8_q02` | latest: `step_80000`
- `can_mh_model/joint_strict/can_mh_joint_score_last8_q03` | latest: `step_80000`
- `can_mh_model/joint_strict/can_mh_joint_score_last8_q04` | latest: `step_80000`
- `can_mh_model/joint_strict/can_mh_joint_score_last8_q05` | latest: `step_80000`
- `can_mh_model/joint_strict/can_mh_joint_score_last12_q01` | latest: `step_80000`
- `can_mh_model/joint_strict/can_mh_joint_score_last12_q02` | latest: `step_80000`

#### quality_only_model

- `can_mh_model/quality_only_model/can_mh_quality_only_keep05` | latest: `step_80000`
- `can_mh_model/quality_only_model/can_mh_quality_only_keep06` | latest: `step_80000`
- `can_mh_model/quality_only_model/can_mh_quality_only_keep07` | latest: `step_80000`
- `can_mh_model/quality_only_model/can_mh_quality_only_keep08` | latest: `step_80000`
- `can_mh_model/quality_only_model/can_mh_quality_only_keep09` | latest: `step_80000`

#### score_last4

- `can_mh_score_last4_q01` | latest: `step_80000`
- `can_mh_score_last4_q02` | latest: `step_80000`
- `can_mh_score_last4_q03` | latest: `step_80000`
- `can_mh_score_last4_q04` | latest: `step_80000`
- `can_mh_score_last4_q05` | latest: `step_80000`

#### score_last8

- `can_mh_score_last8_q01` | latest: `step_80000`
- `can_mh_score_last8_q02` | latest: `step_80000`
- `can_mh_score_last8_q03` | latest: `step_80000`

#### joint_by_score_ratio

- `joint_by_score_ratio/can_mh_jointsr_score_last4_dq01` | latest: `step_80000`
- `joint_by_score_ratio/can_mh_jointsr_score_last4_dq02` | latest: `step_80000`
- `joint_by_score_ratio/can_mh_jointsr_score_last4_dq03` | latest: `step_80000`
- `joint_by_score_ratio/can_mh_jointsr_score_last4_dq04` | latest: `step_80000`
- `joint_by_score_ratio/can_mh_jointsr_score_last4_dq05` | latest: `step_80000`
- `joint_by_score_ratio/can_mh_jointsr_score_last8_dq01` | latest: `step_80000`
- `joint_by_score_ratio/can_mh_jointsr_score_last8_dq02` | latest: `step_80000`
- `joint_by_score_ratio/can_mh_jointsr_score_last8_dq03` | latest: `step_80000`
- `joint_by_score_ratio/can_mh_jointsr_score_last8_dq04` | latest: `step_80000`
- `joint_by_score_ratio/can_mh_jointsr_score_last8_dq05` | latest: `step_80000`
- `joint_by_score_ratio/can_mh_jointsr_score_last12_dq01` | latest: `step_80000`
- `joint_by_score_ratio/can_mh_jointsr_score_last12_dq02` | latest: `step_80000`
- `joint_by_score_ratio/can_mh_jointsr_score_last12_dq03` | latest: `step_40000`
- `joint_by_score_ratio/can_mh_jointsr_score_last12_dq04` | latest: `step_40000`
- `joint_by_score_ratio/can_mh_jointsr_score_last12_dq05` | latest: `step_40000`

## openpi05

### square_mh

基目录：`openpi/checkpoints/pi05_square_mh_lora/`

- `square_mh_full_openpi05` | latest: `9999`
- `square_mh_scizor_joint_filter` | latest: `9999`

### can_mh

基目录：`openpi/checkpoints/pi05_can_mh_lora/`

#### 主实验

- `can_mh_demoscore` | latest: `9000`
- `can_mh_full_openpi05` | latest: `9999`
- `can_mh_joint_episode_window` | latest: `9999`
- `can_mh_joint_keep80_all_csvq` | latest: `3000`
- `can_mh_joint_keep90_all_csvq` | latest: `9999`
- `can_mh_scizor_joint_filter` | latest: `9999`
- `can_mh_window_joint_keep80_rank_fusion` | latest: `9999`
- `can_mh_window_joint_keep90_rank_fusion` | latest: `9999`
- `can_mh_window_last4_joint_sum_keep90_rank_fusion` | latest: `4000`

#### window_episode_prior

- `can_mh_window_episode_prior_sum_keep30_rank_fusion_w80_e20` | latest: `9999`
- `can_mh_window_episode_prior_sum_keep50_rank_fusion_w80_e20` | latest: `9999`
- `can_mh_window_episode_prior_sum_keep60_rank_fusion_w80_e20` | latest: `9999`
- `can_mh_window_episode_prior_sum_keep70_rank_fusion_w80_e20` | latest: `9999`
- `can_mh_window_episode_prior_sum_keep80_rank_fusion_w80_e20` | latest: `9999`
- `can_mh_window_episode_prior_sum_keep90_rank_fusion_w80_e20` | latest: `9999`
- `can_mh_window_episode_prior_sum_keep90_rank_fusion_w90_e10_apwindow_std_as30_i10_fnminmax_qamean_mw0_brepisode_prior` | latest: `9999`
- `can_mh_window_episode_prior_sum_keep90_rank_fusion_w90_e10_apwindow_std_as30_i10_fnminmax_qamean_mw0_brglobal` | latest: `9999`
- `can_mh_window_episode_prior_sum_keep90_rank_fusion_w90_e10_apwindow_std_as30_i10_fnminmax_qamean_mw1_brepisode_prior` | latest: `7000`

#### random

- `random/can_mh_random_keep80_seed0` | latest: `9999`
- `random/can_mh_random_keep80_seed42` | latest: `9999`
- `random/can_mh_random_keep90_seed0` | latest: `9999`
- `random/can_mh_random_keep90_seed42` | latest: `9999`

#### 目录名异常但已有训练产物

- `<PropagatingMissingType id='124586522632848'>_random_keep70_seed42` | latest: `9999`
- `<PropagatingMissingType id='126391896881872'>_random_mask_keep80_seed42` | latest: `9999`
- `<PropagatingMissingType id='126508620161680'>_random_keep60_seed42` | latest: `9999`
- `<PropagatingMissingType id='127855355558992'>_random_mask_keep60_seed42` | latest: `9999`
- `<PropagatingMissingType id='127910013647952'>_random_mask_keep50_seed42` | latest: `9999`
- `<PropagatingMissingType id='131170773695184'>_random_mask_keep90_seed42` | latest: `9999`
- `<PropagatingMissingType id='132755336536464'>_random_keep50_seed42` | latest: `9999`
- `<PropagatingMissingType id='134109269408400'>_random_mask_keep70_seed42` | latest: `9999`
- `<PropagatingMissingType id='134430884487568'>_random_keep70_seed0` | latest: `4000`
- `<PropagatingMissingType id='134551763172176'>_random_mask_keep40_seed42` | latest: `9999`
- `<PropagatingMissingType id='135987965279120'>_random_mask_keep30_seed42` | latest: `8000`
- `<PropagatingMissingType id='140610923963600'>_random_keep50_seed42` | latest: `3000`
