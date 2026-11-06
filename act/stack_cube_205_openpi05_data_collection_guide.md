# Stack Cube 205 OpenPI05 数据采集指导文档

本文档用于指导 `stack_cube_205` 任务的数据采集与清洗，目标是让新增数据更符合 `OpenPI05` 在该任务上的真实偏好，从而提高训练收益。

参考的现有分析与可视化方式：

- `compare_act_openpi_can_mh` 风格输出目录：`/workspace/liushuo/cupid/output/compare_act_openpi_can_mh`
- OpenPI05 训练配置：[`../openpi/src/openpi/training/config.py`](/workspace/liushuo/cupid/openpi/src/openpi/training/config.py:978)
- `stack_cube_205` 数据转换脚本：[`franka/convert_stack_cube_hdf5_to_lerobot.py`](/workspace/liushuo/cupid/act/franka/convert_stack_cube_hdf5_to_lerobot.py:1)
- joint episode-window 筛选脚本：[`myscripts/filter_joint_episode_window.py`](/workspace/liushuo/cupid/act/myscripts/filter_joint_episode_window.py:133)
- OpenPI05 从 ACT 筛选结果构造训练子集：[`../openpi/myscripts/prepare_openpi_from_act_filter.py`](/workspace/liushuo/cupid/openpi/myscripts/prepare_openpi_from_act_filter.py:203)

---

## 1. 执行摘要

对 `stack_cube_205` 的实际筛选结果表明，`OpenPI05` 更偏好以下类型的数据：

- **终点状态一致**：成功放置后的末态收敛，姿态分布不要太散。
- **局部动作平滑**：连续约 `50` 帧尺度上的关节变化应平滑、可预测。
- **过程单调推进**：接近、抓取、抬升、对位、放置应持续朝目标推进，少回撤、少反复修正。
- **关键阶段清晰**：真正高价值的是“稳定接近/抓取”和“稳定放置/脱离”，而不是长时间搜索或悬停。

简化成一句话：

> OpenPI05 更喜欢“成功且风格统一的专家轨迹”，而不是“成功但包含大量犹豫、抖动和修补动作的轨迹”。

---

## 2. 任务与模型约束

### 2.1 当前训练配置的直接含义

`pi05_stack_cube_205_lora` 的关键配置如下：

- `action_horizon=50`
- `use_delta_joint_actions=True`
- `batch_size=32`
- `num_train_steps=30000`

这意味着模型不是只学习单步动作，而是在学习一段长度约为 `50` 帧的关节动作序列模式。

在 `30 FPS` 下：

- `50` 帧约等于 `1.67s`

因此，新增数据是否有价值，很大程度上取决于：

- `1~2s` 的连续动作段是否平滑
- 该动作段是否能被稳定预测
- 该段是否处于真正推动任务完成的关键阶段

### 2.2 当前数据表达

`stack_cube_205` 转换后的训练数据具有以下语义：

- `state = 当前 qpos(7) + 当前 gripper_cmd(1)`
- `action = 下一时刻 qpos(7) + 下一时刻 gripper_cmd(1)`
- 双相机图像：
  - `fixed_camera`
  - `wrist_camera`

这意味着：

- 关节轨迹的连续性会直接反映到训练 action 上
- 夹爪开闭抖动会直接污染 action token 分布
- 视觉端如果前期长时间搜索，会增加无效上下文

---

## 3. 已有筛选结果反映出的 OpenPI05 偏好

### 3.1 宏观保留比例

在 `stack_cube_205` 上，OpenPI05 使用的 joint episode-window 筛选结果为：

- 总 episode：`205`
- 保留 episode：`185`
- 删除 episode：`20`
- 总样本帧：`144317`
- 保留训练帧：`89043`
- 最终帧保留比例：约 `61.7%`
- 总窗口：`2699`
- 保留窗口：`1836`
- 窗口保留比例：约 `68.0%`

这说明筛选不是简单“删掉坏轨迹”，而是：

1. 先做 episode 级粗筛
2. 再在保留 episode 内继续裁掉低价值窗口
3. 最终形成更稠密的高价值局部片段子集

### 3.2 时间位置偏好

按窗口在 episode 中的相对时间位置统计，保留率大致如下：

- `0% - 25%`：`47.0%`
- `25% - 50%`：`84.9%`
- `50% - 75%`：`54.2%`
- `75% - 100%`：`84.6%`

这表明模型更喜欢：

- 中前段已经开始稳定接近和执行的动作
- 末段放置成功附近与收尾稳定片段

模型不太喜欢：

- 起始阶段长时间搜索、试探、犹豫
- 中后段反复修正、悬停和来回调整

### 3.3 平滑性偏好

保留窗口与未保留窗口的统计差异非常显著：

- 保留窗口 `raw_action_jitter ≈ 0.00385`
- 删除窗口 `raw_action_jitter ≈ 0.01143`
- 保留窗口 `raw_state_jitter ≈ 0.00385`
- 删除窗口 `raw_state_jitter ≈ 0.01143`

可以近似理解为：

- 被保留窗口的局部动作抖动只有被删除窗口的约 `1/3`

这说明对于 OpenPI05：

- **平滑性不是锦上添花，而是硬偏好**

### 3.4 第一阶段 episode 偏好

第一阶段筛选不是只看 influence，而是：

`episode_joint_score = 0.5 * norm(influence) + 0.5 * norm(trajectory_eval quality)`

其中 `trajectory_eval` 的核心奖励项包括：

- `terminal consistency`
- `efficiency`
- `progress`
- `monotonicity`
- `state smoothness`
- `action smoothness`

惩罚项包括：

- `backtrack`
- `stall`

所以，OpenPI05 更偏好：

- 成功后末态一致
- 路径高效
- 朝目标持续推进
- 很少回撤
- 很少卡住不动

---

## 4. 对采集脚本的修改原则

下面的建议按优先级排序，优先改前四项。

### 4.1 减少长搜索段

目标：

- 缩短无效前摇
- 让 episode 尽快进入有效操作阶段

建议：

- 每次示教前把机械臂 reset 到固定且合理的预备姿态
- 初始方块位置允许变化，但不要让示教者必须长时间找目标
- 如果超过设定阈值仍未进入稳定接近阶段，直接终止并重采

推荐约束：

- `pre_grasp_search_too_long`
- 建议阈值：`1.0s ~ 1.5s`

### 4.2 强约束回撤与蛇形修正

目标：

- 避免“接近一下又退回去，再接近”的局部模式

建议：

- 对位阶段允许少量修正，但不允许多次明显回撤
- 如果末端在目标附近出现连续多次方向翻转，判为低质量
- 放置前允许一次小修正，不允许长期悬停磨位置

推荐标签：

- `approach_backtrack_excessive`
- `place_phase_hover_too_long`

### 4.3 提高 50 帧尺度上的可预测性

目标：

- 让 `action_horizon=50` 对应的动作段更连续、更模板化

建议：

- 抓取闭合前后动作尽量连续
- 抓取后抬升动作尽量一气呵成
- 放置下降与脱离尽量一气呵成
- 避免急刹、急转、急抖

推荐标签：

- `joint_delta_jitter_excessive`

### 4.4 统一末态

目标：

- 减少“都成功，但终点姿态分散”的问题

建议：

- 蓝块放到红块上的姿态尽量一致
- 成功后保留 `10 ~ 20` 帧稳定停留
- 放置成功后的撤手姿态尽量固定

推荐标签：

- `post_place_terminal_unstable`

### 4.5 约束夹爪事件

目标：

- 降低二值 gripper token 的噪声

建议：

- 抓取前避免频繁开合
- 抓取闭合后至少保持一段稳定时间
- 放置释放后也保持一段稳定张开

推荐标签：

- `gripper_toggle_near_grasp`

### 4.6 初始条件可以多样，执行风格不要太散

建议保留的多样性：

- 方块初始位置轻微变化
- 起手相位轻微变化
- 光照和视角轻微变化

建议压缩的多样性：

- 接近路径风格差异
- 抓取节奏差异
- 放置对位策略差异
- 成功后的撤离动作差异

原则是：

> 场景可以变，专家操作模板不要乱。

---

## 5. 推荐加入采集脚本的质量标签

建议在采集脚本中增加如下标签，支持在线中止或离线过滤：

- `pre_grasp_search_too_long`
- `approach_backtrack_excessive`
- `joint_delta_jitter_excessive`
- `gripper_toggle_near_grasp`
- `place_phase_hover_too_long`
- `post_place_terminal_unstable`

推荐使用方式：

- 在线采集阶段：若触发高优先级错误，直接终止重采
- 离线清洗阶段：若触发中优先级错误，标记为低质量候选

---

## 6. 采集脚本中的在线门控建议

下面给出一套可直接落地的在线门控逻辑。

### 6.1 起始搜索门控

当以下条件持续超过阈值时，终止重采：

- 末端与目标始终未进入有效接近区
- 夹爪未进入抓取准备状态
- 机械臂主要在做搜索而不是推进任务

建议阈值：

- `search_timeout_frames = 30 ~ 45`

### 6.2 抖动门控

计算最近 `N` 帧的关节增量与状态增量：

- 若 `mean(||Δq||)` 超阈值且缺乏有效任务进展，则判为抖动

建议窗口：

- `N = 15 ~ 25`

建议结合条件：

- 抖动高
- 进展低
- 位置变化方向频繁翻转

### 6.3 悬停门控

如果机械臂在目标附近长时间停滞且没有显著状态推进：

- 判为 `hover_too_long`

建议场景：

- 抓取前在物体上方长期悬停
- 放置前在目标上方长期微调

### 6.4 末态稳定门控

当检测到成功放置后：

- 强制继续记录 `10 ~ 20` 帧
- 如果这段内 block 或末端状态变化过大，则标记末态不稳定

---

## 7. 推荐的离线清洗指标

建议每条 episode 至少计算以下统计量：

- `length`
- `action_jitter`
- `state_jitter`
- `action_energy`
- `state_energy`
- `state_displacement`
- `terminal_state_norm`
- `success`
- `post_success_stability`

建议每个窗口至少计算：

- `window_action_jitter`
- `window_state_jitter`
- `window_progress`
- `window_backtrack_ratio`
- `window_hover_ratio`
- `window_gripper_event_count`

---

## 8. 推荐的可视化输出结构

建议新增一个和 `compare_act_openpi_can_mh` 风格一致的输出目录，例如：

`/workspace/liushuo/cupid/output/compare_act_openpi_stack_cube_205`

目录建议包含：

- `comparison_summary.json`
- `embedding_analysis.md`
- `episode_comparison.csv`
- `window_comparison.csv`
- `episode_feature_embedding.png`
- `episode_rank_scatter.png`
- `episode_score_curves.png`
- `keep_overlap_bar.png`
- `topk_jaccard.png`
- `window_delta_heatmap.png`

### 8.1 `comparison_summary.json`

建议包含：

- episode 保留统计
- frame 保留统计
- window 保留统计
- 不同 cohort 的统计均值
- top-k overlap / Jaccard

### 8.2 `episode_comparison.csv`

建议包含列：

- `episode`
- `act_keep`
- `pi05_keep`
- `cohort`
- `act_score_percentile`
- `pi05_score_percentile`
- `delta_percentile`
- `quality_score`
- `length`
- `action_jitter`
- `state_jitter`
- `state_displacement`
- `terminal_state_norm`

### 8.3 `window_comparison.csv`

建议包含列：

- `episode`
- `start`
- `end`
- `act_score_percentile`
- `pi05_score_percentile`
- `delta_percentile`
- `window_action_jitter`
- `window_state_jitter`
- `window_progress`
- `window_keep`

### 8.4 `window_delta_heatmap.png`

用途：

- 观察 OpenPI05 更偏好哪些时间段
- 识别“长搜索段”“修修补补段”“高价值关键段”

### 8.5 `episode_feature_embedding.png`

用途：

- 观察高价值轨迹是否成簇
- 观察低价值轨迹是否集中在某类统计形态
- 观察新增数据是否落入已有高价值流形

---

## 9. 采集策略优先级建议

如果只能改一轮脚本，建议按下面顺序实施：

1. **标准化起始姿态**
2. **加入长搜索超时**
3. **加入抖动与回撤门控**
4. **加入成功后末态稳定保留**
5. **加入夹爪事件约束**
6. **再考虑扩大初始场景多样性**

原因：

- 前四项直接决定 OpenPI05 是否能学到更稳定的 chunk 级动作模板
- 后两项属于进一步精修

---

## 10. 直接可执行的脚本修改清单

### 10.1 必改项

- 加入统一 reset 姿态
- 加入 `search timeout`
- 加入 `joint jitter` 在线检测
- 加入 `success 后额外稳定记录`

### 10.2 强烈建议项

- 记录每段的阶段标签：接近 / 抓取 / 抬升 / 对位 / 放置 / 脱离
- 记录夹爪事件时间戳
- 记录 episode 级与窗口级质量标签

### 10.3 可选增强项

- 自动生成离线清洗报告
- 自动生成可视化目录
- 自动把新增数据映射到现有 embedding 上看是否落在高价值簇

---

## 11. 最终采集原则

本任务上最重要的不是“增加成功数量”，而是：

- 降低搜索段比例
- 降低回撤和抖动
- 强化连续稳定执行
- 强化终点一致性

最终应追求的数据风格是：

> 起手清晰、接近干净、抓取连贯、放置稳定、收尾一致。

而应尽量避免的数据风格是：

> 成功但过程充满犹豫、悬停、来回修正和末态漂移。

---

## 12. 推荐下一步

建议下一步做两件事：

1. 把上述 6 个质量标签接入采集脚本
2. 生成一套 `stack_cube_205` 的 `compare_act_openpi_can_mh` 风格可视化目录，用于验证新增数据是否真的更接近 OpenPI05 偏好

如果后续要继续推进，优先实现：

- 在线门控版采集脚本
- 离线清洗与可视化脚本

