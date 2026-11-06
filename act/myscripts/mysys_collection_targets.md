# MYSYS 删除结果到补采任务落地说明

本文档说明如何把 `filter_joint_episode_window.py` 的轨迹级、片段级删除结果，
转成采集侧可执行的补采清单：采集区域、采集姿态、轨迹特点修正。

## 1. 输入前提

先完成 MYSYS 两级筛选，得到一个 `selection-dir`，目录内至少包含：

- `episode_scores.csv`
- `window_scores.csv`

同时需要原始 LeRobot 数据集目录，例如：

- `/workspace/liushuo/cupid/datasets/lerobot/can_mh`

## 2. 生成补采清单

```bash
python /workspace/liushuo/cupid/act/myscripts/mysys_collection_targets_from_deletion.py \
  --selection-dir /path/to/filter_output \
  --dataset-root /path/to/lerobot_dataset \
  --output-dir /path/to/filter_output/collection_targets \
  --state-dims 8 \
  --action-dims 8 \
  --phase-bins 5 \
  --region-bins 4 \
  --top-k 12 \
  --min-deleted-records 2 \
  --replacement-ratio 1.0
```

## 3. 输出文件

- `collection_targets.md`：给人看的补采任务报告。
- `collection_targets.csv`：给排产、表格筛选和脚本消费的结构化清单。
- `collection_targets.json`：给后续自动化采集/主动学习系统读取。
- `deletion_region_records.csv`：每个被删或保留片段的区域归属与状态动作摘要。

## 4. 三个核心字段

### 4.1 采集区域 `collection_region`

含义：在哪个 state-action region、哪个轨迹时间相位补采。

字段会包含：

- `region_id`：PCA state-action 分箱区域，例如 `rx2_ry1_p3`
- `phase_name`：轨迹阶段，例如起始接近段、中段转换段、末端收敛段
- `state_center`：该区域被删样本的状态均值前几维
- `action_center`：该区域被删样本的动作均值前几维

解读原则：

- 删除占比高、保留 record 少：说明这个状态区域不是没有数据，而是已有数据质量低，需要补高质量替代样本。
- 片段删除集中在某个 phase：说明不要盲目整库扩增，应围绕该任务阶段定向补采。

### 4.2 采集姿态 `collection_pose`

含义：采集时机器人/末端应该以什么姿态进入该区域。

脚本会根据轨迹相位自动生成建议：

- 起始接近段：从稳定初始位姿进入，先对齐目标物或接触面。
- 中段转换段：保持抓取或接触约束稳定，沿主运动方向连续过渡。
- 末端收敛段：靠近目标终态时减速，末端姿态对齐放置、插入或堆叠目标。

如果出现 episode 级整条删除，说明只补单个片段不够，应该采完整成功轨迹。
如果主要是 window 级删除，可以围绕代表片段补采，但仍要保留进入和离开该姿态的上下文。

### 4.3 轨迹特点修正 `trajectory_corrections`

含义：和被删数据相比，新采样本必须修正什么行为缺陷。

常见修正项：

- 降低 `action jitter`：减少方向来回切换、夹爪高频开合和速度突变。
- 降低 `state jitter`：路径平滑推进，避免绕路、回退和接触后反复调整。
- 修正停滞片段：片段内必须有明确状态推进，不录长时间悬停。
- 提升局部价值：关键接触、对齐阶段放慢速度，动作连续且可复现。
- 提升整轨质量：必须成功收敛到任务终态，补足起始到终端的完整因果链。
- 增加覆盖多样性：同一区域采集不同初始扰动、物体微偏移和接近角。

## 5. 推荐使用方式

第一轮补采不要把所有目标都采满，建议只取 `collection_targets.md` 的 Top 3-5：

1. 每个目标区域先采 `suggested_episode_count` 条。
2. 采完后重新跑 influence/quality 和 MYSYS 筛选。
3. 如果同一区域仍持续被删，说明采集姿态或成功判据仍有系统性问题。
4. 如果删除占比下降但 eval 失败仍集中，继续补该区域的相邻 phase，而不是盲目扩全局数据。

这套闭环的核心不是“删多少补多少”，而是把被删区域当作低质量证据，
用更干净、更稳定、更成功的专家轨迹替换它们在状态-动作空间中的覆盖。
