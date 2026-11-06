
# 1. 数据与任务定义

## 1.1 你需要在每条轨迹中保证可读到（最低要求）

* 每个时刻 (t)：obs（图像/状态）、action（ACT 的监督标签）、robot state
* 物体位姿（建议 6D；最低 3D 位置）

  * peg：(p_{peg}(t))（及朝向 (R_{peg}(t)) 若有）
  * nut：(p_{nut}(t))（及朝向 (R_{nut}(t)) 若有）
原始的hdf5中有object：包含位置和朝向

## 1.2 关键相对量（用于“难度分桶 + 再采集调度”）

在“开始插入前”的关键时刻 (t^*)（定义见 2.2）计算：

* (\Delta p = p_{nut}(t^*) - p_{peg}(t^*))
* (r_{xy}=\sqrt{\Delta x^2+\Delta y^2})（水平对准误差）
* (\Delta z)（高度差）
* 若有姿态：(\Delta\theta)=nut 轴与 peg 轴夹角（或 nut yaw 相对 peg）

---

# 2. 先做一次结构化：阶段切分与关键时刻定位（强烈建议）

## 2.1 阶段切分（用于特征统计与异常剔除）

用“夹爪状态 + 物体高度/相对距离变化”自动切成 6 段：对轨迹进行划分

1. approach nut
2. grasp（夹爪闭合、nut 速度/相对距离突变）
3. lift/clear（nut z 上升到阈值）
4. move above peg（nut 到 peg 的水平距离快速下降）
5. rotate-align（nut 朝向变化显著、且保持在 peg 附近）
6. insert（nut z 下降并接近最终 seat 高度）

## 2.2 关键时刻 (t^*)（用于分桶与影响分析）

* (t^* = t_{\text{insert_start}})：进入 insert 段的第一个时刻
  若分段不稳，可用替代定义：
* 在 nut 距 peg 最近、且 nut z 从上向下开始持续下降的时刻。

产出：每条轨迹一个记录（id, (r_{xy}), (\Delta z), (\Delta\theta), 分段边界, 质量统计）。

---

# 3. 固定 ACT 的闭环评估协议（影响函数稳定性的前提）

## 3.1 固定评估 reset seeds（必须包含“困难分布”）

构造一个固定评估集 (\mathcal{S}_{eval})，建议 150–300 个 ：

* peg：工作区网格点（3×3 或 4×4）+ 小扰动
* nut：以 peg 为中心的条件化采样

  * 距离：近/中/远 = 5:3:2（近距离让任务聚焦对准插入）
  * 角度（若可控）：困难角度占比高（例如 45–90° 占 50%）
* 每个 seed 可重复 rollout 2–3 次（若策略是随机策略）降低方差

## 3.2 评估指标

* 主指标：成功率 SR（成功=1 失败=0）
* 必做：按桶统计 SR（例如按 (|\Delta\theta|)、(r_{xy}) 分桶的困难桶 SR）

---

# 4. 训练基线 ACT（All-1000）并缓存评估轨迹

1. 用全量 1000 条训练 ACT，得到参数 (\theta_0)
2. 在固定 (\mathcal{S}_{eval}) 上 rollout，记录每个 episode 的 ((o_t,a_t))、终止 reward（成功=1）与必要的 log-prob 信息（见 5.2）

产出：

* `SR_all`（总 SR + 各困难桶 SR）
* eval rollout 缓存（供 (\nabla_\theta J)）

---

# 5. CUPID 影响函数：
    在cupid_act_influence.md中详细说明
# 6. 用影响分数做筛选：必须同时考虑“多样性覆盖”

你现在全成功，最大风险是：**易样本冗余 + 偶然成功的坏习惯（抖动/绕路）**。因此筛选规则建议如下。

## 6.1 先做“负影响剔除/降权”

* D-：Infl < 0（或 bottom 10–20%）→ **删除**或权重极低（0.1）
* D0：接近 0（中间 10–20%）→ **降权**（0.3–0.5）
* D+：Infl 高的（top 60–80%）→ 主训练集

## 6.2 再做“按难度桶分层保留”（保证泛化）

用你能拿到的物体位姿，在 (t^*) 处分桶，例如：

* (r_{xy})：0–5mm / 5–15mm / 15–30mm / >30mm
* (|\Delta\theta|)：0–5° / 5–15° / 15–45° / 45–90°

策略：

* 每个桶至少保留 m 条（m=5–20，视桶数而定）
* 若某桶样本少但 Infl 高：该桶样本**全部保留**，并标记为再采集重点

产出：筛选后的训练集 `D_selected`（通常 500–800 条更健康）。

---

# 7. 重新训练 ACT 并验证收益（必须做对照）

训练三组（至少重复 2 次种子）：

1. ACT@All（1000）
2. ACT@Selected（筛选集）
   3)（可选）ACT@Selected + D0(降权)（更稳）

评估：

* 总 SR
* 困难桶 SR（高角度桶、小容差桶）
* 行为质量（插入耗时、对准微调次数、抖动指标）

判据：

* 若总 SR 差不多但困难桶 SR 上升，说明筛选有效（全成功数据常见现象）。

---

# 8. 再采集：用“覆盖缺口 × 影响收益”定配额（核心）

## 8.1 先做一张优先级表

对每个桶 (b) 统计：

* 覆盖：(n_b)（筛选后样本数）
* 平均影响：(\overline{\text{Infl}}_b)
* 该桶 eval 成功率：(SR_b)

优先级建议：

1. (n_b) 低 + (\overline{\text{Infl}}_b) 高（稀缺但关键）
2. (SR_b) 低（模型当前短板）
3. 工作区边界/视角边界桶（迁移风险高）

## 8.2 再采集怎么采（利用你能拿到的物体位姿做条件化 reset）

每次 reset 按目标桶反推初始条件：

* peg：网格点 + 扰动（保证空间覆盖）
* nut：以 peg 为中心采样

  * 近距离（5–10cm）为主：让数据聚焦对准插入
  * 高角度（45–90°）过采样：逼出 rotate-align 行为
  * 采不同接近方位（左/右/前）提升鲁棒性

## 8.3 给一个可直接执行的默认配额（再采 400 条示例）

* 240 条：高角度对孔成功（45–90°）
* 100 条：小容差插入成功（插入前 (r_{xy}\le 15)mm 的过程，包含最后 2–3cm 微调）
* 60 条：边界多样性（peg 在边缘网格点 + 不同接近方向/抓取点）

采完后：

* 合并数据
* 只需对新增数据增量计算 (g_i) 并更新 influence（或每轮重算一次也行）
* 进入下一轮筛选-训练-评估

---

# 9. 一轮完整闭环的“执行清单”（你可以直接按这个跑）

**Round 0（基线）**

1. 结构化 1000 条：分段 + (t^*) + 分桶特征
2. 固定 (\mathcal{S}_{eval})（含困难 seeds）
3. Train ACT@All，得到 `SR_all` 并缓存 eval rollouts

**Round 1（影响筛选）**
4) 估计 (\nabla J)（eval 策略梯度）
5) CG+HVP 得到 (v=H^{-1}\nabla J)（仅最后 K blocks+head）
6) 计算每条轨迹 Infl(i)，构建 D-/D0/D+
7) 分桶分层得到 `D_selected`
8) Train ACT@Selected（可加 D0 降权），评估总 SR + 困难桶 SR

**Round 2（再采集）**
9) 统计每桶 (n_b,\overline{\text{Infl}}_b,SR_b)，生成再采集配额表
10) 条件化 reset 再采集（优先高角度、小容差、边界多样性）
11) 合并数据，重复 Round 1

---

# 10. 关键参数建议（默认可用）

* eval seeds：150–300（困难分布占比 ≥50%）
* 影响函数子参数：最后 2–4 个 transformer blocks + action head
* 阻尼 (\lambda)：1e-2 起步
* CG 迭代：20–50
* influence 归一化：除以轨迹长度 (T_i)
* 负影响剔除：Infl<0 或 bottom 10–20%
* 主训练集保留：top 60–80%（再加分桶最低保留）


先用square_d1_core进行训练：
  act：
    random
    cupid
  openpi05：
    random
    cupid


失败轨迹分类：根据保存的rollout视频判断
  1.抓取失败
  2.抓取成功，搬运过程掉落
  3.插入时未对齐导致失败（在peg顶卡住 or 掉落）



失败轨迹分类：根据env返回的object—nut位置变化判断
  1.抓取失败：z几乎没变化
  2.抓取成功，搬运过程掉落：nut位置z变化，距离peg远的时候掉落
  3.插入时未对齐导致失败（在peg顶卡住 or 
  

抓取阶段：明确检测物体是否被抬起。
搬运阶段与插入阶段的划分：记录物体首次进入 Peg 上方区域（5cm范围内）的时刻。
基于运行趋势的掉落检测：不再只看单帧变化，而是对比抬起后的最大高度，这样能更稳健地识别出“慢慢滑落”或“中途坠落”。
更精准的分类：
  还没到 Peg 就掉了 -> drop_during_transport
  到了 Peg 附近才出问题 -> insertion_fail (包括卡在顶部 stuck_on_top 或在孔附近掉落)。

---

# 11. `eval_cupid_quality.py` 当前实现思路（仅依赖 LeRobot 数据集）

脚本路径：
`/workspace/liushuo/cupid/act/eval_cupid_quality.py`

## 11.1 设计目标

* 不依赖先验质量 CSV。
* 只读取 LeRobot 数据集 `data/chunk-*/episode_*.parquet` 里的轨迹字段进行评估。
* 统一产出 `quality_report.json`、`quality_episodes.csv`、`quality_drop.txt`、`quality_keep.txt`、`quality_top_50.txt`、`quality_bottom_50.txt`。
* 额外产出无监督一致性报告：`quality_consistency_report.json`。

## 11.2 三种质量源（`--quality-source`）

1. `episode_eval`
   * 关注运动学平滑性。
   * 主要特征：action jitter、state jitter、length consistency、(可选) action energy。
   * 适合筛动作抖动/轨迹粗糙样本。

2. `trajectory_eval`
   * 关注任务轨迹结构。
   * 主要特征：terminal consistency(kNN)、path efficiency、progress、monotonic/backtrack/stall、state/action smooth、length consistency。
   * 默认支持 gate（低 terminal/progress 时上限截断）。

3. `rank_fusion`
   * 先分别计算 `episode_eval` 和 `trajectory_eval` 分数。
   * 将两个分数各自转成百分位 rank（0~1）。
   * 再按权重线性融合：
     * `--rank-fusion-episode-weight`
     * `--rank-fusion-trajectory-weight`
   * 这样可降低不同指标量纲和长尾对融合的影响。

## 11.3 Drop/Keep 机制

* 量化删样：`--drop-quantile q`（删除最低 q 分位）。
* 绝对阈值删样：`--drop-threshold t`（删除 `score <= t`）。
* 两者并集作为 `drop` 集。

## 11.4 无监督一致性报告（`quality_consistency_report.json`）

当前包含：

* `score_distribution`：分布统计。
* `component_correlation`：融合分数与组件分数的 Spearman 相关。
* `bootstrap_drop_stability`：bootstrap 重采样后 drop 集 Jaccard 稳定性。
* `drop_bias_by_episode_block`：按 episode block（默认每 50 条）统计 drop 偏置。
* `terminal_mode_coverage`：基于终态聚类的保留/丢弃模式覆盖率。
* `score_pileup_top10`：分数堆积检查（是否大量样本卡同一分数）。

## 11.5 推荐用法

1. 先跑 `trajectory_eval` 作为任务结构基线：

```bash
python /workspace/liushuo/cupid/act/eval_cupid_quality.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --quality-source trajectory_eval \
  --drop-quantile 0.2 \
  --output-dir /workspace/liushuo/cupid/3_model/quality_eval/can_mh_traj_eval
```

2. 再跑 `rank_fusion`，并读取一致性报告做权重选择：

```bash
python /workspace/liushuo/cupid/act/eval_cupid_quality.py \
  --dataset.root /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --quality-source rank_fusion \
  --rank-fusion-episode-weight 0.3 \
  --rank-fusion-trajectory-weight 0.7 \
  --drop-quantile 0.2 \
  --output-dir /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion
```
