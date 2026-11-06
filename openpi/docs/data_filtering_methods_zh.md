# OpenPI 当前数据筛选方法整理

本文档整理当前仓库里已经落地的数据筛选方法，重点覆盖：

- `episode` 级筛选
- `window` 级筛选
- `influence` / `quality` / `joint` 三类打分
- 筛选后 `norm_stats` 重算与重训练入口

适用范围以当前常用的 `pi05_can_mh_lora` 训练链路为主，但大部分方法也可迁移到其他 OpenPI 数据集配置。

## 1. 方法总览

| 方法 | 粒度 | 评分来源 | 代表脚本 | 是否重算 `norm_stats` | 是否直接训练 |
| --- | --- | --- | --- | --- | --- |
| 随机筛选 | episode / sample | 无，随机保留 | [`retrain_random.py`](/workspace/liushuo/cupid/openpi/retrain_random.py)、[`retrain_random_mask.py`](/workspace/liushuo/cupid/openpi/retrain_random_mask.py) | 是 | 是 |
| GraSS 影响分数筛选 | episode | `episode_influence_scores` | [`scripts/influence_pi05_lora_grass.py`](/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_grass.py) + [`retrain_grass.py`](/workspace/liushuo/cupid/openpi/retrain_grass.py) | 是 | 是 |
| GraSS 窗口聚合 | window / episode | `window_influence_scores`、`window_quality_scores` | [`scripts/influence_pi05_lora_grass_window.py`](/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_grass_window.py) | 否，仅打分 | 否 |
| Episode 级联合筛选 | episode | influence + external quality | [`retrain_joint.py`](/workspace/liushuo/cupid/openpi/retrain_joint.py) | 是 | 是 |
| Window 级联合筛选 | window | window influence + window quality | [`retrain_window_joint.py`](/workspace/liushuo/cupid/openpi/retrain_window_joint.py) | 是 | 是 |
| Leaf subset GraSS | sample / episode | 只在部分 leaves 上算 influence | [`scripts/influence_pi05_lora_grass_leaf_subset.py`](/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_grass_leaf_subset.py) | 否，仅打分 | 否 |

## 2. 随机筛选

入口脚本：

- [`retrain_random.py`](/workspace/liushuo/cupid/openpi/retrain_random.py)
- [`retrain_random_mask.py`](/workspace/liushuo/cupid/openpi/retrain_random_mask.py)

特点：

- 不依赖 influence 或 quality。
- [`retrain_random.py`](/workspace/liushuo/cupid/openpi/retrain_random.py) 保持现有逻辑，按 `episode` 随机筛选。
- [`retrain_random_mask.py`](/workspace/liushuo/cupid/openpi/retrain_random_mask.py) 新增按 `sample` / `state-action` 对随机 mask 的对照流程。
- 适合作为数据筛选实验的最基础对照组。

流程：

1. 随机选出要保留的数据。
2. `episode` 随机会重建过滤后的数据集；`sample` 随机 mask 会写出对比目录、保留样本索引和 mask 结果。
3. 重算 `norm_stats.json`。
4. 启动训练。

适用场景：

- 做对照实验。
- 验证“筛选本身”带来的影响，而不是某种评分函数带来的影响。

## 3. GraSS 影响分数筛选

入口脚本：

- 打分：[`scripts/influence_pi05_lora_grass.py`](/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_grass.py)
- 重训：[`retrain_grass.py`](/workspace/liushuo/cupid/openpi/retrain_grass.py)

### 3.1 打分思路

当前 GraSS 的核心公式是：

\[
s_i = - \langle H^{-1} g_{eval}, g^{train}_i \rangle
\]

其中：

- `g_train_i`：训练侧单个 `sample` 的梯度
- `g_eval`：由 eval rollout 回报加权得到的梯度
- `H^{-1}`：压缩空间中的 Hessian / eFIM 逆近似

实现特点：

- 只对 trainable 参数子集打分
- 对梯度做两阶段压缩
- 先得到 `sample` 级影响分数
- 再聚合成 `episode` 级影响分数

### 3.2 当前输出

主输出通常包括：

- `sample_influence_scores`
- `episode_influence_scores`
- `metadata.train_episode_indices`
- `metadata.train_sample_episode_indices`

### 3.3 重训练方式

[`retrain_grass.py`](/workspace/liushuo/cupid/openpi/retrain_grass.py) 会：

1. 按 `episode_influence_scores` 排序筛选 `episode`
2. 重建过滤后的数据集
3. 重算 `norm_stats.json`
4. 再训练

适用场景：

- 只想根据 influence 做筛选
- 需要一个纯 GraSS 的筛选基线

## 4. GraSS 窗口聚合

入口脚本：

- [`scripts/influence_pi05_lora_grass_window.py`](/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_grass_window.py)

### 4.1 作用

这个脚本在 sample 级 GraSS 的基础上，再按固定窗口长度聚合。

默认设计：

- `window_size = 50`
- 同一 `episode` 内分段
- 最后一个不足 `50` 的窗口也保留

### 4.2 当前输出

输出同时包含三层结果：

- `sample_influence_scores`
- `window_influence_scores`
- `window_quality_scores`
- `episode_influence_scores`
- `episode_quality_scores`

其中：

- `window_influence_scores` 默认是窗口内 sample influence 的聚合值
- `window_quality_scores` 是窗口内 influence 的质量化变体

同时会保存窗口到 sample 的映射：

- `metadata.train_window_indices`
- `metadata.train_window_episode_indices`
- `metadata.train_window_frame_starts`
- `metadata.train_window_frame_ends`
- `metadata.train_sample_window_indices`
- `metadata.train_dataset_sample_indices`

适用场景：

- 希望比 `episode` 更细地定位坏数据
- 不想因为局部坏段而整条 episode 删除

## 5. Episode 级联合筛选

入口脚本：

- [`retrain_joint.py`](/workspace/liushuo/cupid/openpi/retrain_joint.py)

### 5.1 打分公式

\[
\text{joint\_score}_i =
\text{quality}_i^{\alpha}
\cdot
\text{minmax}(\text{influence}_i)^{\beta}
\]

默认：

- `joint_quality_power = 1`
- `joint_influence_power = 1`

即：

\[
\text{joint\_score}_i =
\text{quality}_i
\cdot
\text{normalized\_influence}_i
\]

这里对 influence 做 `min-max` 归一化，是因为 influence 原始量纲通常很小，主要关心相对大小关系。

### 5.2 支持的输入来源

influence 可来自：

- [`scripts/influence_pi05_lora_grass.py`](/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_grass.py)
- [`scripts/influence_pi05_lora_grass_leaf_subset.py`](/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_grass_leaf_subset.py)

quality 可来自：

- 另一份 pickle 中的 `episode_quality_scores`
- 外部 CSV，例如：
  [`quality_episodes.csv`](/workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv)

### 5.3 训练流程

[`retrain_joint.py`](/workspace/liushuo/cupid/openpi/retrain_joint.py) 会：

1. 对齐 influence 和 quality 的 `episode id`
2. 计算 `joint_score`
3. 按 `episode` 筛选
4. 重建过滤后的数据集
5. 重算 `norm_stats.json`
6. 启动训练

配套串行 shell 脚本：

- [`scripts/retrain_joint_last4_csvq_serial.sh`](/workspace/liushuo/cupid/openpi/scripts/retrain_joint_last4_csvq_serial.sh)
- [`scripts/retrain_joint_grass_all_csvq_serial.sh`](/workspace/liushuo/cupid/openpi/scripts/retrain_joint_grass_all_csvq_serial.sh)

## 6. Window 级联合筛选

入口脚本：

- [`retrain_window_joint.py`](/workspace/liushuo/cupid/openpi/retrain_window_joint.py)
- 串行入口：[`scripts/retrain_window_joint_serial.sh`](/workspace/liushuo/cupid/openpi/scripts/retrain_window_joint_serial.sh)

### 6.1 设计目标

这是当前仓库里最细粒度、也最完整的一条链路：

- 输入 `window` 级 influence
- 再结合 `window` 级 quality
- 在 `window` 级联合筛选
- 最终回映射到 `sample`
- 重算 sample 子集上的 `norm_stats`
- 再训练

### 6.2 quality 的来源

这个脚本不依赖外部 CSV，而是直接参考
[`eval_cupid_quality.py`](/workspace/liushuo/cupid/act/eval_cupid_quality.py)
的设计，在 `window` 级重新计算质量分数。

目前支持三种 quality source：

- `episode_eval`
- `trajectory_eval`
- `rank_fusion`

### 6.3 评分逻辑

联合打分公式与 episode 级一致：

\[
\text{joint\_window\_score}_i =
\text{window\_quality}_i^{\alpha}
\cdot
\text{minmax}(\text{window\_influence}_i)^{\beta}
\]

### 6.4 筛选流程

[`retrain_window_joint.py`](/workspace/liushuo/cupid/openpi/retrain_window_joint.py) 会：

1. 从 window GraSS pickle 读取 `window_influence_scores`
2. 根据原始 parquet 数据计算 `window_quality_scores`
3. 计算 `joint_window_score`
4. 选择保留的 windows
5. 通过 `metadata.train_sample_window_indices` 和 `metadata.train_dataset_sample_indices` 回映射到 sample
6. 导出 `kept_sample_indices.npy`
7. 基于该 sample 子集重算 `norm_stats.json`
8. 使用 `data.sample_indices_path` 启动训练

### 6.5 输出产物

输出目录里通常会有：

- `kept_sample_indices.npy`
- `joint_window_selection_scores.npz`
- `joint_window_selection_summary.json`
- `window_quality_components.npz`
- `norm_stats.json`

适用场景：

- 需要只删除 episode 内部的局部低质量片段
- 需要更细粒度控制数据清洗强度

## 7. Leaf subset GraSS

入口脚本：

- [`scripts/influence_pi05_lora_grass_leaf_subset.py`](/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_grass_leaf_subset.py)

作用：

- 不在所有 trainable leaves 上打分
- 只在部分 LoRA / 末端 leaves 上计算 influence
- 作为 GraSS 的加速近似版本

支持两种子集方式：

- `--last-n-leaves N`
- `--leaf-name-regex REGEX`

当前定位：

- 更适合作为实验性加速入口
- 更适合做 influence 粗筛
- 目前不建议当作最稳定的正式主入口

## 8. 选择建议

如果目标是快速搭实验，建议这样选：

- 做基线：随机筛选
- 做纯 influence 筛选：GraSS + `retrain_grass.py`
- 做 episode 级联合筛选：`retrain_joint.py`
- 做更细粒度筛选：`retrain_window_joint.py`

如果目标是追求速度，建议这样选：

- 先用 leaf subset GraSS 做粗筛
- 再用全量 GraSS 或 window joint 做精筛

如果目标是正式做数据清洗实验，推荐优先级：

1. episode 级 joint
2. window 级 joint
3. random baseline

## 9. 自适应确定筛选比例

固定 `keep_ratio=0.8` 或 `drop_ratio=0.2` 是最简单的做法，但并不总是合理。更稳的思路是把“筛选比例”当作待估计对象，而不是预设常数。

### 9.1 为什么要自适应

不同数据集、不同打分方式下，低质量数据的占比并不固定：

- 有的分布里坏数据只占很小一部分
- 有的分布里低质量样本很多
- 同一个任务在 `episode` 级和 `window` 级的最佳删除比例也可能不同

因此，更合理的做法是先根据 `joint_score` 的分布形状估计候选比例，再做短训验证。

### 9.2 分布拐点驱动

把 `joint_score` 从低到高排序，寻找“低质量簇”和“正常簇”的分界位置。当前建议的三种近似方法是：

#### A. 排序曲线的 elbow / knee

思路：

- 将分数升序排列
- 画出 `sorted_index -> score` 曲线
- 找曲线从低分平缓段进入正常增长段的拐点

优点：

- 不依赖强分布假设
- 实现简单
- 对 `window` 级大样本分布通常比较稳

局限：

- 如果排序曲线没有明显弯折，结果会不稳定

#### B. 两高斯混合 / 双峰交点

思路：

- 用两个高斯成分拟合分数分布
- 一个成分对应低质量簇，一个对应正常簇
- 取两条密度曲线的交点作为阈值

优点：

- 当分布明显带有“双峰”结构时，可解释性很好

局限：

- 如果分布本身不是双峰，可能不稳定
- 对样本量和初始化更敏感

#### C. 累积均值 / 方差突变点

思路：

- 仍然先对分数排序
- 枚举切分点
- 比较切分点前后两段的均值差和方差变化
- 找“统计特性突然变化”的位置

优点：

- 比纯 GMM 更稳健
- 更适合没有明显双峰、但前后段质量差异明显的场景

局限：

- 需要人为定义“突变”的检测形式

### 9.3 推荐的使用方式

不建议直接把某个启发式给出的单一点比例当作最终训练比例。更稳的用法是：

1. 对 `joint_score` 做分布分析
2. 得到若干候选 `keep_ratio`
3. 在这些候选比例上做短训验证
4. 最终再选最优比例做完整训练

也就是说：

- 分布分析用于自动给出候选区间
- 验证集或短训结果用于最终定比例

### 9.4 当前分析脚本

入口脚本：

- [`scripts/analyze_filter_ratio.py`](/workspace/liushuo/cupid/openpi/scripts/analyze_filter_ratio.py)
- [`scripts/analyze_filter_ratio_serial.sh`](/workspace/liushuo/cupid/openpi/scripts/analyze_filter_ratio_serial.sh)

支持输入：

- 已经生成好的 [`joint_selection_scores.npz`](/workspace/liushuo/cupid/openpi/retrain_joint.py)
- 已经生成好的 [`joint_window_selection_scores.npz`](/workspace/liushuo/cupid/openpi/retrain_window_joint.py)
- 原始 `influence` pickle + 外部 quality CSV / quality pickle

默认分析字段：

- 对 `.npz` 输入，默认优先找 `joint_scores`
- 对 `pkl + quality` 输入，脚本会先在内部构造 `joint_scores`，再做分析

支持的启发式：

- `knee`
- `gmm2`
- `changepoint`

输出内容：

- 推荐 `keep_ratio`
- 推荐 `drop_ratio`
- 候选 `keep_ratio` 列表
- 三种方法各自的阈值与建议比例
- 可选排序曲线图和直方图

语义说明：

- `keep_ratio` 表示保留比例
- `drop_ratio` 表示删除比例
- 两者满足 `drop_ratio = 1 - keep_ratio`

示例：

```bash
uv run python scripts/analyze_filter_ratio.py \
  --score-npz /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/joint_selection_scores.npz \
  --plot
```

如果还没有 `.npz`，也可以直接传 `influence` 和 `quality`：

```bash
uv run python scripts/analyze_filter_ratio.py \
  --score-pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/influence_grass.pkl \
  --quality-score-csv /workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv \
  --episode-influence-field episode_influence_scores \
  --quality-csv-episode-column episode \
  --quality-csv-score-column quality_score \
  --plot
```

对于 `window` 级结果：

```bash
uv run python scripts/analyze_filter_ratio.py \
  --score-npz /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_window/joint_window_selection_scores.npz \
  --plot
```

如果要批量分析现有结果：

```bash
bash scripts/analyze_filter_ratio_serial.sh
```

如果要批量直接分析原始 `influence pkl + quality csv`：

```bash
SCORE_PKLS="/workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/influence_grass.pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/influence_grass.pkl" \
QUALITY_SCORE_CSV="/workspace/liushuo/cupid/3_model/quality_eval/can_mh_rank_fusion/quality_episodes.csv" \
bash scripts/analyze_filter_ratio_serial.sh
```

如果还要一并导出图：

```bash
PLOT=1 bash scripts/analyze_filter_ratio_serial.sh
```

### 9.5 实践建议

- `window` 级数据量更大，优先参考 `knee`
- `episode` 级样本较少，建议同时看 `knee + changepoint + GMM`
- 如果三种方法给出的比例接近，说明阈值比较可信
- 如果三种方法差异很大，说明分数分布不稳定，应优先做短训验证

### 9.6 如何解读分析结果

当前分析脚本会同时给出：

- `recommended_keep_ratio`
- `candidate_keep_ratios`
- `knee`
- `changepoint`
- `gmm2`

推荐这样解读：

#### 情况 A：三种方法结果接近

例如：

- `knee keep_ratio = 0.78`
- `changepoint keep_ratio = 0.81`
- `gmm2 keep_ratio = 0.76`

这种情况下，说明分数分布里确实存在相对清晰的分界区间。可以：

1. 直接把三者中位数作为默认比例
2. 再取其附近 2 到 3 个比例做短训验证

#### 情况 B：结果分成两簇

例如：

- `knee keep_ratio = 0.70`
- `changepoint keep_ratio = 0.68`
- `gmm2 keep_ratio = 0.88`

这种情况下，通常说明：

- 排序曲线上的结构和密度分布结构不完全一致
- 分数分布可能不是标准双峰

更稳的做法是：

1. 把两边都纳入候选集
2. 先在 `0.68/0.7/0.8/0.88` 这类点上做短训
3. 再决定最终比例

#### 情况 C：三种方法差异非常大

例如：

- `knee keep_ratio` 很小
- `changepoint keep_ratio` 接近 `0`
- `gmm2 keep_ratio` 却在 `0.7` 左右

这种情况通常说明：

- 分数分布不稳定
- 排序尾部可能存在极端值或长尾
- 不适合直接把某个单一启发式当成最终阈值

这时建议：

1. 先人工查看排序曲线和直方图
2. 优先把 `candidate_keep_ratios` 当成候选范围，而不是最终答案
3. 做 `prepare-only + 短训验证`
4. 如果是 `window` 级结果，还应检查是否有极少量异常窗口把分布拉坏

### 9.7 当前结果的使用建议

如果某次分析里：

- `knee`
- `changepoint`
- `gmm2`

给出的 `keep_ratio` 差异很大，那么当前更合适的用法是：

- 把这些比例当作候选区间
- 用短训来决定最终比例

而不是直接把 `recommended_keep_ratio` 当成最终训练比例。

也就是说：

- 分布分析负责“缩小搜索空间”
- 短训验证负责“最终定比例”

## 10. 当前相关脚本清单

### 评分脚本

- [`scripts/influence_pi05_lora_grass.py`](/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_grass.py)
- [`scripts/influence_pi05_lora_grass_window.py`](/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_grass_window.py)
- [`scripts/influence_pi05_lora_grass_leaf_subset.py`](/workspace/liushuo/cupid/openpi/scripts/influence_pi05_lora_grass_leaf_subset.py)

### 重训练脚本

- [`retrain_random.py`](/workspace/liushuo/cupid/openpi/retrain_random.py)
- [`retrain_grass.py`](/workspace/liushuo/cupid/openpi/retrain_grass.py)
- [`retrain_joint.py`](/workspace/liushuo/cupid/openpi/retrain_joint.py)
- [`retrain_window_joint.py`](/workspace/liushuo/cupid/openpi/retrain_window_joint.py)

### 串行 shell 脚本

- [`scripts/retrain_joint_last4_csvq_serial.sh`](/workspace/liushuo/cupid/openpi/scripts/retrain_joint_last4_csvq_serial.sh)
- [`scripts/retrain_joint_grass_all_csvq_serial.sh`](/workspace/liushuo/cupid/openpi/scripts/retrain_joint_grass_all_csvq_serial.sh)
- [`scripts/retrain_window_joint_serial.sh`](/workspace/liushuo/cupid/openpi/scripts/retrain_window_joint_serial.sh)
