# Window + Episode Prior 分层筛选说明

本文档说明脚本：

- [`retrain_window_joint_episode_prior.py`](/workspace/liushuo/cupid/openpi/retrain_window_joint_episode_prior.py)
- [`retrain_window_joint_episode_prior_gain_coverage.py`](/workspace/liushuo/cupid/openpi/retrain_window_joint_episode_prior_gain_coverage.py)
- [`scripts/retrain_window_joint_episode_prior_gaincov_last4_serial.sh`](/workspace/liushuo/cupid/openpi/scripts/retrain_window_joint_episode_prior_gaincov_last4_serial.sh)
- [`scripts/retrain_window_joint_episode_prior_gaincov_all_serial.sh`](/workspace/liushuo/cupid/openpi/scripts/retrain_window_joint_episode_prior_gaincov_all_serial.sh)

它在现有 `window` 级联合筛选基础上，再给每个 `window` 加一个 `episode` 级先验分数。

另外现在也支持“自适应保留比例”版本：

- 不再手动枚举 `keep_ratio`
- 通过 `gain + coverage` 启发式自动决定保留比例
- 可额外用 `marginal_gain_threshold` 约束边际收益下降
- 可以像 `act/myscripts/run_filter_and_train_joint_episode_window.sh` 一样直接走“筛选 + 重训练”链路

如果你要复现固定比例筛选，用 `retrain_window_joint_episode_prior.py`。
如果你要自动决定保留比例，用 `retrain_window_joint_episode_prior_gain_coverage.py` 或上面的 launcher。
如果你要用“全部参数”的 influence score，而不是 `last4`，优先用 `retrain_window_joint_episode_prior_gaincov_all_serial.sh`。

## 1. 方法动机

纯 `window` 级筛选的优点是粒度细，不会因为某一小段坏片段就丢掉整条 `episode`。

但它也有一个常见问题：

- 某个 `window` 局部看起来很好
- 但它所在的整条 `episode` 其实整体很差

例如：

- 任务后续失败
- 轨迹整体不稳定
- 大部分片段都不好，只是局部有一段碰巧分高

如果只看 `window` 分数，这类局部偶然高分片段可能会被错误保留。

相反，如果直接做硬门控：

```text
只有 episode_score > tau_e 的 episode 中的 windows 才允许保留
```

又容易过于激进：

- 只要 `episode` 整体分数略低
- 其中所有 `window` 都被连坐删除

因此，这个脚本采用一个更平滑的折中方案：

- `window_joint_score` 负责局部证据
- `episode_joint_score` 负责全局先验
- 最终用加权加法融合，而不是硬阈值门控
- 并支持 3 个增量扩展：
  - 自适应 `episode_prior_weight`
  - `window * episode` 交互项
  - episode-aware 预算式保留

## 2. 评分公式

### 2.1 Window joint score

先计算每个 `window` 的联合分数：

```text
window_joint_score =
    combine(window_quality_score, normalized_window_influence)
```

其中 `combine` 由 `--joint-rule` 决定：

- `sum`
- `product`

更具体地：

```text
quality_term   = signed_power(window_quality_score, joint_quality_power)
influence_term = normalized_window_influence ^ joint_influence_power
```

如果：

- `joint-rule = sum`

则：

```text
window_joint_score = quality_term + influence_term
```

如果：

- `joint-rule = product`

则：

```text
window_joint_score = quality_term * influence_term
```

### 2.2 Episode joint score

再计算每个 `episode` 的先验分数：

```text
episode_joint_score =
    combine(episode_quality_score, normalized_episode_influence)
```

其中：

- `episode_quality_score` 默认由该 `episode` 内所有 `window_quality_score` 聚合得到
- `episode_influence_score` 来自 score pickle 中已有的 `episode_influence_scores`

### 2.3 最终 window score

最终用于筛选的分数不是原始 `window_joint_score`，而是：

```text
final_window_score =
    window_score_weight      * normalize(window_joint_score)
  + adaptive_prior_weight_i * normalize(episode_joint_score_of_its_episode)
  + interaction_weight      * normalize(window_joint_score) * normalize(episode_joint_score_of_its_episode)
```

这里 `normalize(...)` 由 `--fusion-normalizer` 决定，默认用 `minmax`。

其中：

- `adaptive_prior_weight_i` 在默认情况下退化为常数 `episode_prior_weight`
- 当启用自适应 prior 时，它会根据该 `window` 所属 episode 的统计量动态变化

### 2.4 预算式保留

默认仍然是全局 top-k。

但现在也支持 episode-aware 预算式保留：

- 每条 `episode` 至少保留若干 `window`
- 剩余名额按 episode prior 或均匀方式分配
- 然后在每条 `episode` 内再按 `final_window_score` 取 top-k

这能减少少数高分 episode 把所有保留名额吃掉的问题。

## 3. 可解释性

这个方法的可解释性比“把所有信息揉成一个黑盒分数”更强，因为它是分层建模的。

### 3.1 局部证据 + 全局先验

可以把它理解成：

- `window_joint_score` 回答：
  这个局部片段本身好不好？

- `episode_joint_score` 回答：
  这个片段所在的整条轨迹整体值不值得信任？

所以最终分数表达的是：

```text
这个 window 本身值不值得留，
并且它来自怎样的一条 episode。
```

### 3.2 为什么不是硬门控

如果做硬门控：

```text
episode 差 -> 整条 episode 的所有 windows 直接淘汰
```

可解释性虽然也强，但行为太极端。

当前脚本的加权融合更容易解释成：

- `episode` 差，不是一票否决
- 只是降低该 `episode` 中所有 `window` 的优先级

这比“直接判死刑”更符合“先验”这个概念。

### 3.3 为什么比纯 window 更稳

`window` 粒度更细，天然噪声更大：

- 短窗口方差更高
- 局部偶然高分更多
- 排名更容易被局部极值主导

加入 `episode` 先验后，相当于做了一次层级平滑：

- 好 `episode` 里的好 `window` 会更稳定地靠前
- 差 `episode` 里的偶然高分 `window` 会被适度压低

这使得最终排序更符合“局部表现 + 全局可信度”的直觉。

### 3.4 新增三种扩展的可解释性

#### 自适应 prior

它不是改变“prior 是什么”，而是改变“什么时候更该相信 prior”。

因此可解释性仍然很强：

- 局部窗口越不稳定，prior 权重越大
- 局部窗口越稳定，prior 权重越小

#### 交互项

交互项表达的是：

```text
局部好 且 全局也好
```

时的额外奖励。

所以三项可以明确解释为：

- `window`：局部证据
- `episode`：全局先验
- `window * episode`：两者同时高时的协同效应

#### 预算式保留

它不是改分数，而是改“名额怎么分”。

解释上很直接：

- 先保证 episode 覆盖
- 再在 episode 内保留最优 window

这对小数据场景尤其重要，因为你往往不希望几个高分 episode 完全垄断训练集。

## 4. 超参数说明

下面按重要性排序说明。

### 4.1 `--window-score-weight`

含义：

- 局部 `window` 分数的权重

默认：

- `0.8`

作用：

- 越大，越相信局部窗口本身
- 越小，越不容易让局部偶然高分主导结果

调参直觉：

- 如果你更关心局部坏段定位，用大一些，例如 `0.8 ~ 0.9`
- 如果你觉得 `window` 噪声很大，可以降到 `0.6 ~ 0.7`

### 4.2 `--episode-prior-weight`

含义：

- `episode` 全局先验的权重

默认：

- `0.2`

作用：

- 越大，越强调“这个 window 来自哪条 episode”
- 越小，越接近纯 `window` 筛选

调参直觉：

- 初次实验建议 `0.1 ~ 0.3`
- 如果你发现很多差 `episode` 中的局部高分 window 被保留，可以增大到 `0.3 ~ 0.4`
- 不建议一开始就太大，否则会退化成“软版 episode 门控”

### 4.3 `--adaptive-prior-mode`

含义：

- 如何让 `episode` prior 的权重随数据特征变化

可选：

- `none`
- `window_std`
- `inverse_episode_length`
- `quality_std`
- `hybrid`

解释：

- `window_std`：一个 episode 内 window 分数波动越大，prior 越强
- `inverse_episode_length`：episode 越短，prior 越强
- `quality_std`：一个 episode 内 quality 分布越散，prior 越强
- `hybrid`：上面几种信号平均

建议：

- 第一轮先用 `none`
- 如果你确认局部排序噪声偏大，优先试 `window_std`
- 如果数据里短 episode 特别不稳定，可以试 `inverse_episode_length`

### 4.4 `--adaptive-prior-strength`

含义：

- 自适应 prior 的调节强度

默认：

- `0.5`

作用：

- `0` 表示完全关闭自适应
- 越接近 `1`，不同 episode 的 prior 权重差异越明显

建议：

- 起步试 `0.3 ~ 0.5`
- 不建议一开始就用 `1.0`

### 4.5 `--interaction-weight`

含义：

- `window * episode` 交互项的权重

默认：

- `0.0`

作用：

- 越大，越奖励“局部高分且全局高分”的 window
- `0` 时退化回原始线性融合

建议：

- 第一轮试 `0.0 / 0.1 / 0.2`
- 如果担心分数过于激进，不要太大

### 4.6 `--fusion-normalizer`

含义：

- 在做最终加权前，如何把 `window_joint_score` 和 `episode_joint_score` 归一到可比较尺度

可选：

- `minmax`
- `percentile`
- `none`

默认：

- `minmax`

建议：

- 默认先用 `minmax`
- 如果担心长尾或极端值影响，可以试 `percentile`
- 除非你非常清楚两个分数天然同尺度，否则不建议用 `none`

### 4.7 `--episode-quality-aggregation`

含义：

- 如何从一个 `episode` 内的多个 `window_quality_score` 得到 `episode_quality_score`

可选：

- `mean`
- `max`

默认：

- `mean`

解释：

- `mean`：更像“整条 episode 的平均质量”
- `max`：更像“这条 episode 至少出现过一个高质量片段”

建议：

- 默认用 `mean`
- `max` 更激进地偏向“亮点片段”，一般不建议作为默认值

### 4.8 `--min-windows-per-episode`

含义：

- 每条 `episode` 至少保留多少个 `window`

默认：

- `0`

作用：

- 大于 `0` 时启用最低覆盖保障
- 能减少高分 episode 吞掉全部保留窗口的情况

建议：

- 小数据场景可试 `1` 或 `2`
- 如果 `keep_ratio` 很低，不要设太大，否则预算会不够

### 4.9 `--episode-budget-rule`

含义：

- 除了最低保留名额外，剩余保留预算如何在 episode 间分配

可选：

- `global`
- `uniform_episode`
- `episode_prior`

解释：

- `global`：保底后，其余仍按全局分数竞争
- `uniform_episode`：剩余预算尽量均匀分配到各个 episode
- `episode_prior`：episode 分数越高，分到的预算越多

建议：

- 默认用 `global`
- 如果你更关心覆盖面，试 `uniform_episode`
- 如果你既关心覆盖又希望偏向好 episode，试 `episode_prior`

### 4.10 `--joint-rule`

含义：

- `quality` 和 `influence` 在每个层级内如何组合

可选：

- `sum`
- `product`

默认：

- `sum`

解释：

- `sum`：更稳，允许一定程度互补
- `product`：更严格，任一项低都会明显压低总分

建议：

- 初次实验优先 `sum`
- 如果你想做更强筛选，再试 `product`

### 4.11 `--joint-quality-power`

含义：

- 质量分数的幂指数

默认：

- `1.0`

作用：

- 大于 `1`：放大高低质量差异
- 小于 `1`：压缩质量分数差异

建议：

- 第一轮先固定 `1`
- 只有在确认质量分数很可靠时，才考虑调大

### 4.12 `--joint-influence-power`

含义：

- influence 分数的幂指数

默认：

- `1.0`

作用：

- 大于 `1`：更强调高 influence 样本
- 小于 `1`：更平滑 influence 差异

建议：

- 第一轮先固定 `1`
- 如果发现 influence 主导太强，可以尝试 `0.5 ~ 0.8`

## 5. 推荐起始配置

### 5.1 稳健起点

```text
joint-rule = sum
window-score-weight = 0.8
episode-prior-weight = 0.2
adaptive-prior-mode = none
interaction-weight = 0
fusion-normalizer = minmax
episode-quality-aggregation = mean
min-windows-per-episode = 0
episode-budget-rule = global
joint-quality-power = 1
joint-influence-power = 1
```

这组配置更像：

- 以 `window` 为主
- 用 `episode` 做轻量先验修正

### 5.2 更保守

```text
window-score-weight = 0.7
episode-prior-weight = 0.3
```

适合：

- 你怀疑局部 `window` 分数噪声偏大
- 希望多参考整条 episode 的整体质量

### 5.3 更激进

```text
joint-rule = product
window-score-weight = 0.8
episode-prior-weight = 0.2
```

适合：

- 你确认 `quality` 和 `influence` 都比较可靠
- 希望更强地压低“局部好但整体差”的片段

## 6. 使用示例

```bash
uv run python retrain_window_joint_episode_prior.py \
  --score-pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/influence_grass_window.pkl \
  --train-config pi05_can_mh_lora \
  --dataset-dir /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --keep-ratio 0.5 \
  --joint-rule sum \
  --window-score-weight 0.8 \
  --episode-prior-weight 0.2 \
  --adaptive-prior-mode none \
  --interaction-weight 0.0 \
  --fusion-normalizer minmax \
  --episode-quality-aggregation mean \
  --min-windows-per-episode 0 \
  --episode-budget-rule global \
  --quality-source rank_fusion \
  --prepare-only \
  --overwrite
```

启用三项扩展的示例：

```bash
uv run python retrain_window_joint_episode_prior.py \
  --score-pkl /workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/influence_grass_window.pkl \
  --train-config pi05_can_mh_lora \
  --dataset-dir /workspace/liushuo/cupid/datasets/lerobot/can_mh \
  --keep-ratio 0.5 \
  --joint-rule sum \
  --window-score-weight 0.8 \
  --episode-prior-weight 0.2 \
  --adaptive-prior-mode window_std \
  --adaptive-prior-strength 0.5 \
  --interaction-weight 0.1 \
  --fusion-normalizer minmax \
  --episode-quality-aggregation mean \
  --min-windows-per-episode 1 \
  --episode-budget-rule episode_prior \
  --quality-source rank_fusion \
  --prepare-only \
  --overwrite
```

## 7. 输出产物

输出目录下会包含：

- `kept_sample_indices.npy`
- `joint_window_episode_prior_scores.npz`
- `joint_window_episode_prior_summary.json`
- `window_quality_components.npz`
- `norm_stats.json`

其中最重要的是：

- `window_joint_scores`
- `episode_joint_scores`
- `interaction_scores`
- `adaptive_prior_weight_per_window`
- `final_window_scores`

你可以直接比较这三者的分布，判断：

- 先验是否过强
- 局部分数是否仍然主导排序
- 最终筛选是否过于激进

## 8. 什么时候适合用这个方法

适合：

- 你觉得纯 `window` 排序太噪
- 你又不想直接做 `episode` 硬门控
- 你希望在保留局部细粒度筛选能力的同时，引入整条轨迹的全局上下文

不太适合：

- 数据量极小，`episode` 先验本身很不稳定
- `episode_influence_scores` 质量很差
- 你已经确认纯 `window` 筛选足够稳定

## 9. 实践建议

- 先固定 `joint-rule = sum`
- 先用 `window-score-weight=0.8`、`episode-prior-weight=0.2`
- 再只调 `keep_ratio`

如果效果不稳定，再调：

- `episode_prior_weight`
- `adaptive_prior_mode`
- `interaction_weight`
- `fusion_normalizer`

不要一开始同时改：

- `keep_ratio`
- `joint-rule`
- `power`
- `fusion weights`

否则很难判断究竟是哪一项带来了变化。

## 10. 如何选择超参数

推荐按“先定结构，再调强度”的顺序来选。

### 10.1 第一步：先定联合结构

先固定下面这组作为默认起点：

```text
joint-rule = sum
window-score-weight = 0.8
episode-prior-weight = 0.2
adaptive-prior-mode = none
adaptive-prior-strength = 0.5
interaction-weight = 0.0
fusion-normalizer = minmax
episode-quality-aggregation = mean
min-windows-per-episode = 0
episode-budget-rule = global
joint-quality-power = 1.0
joint-influence-power = 1.0
```

理由是：

- `sum` 比 `product` 更稳，不容易因为单项低分把总分压得过狠
- `0.8 / 0.2` 表示以 `window` 为主、`episode` 为辅，符合“局部证据 + 全局先验”的设计意图
- `mean` 更像整条 `episode` 的整体质量，而不是只看亮点片段
- 先关闭 adaptive/interaction/budget，便于看清基线效果

### 10.2 第二步：先只扫 `keep_ratio`

在默认结构不动的情况下，先比较不同保留比例。

建议第一轮扫：

```text
keep_ratio = 0.3 / 0.5 / 0.7
```

如果数据量更大，或你已经知道筛选会比较激进，再加：

```text
keep_ratio = 0.2 / 0.4 / 0.6 / 0.8
```

选择标准不是看分数分布，而是看短训后的验证表现：

- 成功率
- 任务完成率
- 轨迹稳定性
- 收敛速度

### 10.3 第三步：如果排序噪声大，再调先验权重

当你观察到：

- 低质量 `episode` 中仍有很多局部高分 `window` 被保留
- 纯 `window` 排名波动很大

再调：

```text
episode-prior-weight = 0.1 / 0.2 / 0.3 / 0.4
window-score-weight = 0.9 / 0.8 / 0.7 / 0.6
```

调参原则：

- `episode-prior-weight` 越大，越强调全局可信度
- `window-score-weight` 越大，越强调局部可用性

实操上建议保持两者同量级、且 `window` 略大于 `episode`。

### 10.4 第四步：如果联合分数过于激进，再调组合方式

如果你发现：

- 推荐保留比例特别低
- 大量样本被压到低分区
- 训练只在很小子集上有效

优先顺序应是：

1. 保持 `joint-rule = sum`
2. 降低 `episode-prior-weight`
3. 降低 `joint-influence-power`
4. 最后才考虑改 `fusion-normalizer`

不要一开始就换成 `product`，因为那通常会进一步放大激进程度。

### 10.5 第五步：只有在分数可靠时，再调 power

`joint-quality-power` 和 `joint-influence-power` 的作用是“拉开差距”或“压缩差距”。

推荐范围：

```text
joint-quality-power = 0.8 / 1.0 / 1.2
joint-influence-power = 0.5 / 0.8 / 1.0 / 1.2
```

经验判断：

- 如果质量分数很稳定，可以试 `joint-quality-power > 1`
- 如果 influence 太容易主导排序，可以试 `joint-influence-power < 1`
- 如果两个分数本身就不稳定，保持 `1.0` 最稳

## 11. 可以如何调参数

最实用的方式是做“分阶段、小网格”而不是一次性全量搜索。

### 11.1 推荐调参顺序

第一轮：

- 只调 `keep_ratio`

第二轮：

- 固定较优 `keep_ratio`
- 调 `episode-prior-weight`
- 如有需要，再开 `adaptive-prior-mode`

第三轮：

- 只有在前两轮已经稳定后
- 再调 `interaction-weight`、`joint-rule`、`power`、`fusion-normalizer`

### 11.2 推荐的小网格

稳健小网格：

```text
keep_ratio:            0.3 / 0.5 / 0.7
episode-prior-weight:  0.1 / 0.2 / 0.3
adaptive-prior-mode:   none / window_std
interaction-weight:    0.0 / 0.1
joint-rule:            sum
```

更细一点：

```text
keep_ratio:              0.3 / 0.5
window-score-weight:     0.8 / 0.7
episode-prior-weight:    0.2 / 0.3
fusion-normalizer:       minmax / percentile
joint-influence-power:   0.8 / 1.0
```

### 11.3 根据现象反推参数

如果出现下面现象，可以这样调：

- 现象：局部偶然高分太多
  调法：增大 `episode-prior-weight`

- 现象：局部高分只在高质量 episode 里才更可信
  调法：增大 `interaction-weight`

- 现象：很多整条很差的 episode 仍贡献大量 window
  调法：增大 `episode-prior-weight`，并保持 `joint-rule = sum`

- 现象：保留窗口过于集中在少数 episode
  调法：设置 `min-windows-per-episode=1`，并试 `episode-budget-rule=episode_prior`

- 现象：排序过于激进，保留比例一低性能就崩
  调法：减小 `episode-prior-weight`，减小 `joint-influence-power`

- 现象：局部坏段删不掉，筛选不够锋利
  调法：增大 `window-score-weight`，必要时试 `joint-rule = product`

- 现象：最终排序受极端值影响明显
  调法：试 `fusion-normalizer = percentile`

### 11.4 一个推荐实验流程

可以按下面顺序跑：

```text
阶段 1:
  keep_ratio = 0.3 / 0.5 / 0.7
  其他全固定

阶段 2:
  固定最佳 keep_ratio
  episode-prior-weight = 0.1 / 0.2 / 0.3

阶段 3:
  固定最佳 keep_ratio + prior weight
  adaptive-prior-mode = none / window_std
  interaction-weight = 0.0 / 0.1

阶段 4:
  固定最佳 keep_ratio + prior weight + adaptive/interaction
  min-windows-per-episode = 0 / 1
  episode-budget-rule = global / episode_prior

阶段 5:
  只有前面都稳定后
  fusion-normalizer = minmax / percentile
  joint-influence-power = 0.8 / 1.0
  joint-rule = sum / product
```

这样能保证每一步都有明确解释，不会把多个因素混在一起。

## 12. 可扩展创新点

当前脚本已经内置了 3 个增量扩展：

- 自适应 `episode_prior_weight`
- 交互项融合
- 预算式保留

在这基础上，仍然还有进一步扩展空间。

### 12.1 学习型融合权重

当前是手工设：

```text
final = a * window + b * episode
```

可以扩展成：

- 不同任务使用不同 `a, b`
- 不同 episode 长度使用不同 `a, b`
- 甚至学习一个小模型来预测 `a, b`

这样能让“先验强度”随数据特征自适应变化。

### 12.2 非线性先验融合

现在是线性加法，可以扩展为：

- `window + prior + interaction`
- 门控网络
- 分段函数

例如：

```text
final = window + b * prior + c * window * prior
```

这样可以显式建模“局部好且全局也好”时的额外奖励。

### 12.3 位置相关先验

当前同一条 `episode` 内所有 `window` 共享同一个 episode prior。

可以改进为：

- 前段、中段、后段使用不同先验
- 靠近 terminal 的 window 使用更强先验
- 对失败后段做额外惩罚

这会让全局先验更符合时序结构。

### 12.4 更丰富的 episode 质量建模

现在 `episode_quality_score` 主要是由 `window_quality_score` 聚合得到。

可以加入更多 episode 级信号：

- 成功/失败标签
- reward 或 return
- 末态误差
- 安全约束违例
- 人工规则分数

这样 `episode` prior 的语义会更强，不只是“局部质量的平均值”。

### 12.5 不确定性建模

当前所有分数都被当作确定值使用。

可以扩展为：

- 给 `window` 分数配一个置信度
- 给 `episode` prior 配一个置信度
- 低置信度时减小该项权重

这对小数据、噪声标签和分数漂移特别有用。

## 13. 可以改进的方向

### 13.1 先解决“可解释调参”，再做复杂模型

最优先的改进不是上更复杂模型，而是补齐分析工具，例如：

- 画 `window_joint_score`、`episode_joint_score`、`final_window_score` 三者的分布
- 统计不同 `episode` 下窗口得分的抬升/压低幅度
- 输出被先验改变排序最明显的 top-k windows

这样调参会明显更可控。

### 13.2 做分层 ablation

建议系统比较：

- 纯 `window`
- 纯 `episode`
- `window + episode prior`
- `episode hard gate + window rerank`

这样可以验证当前方案真正赢在什么地方：

- 是更稳
- 是更准
- 还是只是更保守

### 13.3 引入自适应权重

可以让 `episode_prior_weight` 不再是常数，而是：

- 与 `window` 方差相关
- 与 episode 长度相关
- 与 quality 分布偏度相关

例如：

```text
window 越不稳定 -> prior 权重越大
window 越稳定   -> prior 权重越小
```

这比固定 `0.2`、`0.3` 更自然。

### 13.4 从排序走向预算分配

当前是统一按 `keep_ratio` 截断。

可以改成：

- 每个 episode 至少保留若干 window
- 每个 episode 按 prior 分配保留预算
- 长 episode 和短 episode 使用不同预算策略

这样能减少少数高分 episode 垄断全部保留名额的问题。

### 13.5 与训练反馈闭环

目前筛选和训练是分开的。

后续可以做成闭环：

- 用训练 early metrics 回写筛选参数
- 动态更新 keep ratio
- 根据验证结果自动调整 prior weight

这是最有潜力、但也最复杂的改进方向。
