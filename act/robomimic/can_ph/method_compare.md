# can_ph 四种方法总结与对比

## 1. 方法总览

当前 `can_ph` 目录下对应 4 种训练策略：

1. `Random`：`can_ph_random.sh`
2. `Quality-only`：`can_ph_quality.sh`
3. `Influence-only`：`can_ph_influence.sh`
4. `Joint (Quality + Influence)`：`can_ph_joint.sh`

其中 `q in {0.1,0.2,0.3,0.4,0.5}` 可理解为删除比例（近似），对应保留率约为 `1 - q`。

### 1.1 各方法标准描述（可直接引用）

- `Random`：随机抽取一定比例的数据进行训练，不依赖质量分或影响分，是最基础的无先验筛选基线。
- `Quality-only`：仅使用轨迹质量分（`quality_score`）进行样本筛选，不使用 influence 信号，重点验证质量评估器本身的有效性。
- `Influence-only`：仅使用影响分数（`influence_score`）进行样本筛选与重加权，重点验证影响函数单一信号对训练效果的贡献。
- `Joint`：联合质量分与影响分进行样本筛选（`quality × influence`），并在训练阶段继续使用 influence 重加权，以兼顾数据可行性与任务敏感性。

## 2. 各方法实现思路

### 2.1 Random

- 核心思想：不使用任何质量分或影响分，直接随机子采样重训，作为基线。
- 实现脚本：`/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_random.sh`
- 实现方式：
  - 调用 `random_train.py`。
  - 用 `--sample-ratio` 控制保留比例（`1.0/0.9/0.8/0.7/0.6/0.5`）。
  - 用固定种子 `--sample-seed 42` 控制随机一致性。
- 关键参数位置：
  - 比例与种子：[can_ph_random.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_random.sh#L23), [can_ph_random.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_random.sh#L24), [can_ph_random.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_random.sh#L37)

### 2.2 Quality-only

- 核心思想：只按轨迹质量分过滤数据，不引入 influence。
- 实现脚本：`/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_quality.sh`
- 实现方式：
  - 调用 `train_act_cupid_reweight_joint.py`，但关闭 influence 分支。
  - 读取 `quality_episodes.csv` 作为质量源。
  - 按 `joint` 分位阈值做删除；由于 `joint_influence_power=0`，等价于仅按质量分删样本。
  - 采样时 `--weight-mode none`，不做 influence 重加权。
- 关键参数位置：
  - 质量文件：[can_ph_quality.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_quality.sh#L20), [can_ph_quality.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_quality.sh#L46)
  - 仅质量过滤：[can_ph_quality.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_quality.sh#L48), [can_ph_quality.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_quality.sh#L49), [can_ph_quality.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_quality.sh#L50), [can_ph_quality.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_quality.sh#L52)

### 2.2.1 Quality 的计算思路（当前 can_ph 配置）

- 当前 `can_ph` 使用 `quality_csv` 路线，不在训练时重新计算质量分，而是读取离线文件：
  - 来源文件：`quality_episodes.csv`（列：`episode, quality_score`）
  - 读取逻辑：按 episode 对齐；缺失或无效值用 `quality_score_default`（默认 0.5）补齐；最后裁剪到非负区间。
- 对应实现：
  - `quality_csv` 分支入口：[train_act_cupid_reweight_joint.py](/workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py#L1342)
  - CSV 解析与补默认值：[train_act_cupid_reweight_joint.py](/workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py#L1049)
- `quality_episodes.csv` 通常由 `eval_cupid_quality.py` 先离线生成；`can_ph_quality.sh` 注释里给的是 `rank_fusion` 方案示例（episode 与 trajectory 两种质量信号加权融合）。

### 2.3 Influence-only

- 核心思想：只用 influence 分做过滤与采样重加权，不使用质量分。
- 实现脚本：`/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_influence.sh`
- 实现方式：
  - 调用 `train_act_cupid_reweight.py`。
  - 输入 influence 结果 `score.pkl / score_last4.pkl / score_last8.pkl / score_last12.pkl`。
  - 按 `--drop-weight-quantile q` 删除低 influence 分位样本。
  - 用 `--weight-mode influence --use-weighted-sampler` 在训练中做 influence 重加权采样。
- 关键参数位置：
  - 多种 influence 输入：[can_ph_influence.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_influence.sh#L53)
  - influence 过滤+采样：[can_ph_influence.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_influence.sh#L92), [can_ph_influence.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_influence.sh#L93), [can_ph_influence.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_influence.sh#L94)

### 2.4 Joint（Quality + Influence）

- 核心思想：先把质量分和 influence 分联合打分过滤，再用 influence 做训练采样重加权。
- 实现脚本：`/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_joint.sh`
- 实现方式：
  - 调用 `train_act_cupid_reweight_joint.py`。
  - 同时输入 `quality_episodes.csv` 与 `score*.pkl`。
  - `--joint-quality-power 1.0` 与 `--joint-influence-power 1.0`，等权联合分。
  - 按 `--joint-drop-quantile q` 删除联合分低的样本。
  - 训练阶段 `--weight-mode influence --use-weighted-sampler`，继续保留 influence 重加权。
- 关键参数位置：
  - 双源输入：[can_ph_joint.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_joint.sh#L54), [can_ph_joint.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_joint.sh#L58)
  - 联合过滤+重采样：[can_ph_joint.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_joint.sh#L60), [can_ph_joint.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_joint.sh#L61), [can_ph_joint.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_joint.sh#L62), [can_ph_joint.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_joint.sh#L64), [can_ph_joint.sh](/workspace/liushuo/cupid/act/robomimic/can_ph/can_ph_joint.sh#L65)

### 2.4.1 联合打分公式（代码实现）

- 训练脚本中的联合分定义为：
  - `joint_score_i = (quality_i ^ a) * (norm(influence_i) ^ b)`
  - 其中：
    - `a = joint_quality_power`
    - `b = joint_influence_power`
    - `norm(x)` 是 min-max 归一化到 `[0,1]`：
      - `norm(x) = (x - min) / (max - min)`（当 `max == min` 时置为 1）
- 对应实现位置：
  - 联合公式函数：[train_act_cupid_reweight_joint.py](/workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py#L1089)
  - influence 的 `[0,1]` 归一化：[train_act_cupid_reweight_joint.py](/workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py#L574)
- `filter_score_mode=joint` 时的过滤规则：
  - 先做 bad-ratio 过滤：`ep_bad < drop_bad_ratio`
  - 再按 `joint_score` 的 `joint_drop_quantile=q` 分位阈值删除低分样本
  - 对应实现：[train_act_cupid_reweight_joint.py](/workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py#L1353), [train_act_cupid_reweight_joint.py](/workspace/liushuo/cupid/act/train_act_cupid_reweight_joint.py#L1360)

## 3. 四种方法对比

| 方法 | 筛选依据 | 训练采样权重 | 依赖的预计算产物 | 工程开销（相对） | 主要用途 |
| --- | --- | --- | --- | --- | --- |
| Random | 随机比例 `sample-ratio` | 无 | 无 | 最低 | 基线对照 |
| Quality-only | `quality_score` 分位过滤 | 无（`weight-mode none`） | `quality_episodes.csv` | 低-中 | 验证质量评估器本身 |
| Influence-only | `influence` 分位过滤 | influence 加权采样 | `score*.pkl` | 中-高 | 验证 influence 单信号有效性 |
| Joint | `quality × influence` 联合分位过滤 | influence 加权采样 | `quality_episodes.csv` + `score*.pkl` | 最高 | 同时利用质量与任务敏感性 |

## 4. 建议的使用顺序

1. 先跑 `Random` 确认训练管线和基线性能。
2. 跑 `Quality-only` 与 `Influence-only` 做单信号归因。
3. 最后跑 `Joint` 做主方案，并对比 `score / last4 / last8 / last12` 不同 influence 作用域。
