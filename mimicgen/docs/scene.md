MimicGen 现有场景与变体说明

本文汇总仓库中已有的 MimicGen 场景（robosuite / robosuite-task-zoo），并对主要变体差异做简要对比。内容基于 `mimicgen/envs/robosuite/*.py` 中的实现与类注释。

**场景总览**

| 任务 | 变体 | 简述 |
|---|---|---|
| Stack | D0, D1 | robosuite Stack 任务（堆叠方块） |
| StackThree | D0, D1 | 三块堆叠任务（在 StackThree 基础上扩展） |
| Square | D0, D1, D2 | NutAssemblySquare 任务（方形螺母装配） |
| NutAssembly | D0 | nut assembly 任务（方/圆螺母装配） |
| PickPlace | D0 | pick-place 任务的简化变体（限制 z 旋转范围） |
| Coffee | D0, D1, D2 | 咖啡任务（咖啡机与咖啡胶囊交互） |
| CoffeePreparation | D0, D1 | 更复杂的咖啡准备任务（含抽屉/杯子等） |
| Threading | D0, D1, D2 | threading 任务（针与环/支架） |
| ThreePieceAssembly | D0, D1, D2 | 三件装配任务 |
| HammerCleanup | D0, D1 | 锤子放入抽屉（HammerPlace 变体） |
| MugCleanup | D0, D1, O1, O2 | 物体清理任务（默认是杯子） |
| Kitchen | D0, D1 | kitchen 任务（来自 robosuite-task-zoo） |

---

**HammerCleanup D0 vs D1 对比**

| 维度 | D0 | D1 |
|---|---|---|
| 随机范围 | 仅锤子，小范围 | 锤子+抽屉，大范围 |
| 锤子范围 | `x(0.10~0.18), y(-0.20~-0.13), z_rot(-0.1~0.1)` | `x(-0.2~0.2), y(-0.25~-0.13), z_rot(0~2π)`，旋转轴改为 `y` |
| 抽屉范围 | 固定位置 | `x(0.0~0.2), y(0.2~0.3), z_rot(-π/6~π/6)` |
| 抽屉是否可动 | 固定（XML 中写死） | 可动（加入 placement initializer） |
| 成功判定 | 继承默认逻辑 | 更严格：锤子接触抽屉底部且抽屉关闭 |
| 难度倾向 | 更稳定 | 更难、泛化更强 |

---

**其他任务变体对比与描述**

**Stack**
Stack_D1 扩大初始化范围，并将默认相机切换为 `agentview_full` 以覆盖更大的桌面区域。

**StackThree**
StackThree_D1 扩大初始化范围，并调整默认相机以覆盖更大桌面范围。

**Square（NutAssemblySquare 变体）**
Square_D1：扩大螺母初始化范围，并固定 peg 位置与旋转；同时在 reset 时避免与 peg 碰撞。  
Square_D2：进一步扩大螺母与 peg 的范围，并对 peg 引入 z 旋转随机；相机切换为全桌面视角。

**NutAssembly**
NutAssembly_D0 为基础 nut assembly 任务，包含方形与圆形螺母，分别在不同区域初始化。

**PickPlace**
PickPlace_D0 限制物体 z 旋转范围为 `0~90°`，比标准任务更容易。

**Coffee**
Coffee_D1：咖啡机与咖啡胶囊的初始化范围更宽。  
Coffee_D2：与 D1 类似，但将胶囊放在左侧、咖啡机放在右侧，并将咖啡机更靠近机器人。

**CoffeePreparation**
CoffeePreparation_D1：扩大杯子初始化范围（右侧桌面+旋转），并允许咖啡机做一定平移/旋转。

**Threading**
Threading_D1：针与三脚架的初始化范围更宽，旋转范围更大。  
Threading_D2：将针放到左侧、三脚架放到右侧（位置与旋转区间相应调整）。

**ThreePieceAssembly**
ThreePieceAssembly_D1：底座与零件位置范围扩大，但旋转仍固定。  
ThreePieceAssembly_D2：位置与旋转范围均扩大（更高随机性）。

**MugCleanup**
MugCleanup_D1：抽屉与物体初始化范围更宽。  
MugCleanup_O1：使用不同的固定杯子模型。  
MugCleanup_O2：每次 reset 随机杯子模型。

**Kitchen**
Kitchen_D1：扩大对象的初始化分布，并允许部分原本固定的物体变为可动。
