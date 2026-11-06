MimicGen 分析总结

1. 任务场景

下表基于仓库 `docs/scene.md` 的梳理结果，覆盖你列出的 10 类任务与变体。

| 任务 | 变体 | 场景要点（简述） |
|---|---|---|
| CoffeePreparation | D0 / D1 | 更复杂的咖啡准备任务；D1 扩大杯子初始化范围（右侧桌面+旋转），并允许咖啡机一定平移/旋转。 |
| ThreePieceAssembly | D0 / D1 / D2 | 三件装配任务；D1 扩大底座与零件位置范围但旋转仍固定；D2 位置与旋转范围均扩大，随机性更高。 |
| Square（NutAssemblySquare） | D0 / D1 / D2 | 方形螺母装配；D1 扩大螺母初始化范围并固定 peg 位姿、避免碰撞；D2 进一步扩大范围并加入 peg 的 z 旋转随机，相机切换为全桌面视角。 |
| NutAssembly | D0 | 基础 nut assembly 任务，包含方形与圆形螺母并在不同区域初始化。 |
| Stack | D0 / D1 | 堆叠方块；D1 扩大初始化范围，并切换默认相机以覆盖更大桌面区域。 |
| StackThree | D0 / D1 | 三块堆叠；D1 扩大初始化范围并调整相机覆盖范围。 |
| HammerCleanup | D0 / D1 | 锤子放入抽屉；D1 同时扩大锤子与抽屉随机范围、抽屉可动、成功判定更严格，整体难度更高。 |
| MugCleanup | D0 / D1 / O1 / O2 | 物体清理任务（默认杯子）；D1 扩大抽屉与物体随机范围；O1 使用固定的不同杯子模型；O2 每次 reset 随机杯子模型。 |
| PickPlace | D0 | pick-place 简化变体，限制物体 z 旋转范围为 0~90 度。 |
| Kitchen | D0 / D1 | kitchen 任务；D1 扩大对象初始化分布，并允许部分原本固定的物体变为可动。 |

2. 数据生成方法

核心流程可归纳为 5 步（源数据 -> 分段 -> 选择 -> 变换 -> 写出），对应入口脚本为 `mimicgen/scripts/prepare_src_dataset.py` 与 `mimicgen/scripts/generate_dataset.py`。

1. 准备源演示数据
- 来源通常是人类遥操作演示，格式为 robomimic 兼容 HDF5。
- 使用 `prepare_src_dataset.py` 回放演示，通过环境接口写入 `datagen_info`（`eef_pose` / `object_poses` / `subtask_term_signals` 等）。

2. 构建任务配置与环境
- `generate_dataset.py` 读取配置，创建输出目录与环境。
- 通过 `MG_TaskSpec` 描述子任务序列与子任务边界信号。

3. 解析源数据为子任务段
- `DataGenerator._load_dataset` 读取 `subtask_term_signals`，将每条演示拆成子任务段。
- 支持通过 `subtask_term_offset_range` 对分段边界进行随机偏移。

4. 生成新轨迹
- 每次尝试先 `env.reset` 采样新场景。
- 用 `SelectionStrategy` 在源段中选择参考子任务段。
- 以对象位姿对齐为核心进行段变换，并用 `WaypointTrajectory` 执行轨迹。

5. 写出与合并
- 成功/失败轨迹先写到临时 HDF5，再合并为最终 `demo.hdf5`。
- 输出附带统计日志与可选回放视频。

3. 优缺点分析（基于代码与文档的工程视角）

优点
- 可扩展性强：只需少量源演示，通过子任务段重组可生成大量数据。
- 对任务结构友好：对象中心子任务建模让长任务可分而治之。
- 数据格式兼容：输出与 robomimic 兼容，训练链路直接复用。
- 可控性高：通过 `TaskSpec`、`SelectionStrategy` 与随机化范围可精细调整分布。
- 可复现性强：配置与脚本流程标准化，方便批量生成。

局限与风险点
- 依赖高质量 `datagen_info`：对象位姿与子任务终止信号若有偏差，分段与生成质量会显著下降。
- 对任务结构有假设：更适合“对象中心 + 清晰子任务边界”的操作；对强耦合、连续接触或非分段任务效果受限。
- 对仿真真实性敏感：接触动力学、摩擦、碰撞细节误差会放大生成失败率。
- 生成开销较高：保证一定成功率时需要反复尝试，时间成本与算力消耗可观。
- 变体分布需谨慎：初始化随机范围扩大虽有助泛化，但可能降低成功率或偏离真实分布。

如需更细的“任务级别生成效果”或“不同 selection 策略的成功率对比”，建议在现有日志基础上再统计生成成功率与失败原因。
