MimicGen 数据生成流程

**Overview**
本流程基于源码梳理，从源演示数据到自动生成新演示数据的完整链路。

**Entry Points**
- `mimicgen/scripts/prepare_src_dataset.py`：为源演示数据补充 `datagen_info`（一次性准备）
- `mimicgen/scripts/generate_dataset.py`：主生成脚本，循环采样并写出新数据集

**Process**
1. 准备源演示数据
2. 创建生成目录与环境
3. 加载并解析源数据为子任务段
4. 逐次生成新演示并写入临时文件
5. 合并输出为最终数据集

**Step Details**
1. 准备源演示数据
2. 使用 `prepare_src_dataset.py` 回放每条演示的状态序列，调用环境接口 `get_datagen_info`
3. 将 `eef_pose`、`object_poses`、`subtask_term_signals`、`target_pose`、`gripper_action` 写回源数据
4. 每条演示的 `datagen_info` 作为后续分段与生成的依据
5. 关键文件：`mimicgen/scripts/prepare_src_dataset.py`、`mimicgen/env_interfaces/base.py`

1. 创建生成目录与环境
2. `generate_dataset.py` 读取配置，创建输出目录与日志
3. 从源数据读取环境元信息并创建仿真环境
4. 构建 `MG_TaskSpec` 与 `DataGenerator`
5. 关键文件：`mimicgen/scripts/generate_dataset.py`、`mimicgen/configs/task_spec.py`

1. 加载并解析源数据为子任务段
2. `DataGenerator._load_dataset` 调用 `parse_source_dataset`
3. 根据 `subtask_term_signals` 检测子任务完成边界，并支持随机偏移 `subtask_term_offset_range`
4. 得到每条演示的子任务段索引与对应的 `DatagenInfo`
5. 关键文件：`mimicgen/datagen/data_generator.py`、`mimicgen/utils/file_utils.py`

1. 逐次生成新演示并写入临时文件
2. 每次尝试先 `env.reset` 采样新场景，然后按子任务顺序生成
3. 选择源段：`SelectionStrategy` 基于当前场景的对象位姿与源段信息选取参考演示
4. 变换源段：按对象位姿对齐，必要时加入旋转偏置与插值段
5. 执行轨迹：`WaypointTrajectory.execute` 逐步执行并记录 `states/obs/actions/datagen_info`
6. 成功写入 `tmp/*.hdf5`，失败可选写入 `tmp_failed/*.hdf5`
7. 关键文件：`mimicgen/datagen/data_generator.py`、`mimicgen/datagen/selection_strategy.py`、`mimicgen/datagen/waypoint.py`

1. 合并输出为最终数据集
2. 结束后将 `tmp` 目录下的单 episode 文件合并为 `demo.hdf5`
3. 失败样本可合并为 `demo_failed.hdf5`
4. 写出统计信息与日志
5. 关键文件：`mimicgen/scripts/generate_dataset.py`、`mimicgen/utils/file_utils.py`

**Key Mechanics**
- 子任务分段依赖 `datagen_info/subtask_term_signals` 中的 0→1 转换
- 源段选择由 `TaskSpec` 中 `selection_strategy` 控制，可按子任务或全程固定
- 姿态变换以对象位姿对齐为核心，通过 `transform_source_data_segment_using_object_pose`
- 执行轨迹由 `WaypointTrajectory` 完成，并可插值、加噪、分段旋转

**Outputs**
- 生成数据：`<exp_dir>/demo.hdf5`
- 失败数据（可选）：`<exp_dir>/demo_failed.hdf5`
- 日志与统计：`<exp_dir>/log.txt`、`<exp_dir>/logs/*.json`
