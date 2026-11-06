MimicGen 数据生成详细分析报告

本文基于当前仓库实现，对 MimicGen 的数据生成流程、关键模块、核心算法与产出结构进行系统性说明，适合非 Python 背景快速理解“代码在做什么”。

**一、整体流程概览**
1. 采集源人类演示（robomimic 兼容 hdf5）
2. 为源数据补齐 `datagen_info`
3. 解析子任务并随机化边界
4. 选择源演示子段
5. 基于对象位姿对齐末端轨迹
6. 构造可执行轨迹并执行
7. 写出成功/失败轨迹并合并为最终数据集

**二、关键脚本与职责**
- `mimicgen/scripts/prepare_src_dataset.py`：回放源演示并写入 `datagen_info`
- `mimicgen/scripts/generate_dataset.py`：主生成脚本，创建环境、循环生成并写出数据
- `mimicgen/datagen/data_generator.py`：生成核心逻辑
- `mimicgen/utils/file_utils.py`：源数据解析与 hdf5 写出
- `mimicgen/utils/pose_utils.py`：位姿变换与对齐
- `mimicgen/datagen/waypoint.py`：轨迹构造与执行

**三、源数据补齐（prepare_src_dataset）**
1. 读取源 hdf5 与环境元信息
2. 创建数据回放环境（只用状态，不采图像）
3. 调用环境接口逐步回放，提取 `DatagenInfo`
4. 将以下字段写回 `data/<demo>/datagen_info`：
1. `eef_pose`
2. `object_poses/<obj_name>`
3. `subtask_term_signals/<signal_name>`
4. `target_pose`
5. `gripper_action`
5. 写入属性 `env_interface_name` 与 `env_interface_type`

目的：让源演示具备“可切分、可对齐”的结构化信息。

**四、子任务切分与边界随机化**
- 子任务终止信号来自 `datagen_info/subtask_term_signals`
- `parse_source_dataset` 查找 0→1 的终止点作为分段边界
- `randomize_subtask_boundaries` 在子任务结束点上加入随机偏移

效果：同一演示可以产生不同子段切分，提高多样性。

**五、源演示子段选择策略**
每个子任务需要选一个源演示子段作为参考：
- 选择策略由 `task_spec` 指定
- 常见策略：随机、对象位姿最近邻
- 是否每个子任务都重新选由 `select_src_per_subtask` 控制

入口：`DataGenerator.select_source_demo`

**六、位姿对齐的核心原理**
函数：`transform_source_data_segment_using_object_pose`

思路：保持“末端相对物体”的位姿关系不变  
公式（语义）：  
1. 把源末端位姿从世界坐标系转到“源物体坐标系”  
2. 再把相对位姿应用到“当前物体坐标系”  

效果：同样的动作模式被迁移到新场景物体上。

**七、轨迹构造（WaypointTrajectory）**
构造步骤在 `DataGenerator.generate` 的子任务循环中：
1. 起始段：当前 eef 或上个子任务末端目标
2. 可选准备段：预旋转 / 预抬升 / 预平移
3. 目标段：对齐后的 target poses + gripper actions
4. 插值合并：起始段与目标段之间插入平滑过渡
5. 执行轨迹并收集 states/obs/actions/datagen_info

**八、输出与合并**
成功演示写到 `tmp/*.hdf5`，失败写到 `tmp_failed/*.hdf5`  
完成后合并为：
- `demo.hdf5`：成功演示
- `demo_failed.hdf5`：失败演示（可选）

**九、生成数据 HDF5 结构**
```text
<dataset>.hdf5
└─ data
   ├─ attrs
   │  ├─ total
   │  └─ env_args
   ├─ demo_0
   │  ├─ actions
   │  ├─ states 或 states/<key>
   │  ├─ obs/<key>
   │  ├─ datagen_info/...
   │  ├─ src_demo_inds (生成数据可选)
   │  ├─ src_demo_labels (生成数据可选)
   │  └─ attrs
   │     ├─ model_file (robosuite 可选)
   │     └─ num_samples
   ├─ demo_1
   └─ ...
```

**十、调试与缩小规模**
`generate_dataset.py --debug` 会自动：
- 只用少量源演示
- 生成极少条数
- 输出到临时目录 `/tmp/tmp_mimicgen`

**十一、读代码的最快路径（建议顺序）**
1. `mimicgen/scripts/generate_dataset.py`
2. `mimicgen/datagen/data_generator.py`
3. `mimicgen/utils/file_utils.py`
4. `mimicgen/utils/pose_utils.py`
5. `mimicgen/datagen/waypoint.py`

如需进一步细化某个函数或补充流程图，可在此报告基础上扩展。
