<!-- 1.abs绝对位姿
    3D位置+四元数+1D夹爪
2.delta位姿
    3D位置增量+3d旋转增量+1D夹爪 -->

dataset:
    1.square_d1是将peg固定在边界采集的
    2.square_d1_core：是原有数据集，peg和nut均是随机的


3.joint绝对关节角度
    采用机械臂的关节角度+夹爪角度拼接的9维
    action是8维：7维关节角度+开关指令

步骤：
    1.数据格式转换：从hdf5转化成lerobot格式，用绝对关节角度训练
    2.训练act
    3.eval
    4.打分







action到具体执行：
    转化成力矩（作用的力）