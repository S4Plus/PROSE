# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].
#让源数据拥有创建所需要的字段
"""
Script to extract information needed for data generation from low-dimensional simulation states
in a source dataset and add it to the source dataset. Basically a stripped down version of 
dataset_states_to_obs.py script in the robomimic codebase, with a handful of modifications.

Example usage:
    
    # prepare a source dataset collected on robosuite Stack task
    python prepare_src_dataset.py --dataset /path/to/stack.hdf5 --env_interface MG_Stack --env_interface_type robosuite

    # prepare a source dataset collected on robosuite Square task, but only use first 10 demos, and write output to new hdf5
    python prepare_src_dataset.py --dataset /path/to/square.hdf5 --env_interface MG_Square --env_interface_type robosuite --n 10 --output /tmp/square_new.hdf5
"""
import os
import shutil
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase

import mimicgen
import mimicgen.utils.file_utils as MG_FileUtils
from mimicgen.env_interfaces.base import make_interface


def extract_datagen_info_from_trajectory(
    env,
    env_interface,
    initial_state,
    states,
    actions,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of robomimic EnvBase): environment

        env_interface (MG_EnvInterface instance): environment interface for some data generation operations

        initial_state (dict): initial simulation state to load

        states (np.array): array of simulation states to load to extract information

        actions (np.array): array of actions

    Returns:
        datagen_infos (dict): the datagen info objects across all timesteps represented as a dictionary of
            numpy arrays, for easy writes to an hdf5


    提取的datagen_info包括：
        object_poses: 
    """
    # 只支持 robomimic 的 EnvBase 环境，且状态序列与动作序列必须一一对应
    assert isinstance(env, EnvBase)
    assert len(states) == actions.shape[0]

    # 先 reset 再加载初始状态，确保环境处于可控基准
    env.reset()
    env.reset_to(initial_state)#state是字典，包含object_poses,subtask_term_signals等

    all_datagen_infos = []
    traj_len = len(states)
    for t in range(traj_len):
        # 回放到第 t 步的仿真状态（逐步重置保证和源演示一致）,逐步按照states会放
        env.reset_to({"states" : states[t]})

        # 抽取 datagen 信息
        # 注意：部分环境接口会用到当前动作（例如夹爪动作）来补全信息
        datagen_info = env_interface.get_datagen_info(action=actions[t]).to_dict()
        all_datagen_infos.append(datagen_info)

    # list[dict] -> dict[list]，便于写入 hdf5
    all_datagen_infos = TensorUtils.list_of_flat_dict_to_dict_of_list(all_datagen_infos)

    for k in all_datagen_infos:
        if k in ["object_poses", "subtask_term_signals"]:
            # object_poses / subtask_term_signals 是嵌套 dict，需要再做一次转换
            all_datagen_infos[k] = TensorUtils.list_of_flat_dict_to_dict_of_list(all_datagen_infos[k])
            # 内层 list 转为 ndarray
            for k2 in all_datagen_infos[k]:
                all_datagen_infos[k][k2] = np.array(all_datagen_infos[k][k2])
        else:
            # 其他字段直接 list -> ndarray
            all_datagen_infos[k] = np.array(all_datagen_infos[k])

    return all_datagen_infos


def prepare_src_dataset(
    dataset_path,
    env_interface_name,
    env_interface_type,
    filter_key=None,
    n=None,
    output_path=None,
):
    """
    Adds DatagenInfo object instance for each timestep in each source demonstration trajectory 
    and stores it under the "datagen_info" key for each episode. Also store the @env_interface_name
    and @env_interface_type used in the attribute of each key. This information is used during 
    MimicGen data generation.

    Args:
        dataset_path (str): path to input hdf5 dataset, which will be modified in-place unless
            @output_path is provided

        env_interface_name (str): name of environment interface class to use for this source dataset

        env_interface_type (str): type of environment interface to use for this source dataset

        filter_key (str or None): name of filter key

        n (int or None): if provided, stop after n trajectories are processed

        output_path (str or None): if provided, write a new hdf5 here instead of modifying the
            original dataset in-place
    """

    # 如果提供 output_path，则复制一份数据集，避免直接修改原始文件，不提供的话默认原地修改
    if output_path is not None:
        shutil.copy(dataset_path, output_path)
        dataset_path = output_path

    # 读取数据集元信息，并创建用于数据回放的环境
    # 这里不采集图像（camera_names=[]），只用于状态回放与信息抽取，只需要提取datagen_info，不需要再提取图像
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[], 
        camera_height=84, 
        camera_width=84, 
        reward_shaping=False,
    )
    print("")
    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # 创建环境接口，用于从仿真中抽取 MimicGen 所需信息
    env_interface = make_interface(
        name=env_interface_name,
        interface_type=env_interface_type,
        # NOTE: env_interface takes underlying simulation environment, not robomimic wrapper
        env=env.base_env,
    )
    print("Created environment interface: {}".format(env_interface))
    print("")

    # robosuite 数据需要额外处理（reset_to 依赖 model_file）
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # 获取需要处理的演示列表
    # filter_key 可以指定数据集中的过滤键，n 用于仅处理前 n 条
    #demos 是一个列表，里面存储的是演示的 key,比如['demo_0', 'demo_1', ...]
    demos = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=dataset_path,
        filter_key=filter_key,
        start=None,
        n=n,
    )

    print("File that will be modified with datagen info: {}".format(dataset_path))
    print("")

    # 以可写模式打开 hdf5 文件，准备写回 datagen_info，会修改dataset_path对应的文件
    f = h5py.File(dataset_path, "a")

    total_samples = 0
    for ind in tqdm(range(len(demos))):
        ep = demos[ind]#demos是['demo_0', 'demo_1', ...]
        #ep_grp是hdf5文件中的一条演示数据
        ep_grp = f["data/{}".format(ep)]#ep_grp是'demo_0'对应的group，比如<File_object>/data/demo_0

        # 读取演示的状态序列与初始状态
        states = ep_grp["states"][()]#states是<File_object>/data/demo_0/states
        initial_state = dict(states=states[0])#initial_state是<File_object>/data/demo_0/states[0]
        if is_robosuite_env:
            # robosuite 需要额外传入 model_file 以便正确恢复仿真
            initial_state["model"] = ep_grp.attrs["model_file"]

        # 读取动作序列，并逐步回放提取 datagen_info
        actions = ep_grp["actions"][()]#actions是<File_object>/data/demo_0/actions
        datagen_info = extract_datagen_info_from_trajectory(
            env=env,
            env_interface=env_interface,
            initial_state=initial_state,
            states=states,
            actions=actions,
        )

        # 如果已存在 datagen_info，先删除再写入，避免冲突，datagen_info是字典,写入每个演示数据中的
        if "datagen_info" in ep_grp:
            del ep_grp["datagen_info"]

        # 将 datagen_info 写回到 hdf5
        for k in datagen_info:
            if k in ["object_poses", "subtask_term_signals"]:
                # 嵌套 dict 字段以子 key 形式写入，保证 datagen_info 的结构与 robomimic 保持一致，都是<object_poses>/<object_name>/<pose_type>
                for k2 in datagen_info[k]:
                    ep_grp.create_dataset("datagen_info/{}/{}".format(k, k2), data=np.array(datagen_info[k][k2]))
            else:
                ep_grp.create_dataset("datagen_info/{}".format(k), data=np.array(datagen_info[k]))

        # 记录所用的环境接口类型，方便后续生成时自动匹配
        ep_grp["datagen_info"].attrs["env_interface_name"] = env_interface_name
        ep_grp["datagen_info"].attrs["env_interface_type"] = env_interface_type

    print("Modified {} trajectories to include datagen info.".format(len(demos)))
    f.close()


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset, which will be modified in-place",
    )
    parser.add_argument(
        "--env_interface",
        type=str,
        required=True,
        help="name of environment interface class to use for this source dataset",
    )
    parser.add_argument(
        "--env_interface_type",
        type=str,
        required=True,
        help="type of environment interface to use for this source dataset",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) name of filter key, to select a subset of demo keys in the source hdf5",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="(optional) path to output hdf5 dataset, instead of modifying existing dataset in-place",
    )

    args = parser.parse_args()
    prepare_src_dataset(
        dataset_path=args.dataset,
        env_interface_name=args.env_interface,
        env_interface_type=args.env_interface_type,
        filter_key=args.filter_key,
        n=args.n,
        output_path=args.output,
    )
