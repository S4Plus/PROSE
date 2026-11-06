# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Main data generation script.

Examples:

    # run normal data generation
    python generate_dataset.py --config /path/to/config.json

    # render all data generation attempts on-screen
    python generate_dataset.py --config /path/to/config.json --render

    # render all data generation attempts to a video
    python generate_dataset.py --config /path/to/config.json --video_path /path/to/video.mp4

    # run a quick debug run
    python generate_dataset.py --config /path/to/config.json --debug

    # pause after every subtask to debug data generation
    python generate_dataset.py --config /path/to/config.json --render --pause_subtask
"""

import os
import shutil
import json
import time
import argparse
import traceback
import random
import csv
import imageio
import numpy as np
from copy import deepcopy

import robomimic
from robomimic.utils.file_utils import get_env_metadata_from_dataset

import mimicgen
import mimicgen.utils.file_utils as MG_FileUtils
import mimicgen.utils.robomimic_utils as RobomimicUtils

from mimicgen.configs import config_factory, MG_TaskSpec
from mimicgen.datagen.data_generator import DataGenerator
from mimicgen.env_interfaces.base import make_interface


def _read_float_from_row(row, keys):
    # 从一行 CSV（dict）中按候选键读取第一个可解析的浮点值
    for key in keys:
        if key in row and row[key] != "":
            try:
                return float(row[key])
            except ValueError:
                continue
    return None


def _load_nut_positions_csv(path, x_key="x", y_key="y"):
    # 读取 nut 位置 CSV，兼容有表头与无表头两种格式
    if path is None:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError("nut positions csv not found: {}".format(path))
    points = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            # 有表头：按列名解析
            for row in reader:
                x_val = _read_float_from_row(row, [x_key, "x", "center_x"])
                y_val = _read_float_from_row(row, [y_key, "y", "center_y"])
                if x_val is None or y_val is None:
                    continue
                points.append((x_val, y_val))
        else:
            # 无表头：按前两列解析
            f.seek(0)
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                try:
                    points.append((float(row[0]), float(row[1])))
                except ValueError:
                    continue
    return points


def _get_active_nut(env):
    # 从环境中获取当前活跃的 nut 对象（兼容不同任务实现）
    if not hasattr(env, "nuts"):
        return None
    if hasattr(env, "nut_id"):
        try:
            return env.nuts[int(env.nut_id)]
        except Exception:
            pass
    if hasattr(env, "nut_to_id"):
        for key in ("square", "square_nut"):
            if key in env.nut_to_id:
                return env.nuts[int(env.nut_to_id[key])]
    try:
        return env.nuts[0]
    except Exception:
        return None


def _set_nut_xy(env, x_val, y_val):
    # 设置 nut 在平面上的位置（用于固定/遍历初始分布）
    nut = _get_active_nut(env)
    if nut is None or not hasattr(env, "sim"):
        return False
    try:
        joint_name = nut.joints[0]
        qpos = np.array(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
        if qpos.size < 2:
            return False
        qpos[0] = x_val
        qpos[1] = y_val
        env.sim.data.set_joint_qpos(joint_name, qpos)
        env.sim.forward()
        return True
    except Exception:
        return False


def get_important_stats(
    new_dataset_folder_path,
    num_success,
    num_failures,
    num_attempts,
    num_problematic,
    start_time=None,
    ep_length_stats=None,
):
    """
    Return a summary of important stats to write to json.

    Args:
        new_dataset_folder_path (str): path to folder that will contain generated dataset
        num_success (int): number of successful trajectories generated
        num_failures (int): number of failed trajectories
        num_attempts (int): number of total attempts
        num_problematic (int): number of problematic trajectories that failed due
            to a specific exception that was caught
        start_time (float or None): starting time for this run from time.time()
        ep_length_stats (dict or None): if provided, should have entries that summarize
            the episode length statistics over the successfully generated trajectories

    Returns:
        important_stats (dict): dictionary with useful summary of statistics
    """
    # 汇总关键统计指标，便于写入日志与 json
    important_stats = dict(
        generation_path=new_dataset_folder_path,
        success_rate=((100. * num_success) / num_attempts),
        failure_rate=((100. * num_failures) / num_attempts),
        num_success=num_success,
        num_failures=num_failures,
        num_attempts=num_attempts,
        num_problematic=num_problematic,
    )
    if (ep_length_stats is not None):
        important_stats.update(ep_length_stats)
    if start_time is not None:
        # add in time taken
        important_stats["time spent (hrs)"] = "{:.2f}".format((time.time() - start_time) / 3600.)
    return important_stats


def generate_dataset(
    mg_config,
    auto_remove_exp=False,
    render=False,
    video_path=None,
    video_skip=5,
    render_image_names=None,
    pause_subtask=False,
    nut_positions=None,
    nut_positions_cycle=False,
):
    """
    Main function to collect a new dataset with MimicGen.

    Args:
        mg_config (MG_Config instance): MimicGen config object

        auto_remove_exp (bool): if True, will remove generation folder if it exists, else
            user will be prompted to decide whether to keep existing folder or not

        render (bool): if True, render each data generation attempt on-screen

        video_path (str or None): if provided, render the data generation attempts to the 
            provided video path

        video_skip (int): skip every nth frame when writing video

        render_image_names (list of str or None): if provided, specify camera names to 
            use during on-screen / off-screen rendering to override defaults

        pause_subtask (bool): if True, pause after every subtask during generation, for
            debugging.
        nut_positions (list or None): list of (x, y) tuples to set nut positions
            after env.reset for each trial.
        nut_positions_cycle (bool): if True, cycle through nut positions when
            number of trials exceeds provided points.
    """

    # time this run
    start_time = time.time()

    # 生成视频与屏幕渲染互斥
    # check some args
    write_video = (video_path is not None)
    assert not (render and write_video) # either on-screen or video but not both
    if pause_subtask:
        assert render, "should enable on-screen rendering for pausing to be useful"

    if write_video:
        # debug video - use same cameras as observations
        if len(mg_config.obs.camera_names) > 0:
            assert render_image_names is None
            render_image_names = list(mg_config.obs.camera_names)

    # 源数据路径
    # path to source dataset
    source_dataset_path = os.path.expandvars(os.path.expanduser(mg_config.experiment.source.dataset_path))

    # 从源数据读取环境元信息
    # get environment metadata from dataset
    env_meta = get_env_metadata_from_dataset(dataset_path=source_dataset_path)

    # 固定随机种子，保证可复现
    # set seed for generation
    random.seed(mg_config.experiment.seed)
    np.random.seed(mg_config.experiment.seed)

    # 创建本次生成的输出目录
    # create new folder for this data generation run
    base_folder = os.path.expandvars(os.path.expanduser(mg_config.experiment.generation.path))
    new_dataset_folder_name = mg_config.experiment.name
    new_dataset_folder_path = os.path.join(
        base_folder,
        new_dataset_folder_name,
    )
    print("\nData will be generated at: {}".format(new_dataset_folder_path))

    # 如果目录存在，按配置决定是否清理
    # ensure dataset folder does not exist, and make new folder
    exist_ok = False
    if os.path.exists(new_dataset_folder_path):
        if not auto_remove_exp:
            ans = input("\nWARNING: dataset folder ({}) already exists! \noverwrite? (y/n)\n".format(new_dataset_folder_path))
        else:
            ans = "y"
        if ans == "y":
            print("Removed old results folder at {}".format(new_dataset_folder_path))
            shutil.rmtree(new_dataset_folder_path)
        else:
            print("Keeping old dataset folder. Note that individual files may still be overwritten.")
            exist_ok = True
    os.makedirs(new_dataset_folder_path, exist_ok=exist_ok)

    # 将终端输出写入日志文件
    # log terminal output to text file
    RobomimicUtils.make_print_logger(txt_file=os.path.join(new_dataset_folder_path, 'log.txt'))

    # 保存配置到输出目录
    # save config to disk
    MG_FileUtils.write_json(
        json_dic=mg_config,
        json_path=os.path.join(new_dataset_folder_path, "mg_config.json"),
    )

    print("\n============= Config =============")
    print(mg_config)
    print("")

    # some paths that we will create inside our new dataset folder

    # new dataset that will be generated
    new_dataset_path = os.path.join(new_dataset_folder_path, "demo.hdf5")

    # 保存单条演示的临时目录
    # tmp folder that will contain per-episode hdf5s that were successful (they will be merged later)
    tmp_dataset_folder_path = os.path.join(new_dataset_folder_path, "tmp")
    os.makedirs(tmp_dataset_folder_path, exist_ok=exist_ok)

    # folder containing logs
    json_log_path = os.path.join(new_dataset_folder_path, "logs")
    os.makedirs(json_log_path, exist_ok=exist_ok)

    if mg_config.experiment.generation.keep_failed:
        # new dataset for failed trajectories, and tmp folder for per-episode hdf5s that failed
        new_failed_dataset_path = os.path.join(new_dataset_folder_path, "demo_failed.hdf5")
        tmp_dataset_failed_folder_path = os.path.join(new_dataset_folder_path, "tmp_failed")
        os.makedirs(tmp_dataset_failed_folder_path, exist_ok=exist_ok)

    # 获取源数据的演示列表
    # get list of source demonstration keys from source hdf5
    all_demos = MG_FileUtils.get_all_demos_from_dataset(
        dataset_path=source_dataset_path,
        filter_key=mg_config.experiment.source.filter_key,
        start=mg_config.experiment.source.start,
        n=mg_config.experiment.source.n,
    )

    # prepare args for creating simulation environment

    # auto-fill camera rendering info if not specified
    if (write_video or render) and (render_image_names is None):
        render_image_names = RobomimicUtils.get_default_env_cameras(env_meta=env_meta)
    if render:
        # on-screen rendering can only support one camera
        assert len(render_image_names) == 1

    # 构建仿真环境
    # env args: cameras to use come from debug camera video to write, or from observation collection
    camera_names = (mg_config.obs.camera_names if not write_video else render_image_names)

    # env args: don't use image obs when writing debug video
    use_image_obs = ((mg_config.obs.collect_obs and (len(mg_config.obs.camera_names) > 0)) if not write_video else False)
    use_depth_obs = False
    
    # simulation environment
    env = RobomimicUtils.create_env(
        env_meta=env_meta,
        env_class=None,
        env_name=mg_config.experiment.task.name,
        robot=mg_config.experiment.task.robot,
        gripper=mg_config.experiment.task.gripper,
        env_meta_update_kwargs=mg_config.experiment.task.env_meta_update_kwargs,
        camera_names=camera_names,
        camera_height=mg_config.obs.camera_height,
        camera_width=mg_config.obs.camera_width,
        render=render, 
        render_offscreen=write_video,
        use_image_obs=use_image_obs,
        use_depth_obs=use_depth_obs,
    )
    print("\n==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # 从源数据读取环境接口信息
    # get information necessary to create env interface
    env_interface_name, env_interface_type = MG_FileUtils.get_env_interface_info_from_dataset(
        dataset_path=source_dataset_path,
        demo_keys=all_demos,
    )
    # possibly override from config
    if mg_config.experiment.task.interface is not None:
        env_interface_name = mg_config.experiment.task.interface
    if mg_config.experiment.task.interface_type is not None:
        env_interface_type = mg_config.experiment.task.interface_type

    # 创建环境接口实例
    # create environment interface to use during data generation
    env_interface = make_interface(
        name=env_interface_name,
        interface_type=env_interface_type,
        # NOTE: env_interface takes underlying simulation environment, not robomimic wrapper
        env=env.base_env,
    )
    print("Created environment interface: {}".format(env_interface))

    # 仅捕获 rollouts 相关异常，避免中断
    # make sure we except the same exceptions that we would normally except during policy rollouts
    exceptions_to_except = env.rollout_exceptions

    # 读取 TaskSpec
    # get task spec object from config
    task_spec_json_string = mg_config.task.task_spec.dump()
    task_spec = MG_TaskSpec.from_json(json_string=task_spec_json_string)

    # 构建 DataGenerator 并解析源数据
    # make data generator object
    data_generator = DataGenerator(
        task_spec=task_spec,
        dataset_path=source_dataset_path,
        demo_keys=all_demos,
    )

    print("\n==== Created Data Generator ====")
    print(data_generator)
    print("")

    # 可选视频写入
    # we might write a video to show the data generation attempts
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(video_path, fps=20)

    # data generation statistics
    num_success = 0
    num_failures = 0
    num_attempts = 0
    num_problematic = 0
    ep_lengths = [] # episode lengths for successfully generated data
    selected_src_demo_inds_all = [] # selected source demo index in @all_demos for each trial
    selected_src_demo_inds_succ = [] # selected source demo index in @all_demos for each successful trial
    if nut_positions:
        print("Using {} nut positions for generation.".format(len(nut_positions)))

    # we will keep generating data until @num_trials successes (if @guarantee_success) else @num_trials attempts
    num_trials = mg_config.experiment.generation.num_trials
    guarantee_success = mg_config.experiment.generation.guarantee

    while True:

        # 可选设置物体初始位置（如 nut）
        nut_xy = None
        reset_callback = None
        if nut_positions:
            nut_index = num_success if guarantee_success else num_attempts
            if nut_index >= len(nut_positions):
                if nut_positions_cycle and len(nut_positions) > 0:
                    nut_index = nut_index % len(nut_positions)
                else:
                    print("No more nut positions ({}). Stopping early.".format(len(nut_positions)))
                    break
            nut_xy = nut_positions[nut_index]
            print("Using nut position {}: ({:.6f}, {:.6f})".format(nut_index, nut_xy[0], nut_xy[1]))

            def reset_callback(env, env_interface, nut_xy=nut_xy):
                base_env = env.base_env if hasattr(env, "base_env") else env
                if not _set_nut_xy(base_env, nut_xy[0], nut_xy[1]):
                    raise RuntimeError(
                        "Failed to set nut xy to ({:.6f}, {:.6f})".format(nut_xy[0], nut_xy[1])
                    )

        # 生成一条新演示
        # generate trajectory
        try:
            generated_traj = data_generator.generate(
                env=env,
                env_interface=env_interface,
                select_src_per_subtask=mg_config.experiment.generation.select_src_per_subtask,
                transform_first_robot_pose=mg_config.experiment.generation.transform_first_robot_pose,
                interpolate_from_last_target_pose=mg_config.experiment.generation.interpolate_from_last_target_pose,
                render=render,
                video_writer=video_writer,
                video_skip=video_skip,
                camera_names=render_image_names,
                pause_subtask=pause_subtask,
                reset_callback=reset_callback,
            )
        except exceptions_to_except as e:
            # problematic trajectory - do not have this count towards our total number of attempts, and re-try
            print("")
            print("*" * 50)
            print("WARNING: got rollout exception {}".format(e))
            print("*" * 50)
            print("")
            num_problematic += 1
            continue

        # 记录源演示选择索引
        # remember selection of source demos for each subtask
        selected_src_demo_inds_all.append(generated_traj["src_demo_inds"])

        # 判断是否成功
        # check if generated trajectory was successful
        success = bool(generated_traj["success"])

        if success:
            num_success += 1

            # 成功轨迹写入临时文件
            # store successful demonstration
            ep_lengths.append(generated_traj["actions"].shape[0])
            MG_FileUtils.write_demo_to_hdf5(
                folder=tmp_dataset_folder_path,
                env=env,
                initial_state=generated_traj["initial_state"],
                states=generated_traj["states"],
                observations=(generated_traj["observations"] if mg_config.obs.collect_obs else None),
                datagen_info=generated_traj["datagen_infos"],
                actions=generated_traj["actions"],
                src_demo_inds=generated_traj["src_demo_inds"],
                src_demo_labels=generated_traj["src_demo_labels"],
            )
            selected_src_demo_inds_succ.append(generated_traj["src_demo_inds"])
        else:
            num_failures += 1

            # 可选保存失败轨迹
            # check if this failure should be kept
            if mg_config.experiment.generation.keep_failed and \
                ((mg_config.experiment.max_num_failures is None) or (num_failures <= mg_config.experiment.max_num_failures)):
                
                # save failed trajectory in separate folder
                MG_FileUtils.write_demo_to_hdf5(
                    folder=tmp_dataset_failed_folder_path,
                    env=env,
                    initial_state=generated_traj["initial_state"],
                    states=generated_traj["states"],
                    observations=(generated_traj["observations"] if mg_config.obs.collect_obs else None),
                    datagen_info=generated_traj["datagen_infos"],
                    actions=generated_traj["actions"],
                    src_demo_inds=generated_traj["src_demo_inds"],
                    src_demo_labels=generated_traj["src_demo_labels"],
                )

        num_attempts += 1
        print("")
        print("*" * 50)
        print("trial {} success: {}".format(num_attempts, success))
        print("have {} successes out of {} trials so far".format(num_success, num_attempts))
        print("have {} failures out of {} trials so far".format(num_failures, num_attempts))
        print("*" * 50)

        # regularly log progress to disk every so often
        if (num_attempts % mg_config.experiment.log_every_n_attempts) == 0:

            # get summary stats
            summary_stats = get_important_stats(
                new_dataset_folder_path=new_dataset_folder_path,
                num_success=num_success,
                num_failures=num_failures,
                num_attempts=num_attempts,
                num_problematic=num_problematic,
                start_time=start_time,
                ep_length_stats=None,
            )

            # write stats to disk
            max_digits = len(str(num_trials * 1000)) + 1 # assume we will never have lower than 0.1% data generation SR
            json_file_path = os.path.join(json_log_path, "attempt_{}_succ_{}_rate_{}.json".format(
                str(num_attempts).zfill(max_digits), # pad with leading zeros for ordered list of jsons in directory
                num_success,
                np.round((100. * num_success) / num_attempts, 2),
            ))
            MG_FileUtils.write_json(json_dic=summary_stats, json_path=json_file_path)

        # termination condition is on enough successes if @guarantee_success or enough attempts otherwise
        check_val = num_success if guarantee_success else num_attempts
        if check_val >= num_trials:
            break

    if write_video:
        video_writer.close()

    # 合并临时 hdf5 为最终数据集
    # merge all new created files
    print("\nFinished data generation. Merging per-episode hdf5s together...\n")
    MG_FileUtils.merge_all_hdf5(
        folder=tmp_dataset_folder_path,
        new_hdf5_path=new_dataset_path,
        delete_folder=True,
    )
    if mg_config.experiment.generation.keep_failed:
        MG_FileUtils.merge_all_hdf5(
            folder=tmp_dataset_failed_folder_path,
            new_hdf5_path=new_failed_dataset_path,
            delete_folder=True,
        )

    # get episode length statistics
    ep_length_stats = None
    if len(ep_lengths) > 0:
        ep_lengths = np.array(ep_lengths)
        ep_length_mean = float(np.mean(ep_lengths))
        ep_length_std = float(np.std(ep_lengths))
        ep_length_max = int(np.max(ep_lengths))
        ep_length_3std = int(np.ceil(ep_length_mean + 3. * ep_length_std))
        ep_length_stats = dict(
            ep_length_mean=ep_length_mean,
            ep_length_std=ep_length_std,
            ep_length_max=ep_length_max,
            ep_length_3std=ep_length_3std,
        )

    stats = get_important_stats(
        new_dataset_folder_path=new_dataset_folder_path,
        num_success=num_success,
        num_failures=num_failures,
        num_attempts=num_attempts,
        num_problematic=num_problematic,
        start_time=start_time,
        ep_length_stats=ep_length_stats,
    )
    print("\nStats Summary")
    print(json.dumps(stats, indent=4))

    # maybe render videos
    if mg_config.experiment.render_video:
        if (num_success > 0):
            playback_video_path = os.path.join(new_dataset_folder_path, "playback_{}.mp4".format(new_dataset_folder_name))
            num_render = mg_config.experiment.num_demo_to_render
            print("Rendering successful trajectories...")
            RobomimicUtils.make_dataset_video(
                dataset_path=new_dataset_path,
                video_path=playback_video_path,
                num_render=num_render,
            )
        else:
            print("\n" + "*" * 80)
            print("\nWARNING: skipping dataset video creation since no successes")
            print("\n" + "*" * 80 + "\n")
        if mg_config.experiment.generation.keep_failed:
            if (num_failures > 0):
                playback_video_path = os.path.join(new_dataset_folder_path, "playback_{}_failed.mp4".format(new_dataset_folder_name))
                num_render = mg_config.experiment.num_fail_demo_to_render
                print("Rendering failure trajectories...")
                RobomimicUtils.make_dataset_video(
                    dataset_path=new_failed_dataset_path,
                    video_path=playback_video_path,
                    num_render=num_render,
                )
            else:
                print("\n" + "*" * 80)
                print("\nWARNING: skipping dataset video creation since no failures")
                print("\n" + "*" * 80 + "\n")

    # return some summary info
    final_important_stats = get_important_stats(
        new_dataset_folder_path=new_dataset_folder_path,
        num_success=num_success,
        num_failures=num_failures,
        num_attempts=num_attempts,
        num_problematic=num_problematic,
        start_time=start_time,
        ep_length_stats=ep_length_stats,
    )

    # write stats to disk
    json_file_path = os.path.join(new_dataset_folder_path, "important_stats.json")
    MG_FileUtils.write_json(json_dic=final_important_stats, json_path=json_file_path)

    # NOTE: we are not currently saving the choice of source human demonstrations for each trial,
    #       but you can do that if you wish -- the information is stored in @selected_src_demo_inds_all
    #       and @selected_src_demo_inds_succ

    return final_important_stats


def main(args):

    # 读取外部 json 配置并构建 MG_Config
    # load config object
    with open(args.config, "r") as f:
        ext_cfg = json.load(f)
        # config generator from robomimic generates this part of config unused by MimicGen
        if "meta" in ext_cfg:
            del ext_cfg["meta"]
    mg_config = config_factory(ext_cfg["name"], config_type=ext_cfg["type"])

    # 用外部 json 覆盖默认配置（若含未知字段会报错）
    # update config with external json - this will throw errors if
    # the external config has keys not present in the base config
    with mg_config.values_unlocked():
        mg_config.update(ext_cfg)

        # We assume that the external config specifies all subtasks, so
        # delete any subtasks not in the external config.
        source_subtasks = set(mg_config.task.task_spec.keys())
        new_subtasks = set(ext_cfg["task"]["task_spec"].keys())
        for subtask in (source_subtasks - new_subtasks):
            print("deleting subtask {} in original config".format(subtask))
            del mg_config.task.task_spec[subtask]

        # 命令行参数可覆盖部分配置
        # maybe override some settings
        if args.task_name is not None:
            mg_config.experiment.task.name = args.task_name

        if args.source is not None:
            mg_config.experiment.source.dataset_path = args.source

        if args.folder is not None:
            mg_config.experiment.generation.path = args.folder

        if args.num_demos is not None:
            mg_config.experiment.generation.num_trials = args.num_demos

        if args.seed is not None:
            mg_config.experiment.seed = args.seed

        # debug 模式：缩短生成规模，输出到临时目录
        # maybe modify config for debugging purposes
        if args.debug:
            # shrink length of generation to test whether this run is likely to crash
            mg_config.experiment.source.n = 3
            mg_config.experiment.generation.guarantee = False
            mg_config.experiment.generation.num_trials = 2

            # send output to a temporary directory
            mg_config.experiment.generation.path = "/tmp/tmp_mimicgen"

    # 可选从 CSV 读取 nut 位置，用于固定/遍历初始分布
    nut_positions = None
    if args.nut_positions_csv is not None:
        nut_positions = _load_nut_positions_csv(
            path=args.nut_positions_csv,
            x_key=args.nut_positions_x_key,
            y_key=args.nut_positions_y_key,
        )
        if not nut_positions:
            raise SystemExit("No nut positions found in {}".format(args.nut_positions_csv))
        if args.num_demos is None and not args.debug:
            mg_config.experiment.generation.num_trials = len(nut_positions)

    # 执行生成并捕获异常，保证日志输出
    # catch error during generation and print it
    res_str = "finished run successfully!"
    important_stats = None
    try:
        important_stats = generate_dataset(
            mg_config=mg_config,
            auto_remove_exp=args.auto_remove_exp,
            render=args.render,
            video_path=args.video_path,
            video_skip=args.video_skip,
            render_image_names=args.render_image_names,
            pause_subtask=args.pause_subtask,
            nut_positions=nut_positions,
            nut_positions_cycle=args.nut_positions_cycle,
        )
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)
    if important_stats is not None:
        important_stats = json.dumps(important_stats, indent=4)
        print("\nFinal Data Generation Stats")
        print(important_stats)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to MimicGen config json",
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick generation run for debugging purposes",
    )
    parser.add_argument(
        "--auto-remove-exp",
        action='store_true',
        help="force delete the experiment folder if it exists"
    )
    parser.add_argument(
        "--render",
        action='store_true',
        help="render each data generation attempt on-screen",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="if provided, render the data generation attempts to the provided video path",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="skip every nth frame when writing video",
    )
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=None,
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )
    parser.add_argument(
        "--pause_subtask",
        action='store_true',
        help="pause after every subtask during generation for debugging - only useful with render flag",
    )
    parser.add_argument(
        "--source",
        type=str,
        help="path to source dataset, to override the one in the config",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="environment name to use for data generation, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="folder that will be created with new data, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--num_demos",
        type=int,
        help="number of demos to generate, or attempt to generate, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed, to override the one in the config",
        default=None,
    )
    parser.add_argument(
        "--nut_positions_csv",
        type=str,
        default=None,
        help="csv with nut (x,y) positions to use after env.reset for each trial",
    )
    parser.add_argument(
        "--nut_positions_x_key",
        type=str,
        default="x",
        help="column name for x in nut_positions_csv",
    )
    parser.add_argument(
        "--nut_positions_y_key",
        type=str,
        default="y",
        help="column name for y in nut_positions_csv",
    )
    parser.add_argument(
        "--nut_positions_cycle",
        action="store_true",
        help="cycle through nut positions when trials exceed provided points",
    )

    args = parser.parse_args()
    main(args)
