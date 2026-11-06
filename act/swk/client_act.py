import argparse
import base64
import json
import traceback
import h5py
import pickle
import rospy
import time
import sys
import os
import cv2
import numpy as np
from urllib import error, request
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
from std_msgs.msg import Header

import collections
from collections import deque


#===== 配置 =====
 
# === 运行期日志缓冲（仅保存“已发布”的动作） ===
ACTIONS_LOG   = []  # 每帧 (14,) np.float32
STATE_LOG     = []  # 每帧 (14,) np.float32（发布时刻的 latest_state，若无则全0）
STAMP_LOG     = []  # 每帧 float（rospy.get_time() 单调时间）
ROLLOUT_LOG   = []  # 每条推理记录：observation + action




class RosOperator:
    def __init__(self, args):
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.bridge = None
        self.args = args
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()


    def get_frame(self):
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0: 
            return False
        else:
            frame_time = min([self.img_left_deque[-1].header.stamp.to_sec(), self.img_right_deque[-1].header.stamp.to_sec(), self.img_front_deque[-1].header.stamp.to_sec()])

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), desired_encoding="bgr8")
        # print("img_left:", img_left.shape)

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), desired_encoding="bgr8")

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), desired_encoding="bgr8")

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

       
        return (img_front, img_left, img_right, 
                puppet_arm_left, puppet_arm_right)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def init_ros(self):
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        
        rospy.Subscriber(self.args.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)


    def process(self):

        rate = rospy.Rate(self.args.frame_rate)
        print_flag = True
        #收集数据
        while not rospy.is_shutdown():
            result = self.get_frame()
            if not result:
                if print_flag:
                    print("syn fail")
                    print_flag = False
                # rate.sleep()
                rospy.sleep(0.03)
                continue
            print_flag = True
            (img_front, img_left, img_right,
             puppet_arm_left, puppet_arm_right) = result
            qpos = np.concatenate(
            (np.array(puppet_arm_left.position),
             np.array(puppet_arm_right.position)),
            axis=0
            )

            yield img_front, img_left, img_right, qpos

            rate.sleep()
            
    def get_latest_qpos(self):
        """非阻塞地读取当前最新的 qpos"""
        if len(self.puppet_arm_left_deque) == 0 or len(self.puppet_arm_right_deque) == 0:
            return None
        left = np.array(self.puppet_arm_left_deque[-1].position)
        right = np.array(self.puppet_arm_right_deque[-1].position)
        return np.concatenate([left, right], axis=0).astype(np.float32)

            
def http_get(url: str, timeout: float) -> dict:
    with request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8") if exc.fp is not None else ""
        if body:
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                pass
        raise


def encode_image_array(image: np.ndarray) -> str:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.shape[2] != 3:
        raise ValueError(f"Unsupported channel count: {array.shape}")

    if array.dtype != np.uint8:
        max_val = float(array.max()) if array.size else 1.0
        if max_val <= 1.0:
            array = np.clip(array, 0.0, 1.0) * 255.0
        else:
            array = np.clip(array, 0.0, 255.0)
        array = array.astype(np.uint8)

    ok, buf = cv2.imencode(".png", array)
    if not ok:
        raise ValueError("Failed to encode image to PNG.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class HTTPClient:
    def __init__(self, server_url: str, timeout: float) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)

    def health(self) -> dict:
        return http_get(f"{self.server_url}/health", self.timeout)

    def infer(self, payload: dict) -> dict:
        return http_post(f"{self.server_url}/infer", payload, self.timeout)


def resolve_env_state_vector(
    cli_value: str | None,
    expect_env_state: bool,
    env_state_dim: int | None,
) -> list[float] | None:
    if cli_value:
        values = [float(item.strip()) for item in cli_value.split(",") if item.strip()]
        if not values:
            raise ValueError("--env_state 不能为空。")
        if env_state_dim is not None and len(values) != env_state_dim:
            raise ValueError(f"--env_state 维度错误，期望 {env_state_dim}，实际 {len(values)}")
        return values

    if not expect_env_state:
        return None
    if env_state_dim == 1:
        return [1.0]
    raise ValueError("服务端要求 env_state，但当前未传 --env_state，且无法自动推断。")

def print_observation_summary(obs):
    print("=== observation summary ===")
    s = np.asarray(obs["state"], dtype=np.float32)
    print(f"state: shape={s.shape}, dtype={s.dtype}, head={s[:min(14, s.size)]}")
    for k, v in obs["images"].items():
        if isinstance(v, np.ndarray):
            patch = v[:2, :2, :]  # HWC 左上角 2x2
            print(f"{k}: shape={v.shape}, dtype={v.dtype}, top-left 2x2=\n{patch}\n")
        else:
            print(f"{k}: type={type(v)}\n")

def print_send_info_summary(obs):
    """打印发送信息的简要概述（格式、大小等）"""
    print("\n=== 发送数据概述 ===")

    # 计算总数据大小
    total_bytes = 0

    # 图像数据
    print("图像数据:")
    for k, v in obs["images"].items():
        if isinstance(v, np.ndarray):
            size_bytes = v.nbytes
            total_bytes += size_bytes
            h, w = v.shape[0], v.shape[1]
            print(f"  {k}: {v.shape} ({h}x{w}), dtype={v.dtype}, size={size_bytes/1024:.2f} KB")
        elif isinstance(v, str):
            size_bytes = len(v.encode("utf-8"))
            total_bytes += size_bytes
            print(f"  {k}: base64 string, size={size_bytes/1024:.2f} KB")
        else:
            print(f"  {k}: unsupported type {type(v)}")

    # 状态数据
    state = obs["state"]
    state_arr = np.asarray(state, dtype=np.float32)
    state_bytes = state_arr.nbytes
    total_bytes += state_bytes
    print(f"状态数据: shape={state_arr.shape}, dtype={state_arr.dtype}, size={state_bytes} bytes")


def print_receive_info_summary(action_chunk):
    """打印接收数据的简要概述"""
    print("=== 接收数据概述 ===")

    # 转换为numpy数组以便分析
    actions = np.atleast_2d(action_chunk)

    # 基本信息
    data_bytes = actions.nbytes
    print(f"动作数据: shape={actions.shape}, dtype={actions.dtype}, size={data_bytes} bytes")

    # 数值范围
    print(f"数值范围: min={actions.min():.4f}, max={actions.max():.4f}")

    # 如果是多步动作，显示步数
    if len(actions.shape) > 1 and actions.shape[0] > 1:
        print(f"动作步数: {actions.shape[0]} 步")

    print("=" * 40)

def infer_with_timing(client, observation):
    """执行推理并统计延迟"""
    # 打印发送信息概述
    print_send_info_summary(observation)

    # 记录发送时间
    send_time = time.time()
    action_chunk = client.infer(observation)["action"]
    # 记录接收时间并计算延迟
    recv_time = time.time()
    delay_ms = (recv_time - send_time) * 1000

    print(f"[延迟统计] 发送到接收延迟: {delay_ms:.2f} ms ({delay_ms/1000:.3f} s)\n")

    return action_chunk

# === logs 记录 ===

class Save_as_hdf5:
    def _stack_safe(lst, dim=1, dtype=np.float32):
        if not lst:
            return np.zeros((0, 14), dtype=dtype) if dim == 2 else np.zeros((14,), dtype=dtype)
        return np.stack(lst, axis=0).astype(dtype)

    def _ensure_dir(p):
        if not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)

    def save_to_hdf5(save_dir, save_name=None):
        """
        把内存中的动作日志一次性写入 HDF5。
        结构：
        /action                (N,14) float32  —— 已发布的每帧动作
        /observations/state    (N,14) float32  —— 发布时刻的 state（若不可得则为0）
        /timestamps            (N,)   float64  —— 发布时刻的 rospy 时间
        attrs:
            sim      False
            compress True
            created  ISO8601 字符串
            count    N
        #hdf5
        """
        N = len(ACTIONS_LOG)
        if N == 0:
            rospy.logwarn("[logger] No actions to save; skip writing HDF5.")
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.expanduser(save_dir or "~/openpi_logs")
        Save_as_hdf5._ensure_dir(save_dir)
        fname = save_name if save_name else f"actions_{ts}.hdf5"
        fpath = os.path.join(save_dir, fname)

        actions_np = Save_as_hdf5._stack_safe(ACTIONS_LOG, dtype=np.float32)
        states_np  = Save_as_hdf5._stack_safe(STATE_LOG,   dim=2, dtype=np.float32)
        stamps_np  = np.array(STAMP_LOG, dtype=np.float64)

        with h5py.File(fpath, "w") as f:
            f.attrs["sim"] = False
            f.attrs["compress"] = True
            f.attrs["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            f.attrs["count"] = int(N)

            # 使用 chunk + gzip 压缩
            f.create_dataset("/action", data=actions_np,
                            dtype="float32", compression="gzip", compression_opts=2,
                            chunks=(min(N, 4096), 14))
            f.create_dataset("/observations/state", data=states_np,
                            dtype="float32", compression="gzip", compression_opts=2,
                            chunks=(min(N, 4096), 14))
            f.create_dataset("/timestamps", data=stamps_np,
                            dtype="float64", compression="gzip", compression_opts=2,
                            chunks=(min(N, 4096),))

        rospy.loginfo(f"[logger] Saved {N} frames to {fpath}")


    def _on_shutdown():  # === NEW ===
        """
        ROS 正常退出或 Ctrl+C 时触发。把累计的动作写入 HDF5。
        可通过参数覆盖：
        ~save_dir  (默认: ~/openpi_logs)
        ~save_name (默认: actions_YYYYmmdd_HHMMSS.hdf5)
        """
        save_dir  = rospy.get_param("~save_dir", "~/code/dp/act/logs")
        save_name = rospy.get_param("~save_name", None)
        try:
            Save_as_hdf5.save_to_hdf5(save_dir, save_name)
        except Exception as e:
            rospy.logerr(f"[logger] Failed to save HDF5: {e}")


class Save_as_pkl:
    def _ensure_dir(p):
        if not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)

    def save_to_pkl(save_dir, save_name=None):
        """
        把推理的 observation + action 保存为 pkl。
        结构：
        {"frames": [ {"frame_index","timestamp","images","state","observation","action"}, ... ]}
        """
        if not ROLLOUT_LOG:
            rospy.logwarn("[logger] No rollout records to save; skip writing PKL.")
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.expanduser(save_dir or "~/code/dp/act/logs_pkl")
        Save_as_pkl._ensure_dir(save_dir)
        fname = save_name if save_name else f"rollout_{ts}.pkl"
        fpath = os.path.join(save_dir, fname)

        with open(fpath, "wb") as f:
            pickle.dump({"frames": ROLLOUT_LOG}, f, protocol=pickle.HIGHEST_PROTOCOL)

        rospy.loginfo(f"[logger] Saved {len(ROLLOUT_LOG)} rollout records to {fpath}")

    def _on_shutdown():
        save_dir = rospy.get_param("~rollout_dir", rospy.get_param("~save_dir", "~/code/dp/act/logs_pkl"))
        save_name = rospy.get_param("~rollout_name", None)
        try:
            Save_as_pkl.save_to_pkl(save_dir, save_name)
        except Exception as e:
            rospy.logerr(f"[logger] Failed to save PKL: {e}")

# 运行时缓存
pending_actions = None

pending_curpos = None
last_pub_time = 0.0

        
def main(args):
    print(1)
    rospy.init_node('record_episodes', anonymous=True)
    rospy.on_shutdown(Save_as_hdf5._on_shutdown)  
    rospy.on_shutdown(Save_as_pkl._on_shutdown)
    ros_operator = RosOperator(args)

    client = HTTPClient(args.server_url, args.timeout)
    expect_env_state = False
    env_state_dim = None
    try:
        health = client.health()
        if health.get("status") not in ("ok", "healthy"):
            rospy.logwarn(f"[client] Server health check failed: {health}")
        expect_env_state = bool(health.get("expect_env_state", False))
        if health.get("env_state_dim") is not None:
            env_state_dim = int(health["env_state_dim"])
    except Exception as exc:
        rospy.logwarn(f"[client] Server health check error: {exc}")

    env_state_vector = resolve_env_state_vector(args.env_state, expect_env_state, env_state_dim)
    
    master_arm_left_publisher = rospy.Publisher(args.master_arm_left_topic, JointState, queue_size=10)
    master_arm_right_publisher = rospy.Publisher(args.master_arm_right_topic, JointState, queue_size=10)
    
    joint_state_msg = JointState()
    joint_state_msg.header =  Header()
    joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称

    rate = rospy.Rate(args.frame_rate)
    rollout_idx = 0

    for img_front, img_left, img_right, qpos in ros_operator.process():
        observation = {
            "images": {
                "observation.images.cam_high": encode_image_array(img_front),
                "observation.images.cam_left_wrist": encode_image_array(img_left),
                "observation.images.cam_right_wrist": encode_image_array(img_right),
            },
            "state": qpos.astype(float).tolist(),      # (14,) float32
            }
        if env_state_vector is not None:
            observation["env_state"] = env_state_vector
        
        # 打印发送信息概述
        print_send_info_summary(observation)

        try:
            # 记录发送时间
            send_time = time.time()
            action_chunk = client.infer(observation)["action"]
            # 记录接收时间并计算延迟
            recv_time = time.time()
            delay_ms = (recv_time - send_time) * 1000
            print(f"[延迟统计] 发送到接收延迟: {delay_ms:.2f} ms ({delay_ms/1000:.3f} s)\n")

            # 打印接收数据概述
            print_receive_info_summary(action_chunk)

        except Exception as e:
            rospy.logwarn(f"Infer timeout or error: {e}\n{traceback.format_exc()}")
            continue

        actions = np.atleast_2d(np.asarray(action_chunk, dtype=np.float32))
        ROLLOUT_LOG.append(
            {
                "frame_index": rollout_idx,
                "timestamp": float(rospy.get_time()),
                "images": observation["images"],
                "state": observation["state"],
                "observation": observation,
                "action": actions.copy(),
            }
        )
        rollout_idx += 1

        for i in range(min(len(actions), args.max_timesteps)):
            if rospy.is_shutdown():
                break
            print("left: ", np.round(actions[i][:7], 4), " right: ", np.round(actions[i][7:], 4))

            cur_timestamp = rospy.Time.now()  # 设置时间戳

            # 记录（确保不与原数组共享引用）
            qposs = ros_operator.get_latest_qpos()

            cmd14 = np.concatenate([np.array(actions[i][:7], dtype=np.float32),
                                    np.array(actions[i][7:], dtype=np.float32)], axis=0)
            ACTIONS_LOG.append(cmd14)
            STAMP_LOG.append(rospy.get_time())
            STATE_LOG.append(qposs.copy() if qposs is not None else np.zeros(14, np.float32))
            rospy.loginfo_throttle(1.0,
                f"L0-2={np.array(qposs)[:3]} R0-2={np.array(qposs)[7:10]} (logged total={len(ACTIONS_LOG)})"
            )

            # input()
            joint_state_msg.header.stamp = cur_timestamp
            joint_state_msg.position = actions[i][:7]
            master_arm_left_publisher.publish(joint_state_msg)

            joint_state_msg.position = actions[i][7:]
            master_arm_right_publisher.publish(joint_state_msg)

            rate.sleep()
        rospy.sleep(0.3)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset_dir.',
                        default="./data", required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.',
                        default="aloha_mobile_dummy", required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.',
                        default=0, required=False)
    
    parser.add_argument('--max_timesteps', action='store', type=int, help='Max_timesteps.',
                        default=20, required=False)

    parser.add_argument('--camera_names', nargs='+',
                        default=['observation.images.cam_high', 'observation.images.cam_left_wrist',
                                 'observation.images.cam_right_wrist'])
    #  topic name of color image
    parser.add_argument('--img_front_topic',
                        default='/camera_f/color/image_raw')
    parser.add_argument('--img_left_topic', 
                        default='/camera_l/color/image_raw')
    parser.add_argument('--img_right_topic',
                        default='/camera_r/color/image_raw')
    
    # topic name of arm
    parser.add_argument('--master_arm_left_topic',
                        default='/master/joint_left')
    parser.add_argument('--master_arm_right_topic',
                        default='/master/joint_right')
    parser.add_argument('--puppet_arm_left_topic', 
                        default='/puppet/joint_left')
    parser.add_argument('--puppet_arm_right_topic', 
                        default='/puppet/joint_right')
    
    parser.add_argument('--frame_rate', action='store', type=int, help='frame_rate',
                        default=30, required=False)
    parser.add_argument('--server_url', action='store', type=str, help='HTTP server URL.',
                        default="http://10.0.40.113:8000", required=False)
    parser.add_argument('--timeout', action='store', type=float, help='HTTP timeout seconds.',
                        default=10.0, required=False)
    parser.add_argument('--env_state', action='store', type=str,
                        help='可选环境条件向量，逗号分隔；单任务 dim=1 时默认自动补 1.0',
                        default=None, required=False)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments() 
    main(args)
  
