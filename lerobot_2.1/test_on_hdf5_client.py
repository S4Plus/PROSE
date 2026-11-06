"""
通过 FastAPI 调用远程策略模型进行推理，然后计算 MSE/RMSE/MAE 多步误差。
原来的本地 policy.predict_action_chunk 替换为 HTTP POST 请求到服务端。
"""

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import requests
from PIL import Image
import io

device = "cpu"  # 本地只做数据处理和绘图，不做模型推理

# ==== 图像预处理 ====
def resize_img(img, size=(480, 640)):
    """
    img: Tensor (B, C, H, W) 或 (C, H, W)
    size: (H, W)
    """
    import torch
    if img.dim() == 3:
        img = img.unsqueeze(0)
        out = F.interpolate(img, size=size, mode="bilinear", align_corners=False)
        return out.squeeze(0)
    elif img.dim() == 4:
        return F.interpolate(img, size=size, mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"Unsupported shape {img.shape}")

def tensor_to_bytes(img_tensor):
    """
    将 Tensor (C,H,W) 或 (1,C,H,W) 转换为 PNG bytes
    """
    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    img_np = (img_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# ==== 调用 FastAPI 服务 ====
def predict_next_action_server(head_img, wrist_img, gelsight_img, qpos, server_url="http://192.168.1.100:8000/predict_next_action",task=""):
    """
    head_img, wrist_img, gelsight_img: Tensor (C,H,W) 或 (1,C,H,W)
    qpos: Tensor (1, 16) 或 list
    返回: 下一时刻动作 np.array
    """
    if torch.is_tensor(qpos):
        qpos_list = qpos.squeeze(0).cpu().numpy().tolist()
    else:
        qpos_list = list(qpos)

    files = {
        "head_img": tensor_to_bytes(head_img),
        "wrist_img": tensor_to_bytes(wrist_img),
        "gelsight_img": tensor_to_bytes(gelsight_img),
    }
    data = {"qpos": ",".join([str(x) for x in qpos_list]), "task": task}

    resp = requests.post(server_url, files=files, data=data)
    if resp.status_code == 200:
        return np.array(resp.json()["next_action"])
    else:
        raise RuntimeError(f"Server error: {resp.text}")

# ==== 计算多步误差 ====
def run_inference_chunk_server(hdf5_path, N=10, metric="RMSE", server_url="http://192.168.1.100:8000/predict_next_action", save_path="loss_plot.png"):
    import torch

    with h5py.File(hdf5_path, "r") as f:
        actions = torch.tensor(f["action"][:], dtype=torch.float32)
        qpos = torch.tensor(f["observations/qpos"][:], dtype=torch.float32)
        head_images = torch.tensor(f["observations/images/front"][:], dtype=torch.float32).permute(0,3,1,2) / 255
        wrist_images = torch.tensor(f["observations/images/left_wrist"][:], dtype=torch.float32).permute(0,3,1,2) / 255
        gelsight_images = torch.tensor(f["observations/images/left_gelsight_with_marker"][:], dtype=torch.float32).permute(0,3,1,2) / 255

    T = len(qpos)
    metric_list = []

    for t in range(T - N):
        batch_qpos = qpos[t].unsqueeze(0)
        head_img = resize_img(head_images[t].unsqueeze(0))
        wrist_img = resize_img(wrist_images[t].unsqueeze(0))
        gelsight_img = resize_img(gelsight_images[t].unsqueeze(0))

        # 调用远程服务器预测下一步动作
        pred_actions = []
        current_qpos = batch_qpos.clone()
        
        pred_actions = predict_next_action_server(
            head_img, 
            wrist_img, 
            gelsight_img, 
            current_qpos, 
            server_url=server_url, 
            task ="place a peg in a hole"
        )

        pred_actions = torch.tensor(pred_actions[:N]) # 只取前 N 步计算预测误差
        target_actions = actions[t:t+N]

        if metric.upper() == "MSE":
            value = torch.mean((pred_actions - target_actions) ** 2).item()
        elif metric.upper() == "RMSE":
            value = torch.sqrt(torch.mean((pred_actions - target_actions) ** 2)).item()
        elif metric.upper() == "MAE":
            value = torch.mean(torch.abs(pred_actions - target_actions)).item()
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        metric_list.append(value)
        print(f"Start {t}: {metric.upper()} over {N} steps = {value:.6f}")

    # 绘图
    plt.figure(figsize=(10,4))
    plt.plot(metric_list, label=f"{metric.upper()} {N}-step")
    plt.xlabel("Start Frame")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} over {N}-Step Prediction (Server)")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

    avg_value = sum(metric_list) / len(metric_list)
    print(f"Average {metric.upper()} over dataset ({N}-step): {avg_value:.6f}")

    return metric_list

def debug(host="0.0.0.0", port=5678, wait_client=True):
    import debugpy
    debugpy.listen((host, port))
    print(f"✅ debugpy is listening on {host}:{port}")
    if wait_client:
        print("⏳ Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print("🟢 Debugger attached, execution will continue.")

# ==== 主程序 ====
if __name__ == "__main__":
    hdf5_path = "data/peg-in-hole/episode_0.hdf5"
    server_url = "http://127.0.0.1:8000/predict_next_action"

    # debug()
    run_inference_chunk_server(hdf5_path, N=4, metric="RMSE", server_url=server_url)
