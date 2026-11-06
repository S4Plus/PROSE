# server.py
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn.functional as F
from lerobot.policies.act.modeling_act import ACTPolicy
# from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def debug(host="0.0.0.0", port=5678, wait_client=True):
    import debugpy
    debugpy.listen((host, port))
    print(f"✅ debugpy is listening on {host}:{port}")
    if wait_client:
        print("⏳ Waiting for debugger to attach...")
        debugpy.wait_for_client()
        print(" Debugger attached, execution will continue.")

# debug()    
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="ACTPolicy Inference API")

# ====== 辅助函数 ======
def preprocess_image(file: UploadFile, size=(480, 640)):
    """
    读取上传的图片，转换为Tensor并归一化
    返回: (1, C, H, W) Tensor
    """
    img = Image.open(file.file).convert("RGB")
    img = img.resize(size[::-1], Image.BILINEAR)
    img = torch.tensor(np.array(img), dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0
    return img.to(device)

def preprocess_qpos(qpos_list):
    """
    qpos_list: List[float] 或 np.array
    返回: (1, len) Tensor
    """
    qpos = torch.tensor(qpos_list, dtype=torch.float32).unsqueeze(0)
    return qpos.to(device)

def resize_img_tensor(img_tensor, size=(360, 640)):
    """
    img_tensor: (B, C, H, W) 或 (C, H, W)
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        out = F.interpolate(img_tensor, size=size, mode="bilinear", align_corners=False)
        return out.squeeze(0)
    elif img_tensor.dim() == 4:
        return F.interpolate(img_tensor, size=size, mode="bilinear", align_corners=False)
    else:
        raise ValueError(f"Unsupported shape {img_tensor.shape}")

# ====== 加载策略模型 ======

policy = ACTPolicy.from_pretrained(
    "outputs/train/smothVLA/peg-in-hole-sim/checkpoints/080000/pretrained_model"
).to(device)
policy.eval()

# ====== API 接口 ======
@app.post("/predict_next_action")
async def predict_next_action(
    head_img: UploadFile = File(...),
    wrist_img: UploadFile = File(...),
    gelsight_img: UploadFile = File(...),
    qpos: str = Form(...),
    task: str = Form(...),
):
    """
    qpos: JSON 字符串或者逗号分隔的关节角列表，例如 "0.0,0.1,0.2,..."
    """
    try:
        # 1. 处理图像
        
        head_tensor = preprocess_image(head_img)
        wrist_tensor = preprocess_image(wrist_img)
        gelsight_tensor = preprocess_image(gelsight_img)

        # 2. 处理 qpos
        if "," in qpos:
            qpos_list = [float(x.strip()) for x in qpos.split(",")]
        else:
            import json
            qpos_list = json.loads(qpos)
        qpos_tensor = preprocess_qpos(qpos_list)

        # 3. 构造 batch
        batch = {
            "observation.state": qpos_tensor,
            "observation.images.front": resize_img_tensor(head_tensor),
            "observation.images.left_wrist": resize_img_tensor(wrist_tensor),
            # "observation.images.left_gelsight_with_marker": resize_img_tensor(gelsight_tensor),
            "task": task
        }

        # 4. 推理
        with torch.no_grad():
            # next_action = policy.predict_action_chunk(batch).squeeze(0)[0]  # 取下一时刻动作
            next_action = policy.predict_action_chunk(batch).squeeze(0)  # 取所有动作
            next_action_list = next_action.cpu().numpy().tolist()

        return JSONResponse({"next_action": next_action_list})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

# ====== 运行命令 ======
# uvicorn api_server:app --host 0.0.0.0 --port 8000
