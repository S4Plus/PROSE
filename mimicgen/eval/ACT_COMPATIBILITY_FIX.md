# ACT Client & Server 兼容性修复总结

## 问题

ACT client 在连接 server 时遇到 400 错误,原因是数据格式不匹配:
- Client 发送的格式不符合 server 期望
- Server 期望: `{"images": {...}, "state": [...]}`  
- Client 之前发送: `{"obs": {...}}`

## 解决方案

采用了**双向修改**的方案,确保最大兼容性:

### 1. 修改 Client (`client_mimicgen_act.py`)

**添加新功能:**
- ✅ 添加 `cv2` 导入
- ✅ 添加 `_encode_image_to_base64()` 函数 - 将图像编码为 JPEG base64
- ✅ 添加 `_format_act_payload()` 函数 - 将 robosuite 观测转换为 ACT server 格式
  - 映射相机名称: `agentview_image` → `observation.images.agentview`
  - 映射相机名称: `robot0_eye_in_hand_image` → `observation.images.wrist`
  - 提取状态: `robot0_joint_pos` + `robot0_gripper_qpos`
  
**修改 rollout_episode:**
```python
# 旧代码
payload = {
    "obs": encode_item(obs_payload, args.obs_format),
    ...
}

# 新代码
act_payload = _format_act_payload(obs)
payload = {
    "images": act_payload["images"],
    "state": act_payload["state"],  # 如果有
    ...
}
```

### 2. 修改 Server (`server_act.py`)

**添加向后兼容逻辑:**
```python
def infer(self, payload: Dict[str, object]) -> Dict[str, object]:
    # 处理从 mimicgen client 来的请求格式
    # 如果有 "obs" 字段,从中提取 images 和 state
    if "obs" in payload and "images" not in payload:
        obs = payload.get("obs")
        if isinstance(obs, dict):
            # 提取 images
            images_dict = {}
            for key in self.image_keys:
                if key in obs:
                    images_dict[key] = obs[key]
            payload["images"] = images_dict
            
            # 提取 state (如果需要)
            if self.expect_state and "state" not in payload:
                if "observation.state" in obs:
                    payload["state"] = obs["observation.state"]
                elif "state" in obs:
                    payload["state"] = obs["state"]
    
    # 继续原有的处理逻辑...
```

**同时支持两个端点:**
```python
# 支持 /infer 和 /act 两个端点
if self.path not in ("/infer", "/act"):
    self._send_json(404, {"error": "Not found"})
    return
```

## 使用方法

### 重启 Server

修改后必须重启 server:

```bash
# 停止旧的 server (Ctrl+C)
python server_act.py --checkpoint ./outputs/square_d1/checkpoints/100000/pretrained_model/
```

### 运行 Client

```bash
# 基本测试
python client_mimicgen_act.py --video_dir ./act

# 保存 rollouts
python client_mimicgen_act.py \
    --num_episodes 50 \
    --save_rollouts_dir ./eval_rollouts_act \
    --use_camera_obs
```

## 兼容性

✅ 新 Client → 新 Server (推荐)
✅ 旧 Client → 新 Server (向后兼容)
✅ 支持 `/act` 和 `/infer` 两个端点

## 数据流程

```
Robosuite Env
    ↓ (obs)
_format_act_payload()
    ↓ {"images": {base64}, "state": [...]}
HTTP Request
    ↓
Server.infer()
    ↓ (decode images, process)
ACT Policy
    ↓
Actions
```

## 测试

重启 server 后运行 client,应该不再出现 400 错误。
