# ACT Server 端点修改说明

## 修改内容

修改了 `/workspace/liushuo/cupid/mimicgen/eval/server_act.py`,现在同时支持两个推理端点:

- `/act` - 原有端点(向后兼容)
- `/infer` - 新增端点(与标准命名一致)

## 修改代码

```python
# 支持 /infer 和 /act 两个端点
if self.path not in ("/infer", "/act"):
    self._send_json(404, {"error": "Not found"})
    return
```

## 使用方法

现在 client 可以使用任一端点:

```bash
# 使用 /act 端点
python client_mimicgen_act.py --server_url http://127.0.0.1:8000/act

# 或使用 /infer 端点
python client_mimicgen_act.py --server_url http://127.0.0.1:8000/infer
```

## 重启 Server

修改后需要重启 server 使更改生效:

```bash
# 停止旧的 server (Ctrl+C)
# 然后重新启动
python server_act.py --checkpoint <checkpoint_path>
```

默认的 client URL 已设置为 `http://127.0.0.1:8000/act`,无需额外配置即可使用。
