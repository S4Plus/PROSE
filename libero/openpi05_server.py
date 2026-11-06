#!/usr/bin/env python3
"""启动 OpenPI05 的 LIBERO websocket 推理服务。"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = ROOT / "openpi"
OPENPI_SRC = OPENPI_ROOT / "src"

if str(OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(OPENPI_SRC))


DEFAULT_CONFIG = "pi05_libero"
DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 OpenPI05 LIBERO websocket server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="OpenPI 训练配置名")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="checkpoint 目录，可为本地路径或 gs:// 路径",
    )
    parser.add_argument(
        "--default-prompt",
        type=str,
        default=None,
        help="当客户端未显式传 prompt 时使用的默认任务文本",
    )
    parser.add_argument("--record", action="store_true", help="是否记录 policy 行为用于调试")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from openpi.policies import policy as _policy
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config),
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Starting OpenPI05 LIBERO server on %s:%s", args.host, args.port)
    logging.info("Host machine: %s (%s)", hostname, local_ip)
    logging.info("Policy config: %s", args.config)
    logging.info("Checkpoint dir: %s", args.checkpoint_dir)
    logging.info("Policy metadata: %s", metadata)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
