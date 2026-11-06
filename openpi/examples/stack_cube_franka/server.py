import dataclasses
import logging
import socket
from pathlib import Path

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """stack_cube_franka 专用 policy server 参数。"""

    checkpoint_dir: Path
    config_name: str = "pi05_stack_cube_franka_lora"
    host: str = "0.0.0.0"
    port: int = 8000
    default_prompt: str = "stack cube"
    record: bool = False
    pytorch_device: str | None = None


def create_policy(args: Args) -> _policy.Policy:
    """加载 stack_cube_franka 训练好的策略。"""
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")
    return policy


def main(args: Args) -> None:
    policy = create_policy(args)
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Starting stack_cube_franka server on %s:%d (host=%s, ip=%s)", args.host, args.port, hostname, local_ip)
    logging.info("Loaded config=%s checkpoint=%s", args.config_name, args.checkpoint_dir)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
