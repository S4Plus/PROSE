import dataclasses
import logging
import pathlib
import sys
from pathlib import Path


# Ensure local openpi_client package is importable when running from repo root.
ROOT = Path(__file__).resolve().parents[2]
CLIENT_SRC = ROOT / "packages" / "openpi-client" / "src"
if str(CLIENT_SRC) not in sys.path:
    sys.path.insert(0, str(CLIENT_SRC))

import env as _env
from lerobot.envs.configs import MimicGenRobosuiteEnv
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
import tyro


@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/mimicgen/videos")
    actions_out_dir: pathlib.Path | None = None
    rollout_out_dir: pathlib.Path | None = None

    task: str = "Square_D1"
    seed: int = 0
    episode_length: int = 250
    fps: int = 20
    control_mode: str = "JOINT_POSITION"
    robots: str = "Panda"

    action_horizon: int = 20

    host: str = "0.0.0.0"
    port: int = 8000

    num_episodes: int = 100
    display: bool = False
    bgr_to_rgb: bool = False
    flip_vertical: bool = True
    terminate_on_success: bool = True


def main(args: Args) -> None:
    env_cfg = MimicGenRobosuiteEnv(
        task=args.task,
        episode_length=args.episode_length,
        fps=args.fps,
        render=args.display,
        control_mode=args.control_mode,
        robots=args.robots,
    )
    # Allow correcting BGR-rendered images to RGB if needed.
    setattr(env_cfg, "bgr_to_rgb", args.bgr_to_rgb)
    # Allow disabling vertical flip for renderers that are already upright.
    setattr(env_cfg, "flip_vertical", args.flip_vertical)
    # Optionally stop the episode as soon as success is reached.
    setattr(env_cfg, "terminate_on_success", args.terminate_on_success)

    env = _env.MimicGenGymEnv(env_cfg)
    rollout_dir = args.rollout_out_dir or args.out_dir.parent / "rollouts"

    runtime = _runtime.Runtime(
        environment=env,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[
            _saver.VideoSaver(args.out_dir),
            _saver.ActionSaver(args.actions_out_dir or args.out_dir.parent / "actions"),
            _saver.RolloutSaver(rollout_dir, env),
        ],
        max_hz=args.fps,
        num_episodes=args.num_episodes,
        max_episode_steps=args.episode_length,
    )

    episode_results = runtime.run()
    success_count = sum(1 for res in episode_results if res.get("is_success"))
    total_episodes = len(episode_results)
    success_rate = (success_count / total_episodes) if total_episodes else 0.0

    for idx, res in enumerate(episode_results, start=1):
        logging.info(
            "Episode %d summary: success=%s, steps=%s",
            idx,
            bool(res.get("is_success")),
            res.get("episode_steps"),
        )

    logging.info(
        "Eval complete: %d/%d successes (%.1f%%)",
        success_count,
        total_episodes,
        success_rate * 100,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
