import asyncio
import http
import logging
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def _log_action_summary(action: dict, remote_address: object) -> None:
    """打印服务端推理出的 action 与 gripper width 摘要，便于区分服务端/客户端问题。"""
    if not isinstance(action, dict) or "actions" not in action:
        logger.info("Inference result for %s has no `actions` field.", remote_address)
        return

    try:
        actions = np.asarray(action["actions"], dtype=np.float32)
    except Exception as exc:
        logger.warning("Failed to summarize actions for %s: %s", remote_address, exc)
        return

    if actions.ndim != 2:
        logger.info("Inference result for %s has actions with unexpected shape=%s", remote_address, actions.shape)
        return

    logger.info(
        "Server action summary for %s: shape=%s first_action=%s",
        remote_address,
        tuple(actions.shape),
        np.round(actions[0], 6).tolist() if len(actions) > 0 else [],
    )
    logger.info(
        "Server action chunk for %s: actions=%s",
        remote_address,
        np.round(actions, 6).tolist(),
    )

    if actions.shape[1] >= 8:
        gripper_width = np.asarray(actions[:, 7], dtype=np.float32)
        logger.info(
            "Server gripper width for %s: all20=%s min=%.6f max=%.6f mean=%.6f",
            remote_address,
            np.round(gripper_width, 6).tolist(),
            float(gripper_width.min()),
            float(gripper_width.max()),
            float(gripper_width.mean()),
        )
    else:
        logger.info(
            "Server actions for %s have only %d dims, no gripper width at dim 7.",
            remote_address,
            actions.shape[1],
        )


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time
                _log_action_summary(action, websocket.remote_address)

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
