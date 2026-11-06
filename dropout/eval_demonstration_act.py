from __future__ import annotations

"""Demonstration-scoring pipeline for ACT/LeRobot policies.

This script mirrors ``eval_demonstration_lerobot.py`` but configures the
default load keys and experiment naming for ACT checkpoints saved via the
helpers in ``act/``.  It reuses the underlying scoring utilities while
pointing them at ACT-produced loss, variance, embedding, and TRAK artifacts.
 python act/eval_demonstration_act.py \
    --exp_name exp_act_eval \
    --eval_dir /root/eval_belt_pick_bottle \
    --pretrained_dir /root/pretrained_model \
    --train_dataset_path /root/.cache/huggingface/lerobot/ls/Static_Pick_Bottle_400\
    --result_date 2024-05-01 

"""

from typing import Any, Dict, Optional

import sys
import pathlib as _pathlib


# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

REPO_ROOT = _pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import pathlib
import pickle
import random

import click
import numpy as np
import torch

from act import scoring_utils as scoring
from train_trak_act import load_act_policy


ACT_LOAD_KEY = "default_act"
if ACT_LOAD_KEY not in scoring.LOAD_POLICY_LOSS_KWARGS:
    policy_kwargs = dict(scoring.LOAD_POLICY_LOSS_KWARGS["default_diffusion"])
    policy_kwargs["exp_type"] = "act"
    scoring.LOAD_POLICY_LOSS_KWARGS[ACT_LOAD_KEY] = policy_kwargs

if ACT_LOAD_KEY not in scoring.LOAD_ACTION_VARIANCE_KWARGS:
    action_kwargs = dict(scoring.LOAD_ACTION_VARIANCE_KWARGS["default_diffusion"])
    action_kwargs["exp_type"] = "act"
    scoring.LOAD_ACTION_VARIANCE_KWARGS[ACT_LOAD_KEY] = action_kwargs

if ACT_LOAD_KEY not in scoring.LOAD_TRAK_KWARGS:
    trak_kwargs = dict(scoring.LOAD_TRAK_KWARGS["default_diffusion"])
    trak_kwargs["exp_type"] = "act"
    scoring.LOAD_TRAK_KWARGS[ACT_LOAD_KEY] = trak_kwargs


def _get_trak_scores_act(
    eval_dir: pathlib.Path,
    train_set_size: int,
    test_set_size: int,
    exp_date: str,
    return_dtype: type,
    **kwargs,
) -> np.ndarray:
    exp_name = (
        f"{exp_date}_trak_results-proj_dim={kwargs['proj_dim']}"
        f"-lambda_reg={kwargs['lambda_reg']}-num_ckpts={kwargs['num_ckpts']}-seed={kwargs['seed']}"
    )
    exp_dir = eval_dir / exp_name
    trak_exp = kwargs.get("trak_exp_name", "all_episodes")
    scores_path = "/workspace/liushuo/cupid/eval_belt_pick_bottle/act_trak_run/scores/all_episodes.mmap"#exp_dir / "scores" / f"{trak_exp}.mmap"
    scores = scoring.load_mmap(scores_path, (train_set_size, test_set_size), kwargs["dtype"])
    return scores.T.astype(return_dtype)


scoring.get_trak_scores = _get_trak_scores_act


def _infer_policy_embedding_dim(policy: Any) -> Optional[int]:
    """Best-effort inference of the policy embedding dimension for ACT."""

    for attr in ("emb_dim", "latent_dim"):
        value = getattr(policy, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    config = getattr(policy, "config", None)
    if config is not None:
        for attr in ("dim_model", "latent_dim"):
            value = getattr(config, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        robot_state = getattr(config, "robot_state_feature", None)
        if robot_state is not None and hasattr(robot_state, "shape"):
            state_dim = int(np.prod(robot_state.shape))
            if state_dim > 0:
                return state_dim
    return None


@click.command()
@click.option("--exp_name", type=str, required=True)
@click.option("--eval_dir", type=str, required=True)
@click.option("--pretrained_dir", type=str, required=True)
@click.option("--train_dataset_path", type=str, required=True)
@click.option("--holdout_dataset_path", type=str, default=None)
@click.option("--result_date", type=str, required=True)
@click.option("--overwrite", type=bool, default=False)
@click.option("--device", type=str, default="cuda:0")
@click.option("--seed", type=int, default=0)
@click.option("--use_half_precision", type=bool, default=False)
@click.option("--compute_holdout", type=bool, default=False)
@click.option("--policy_embedding_dim", type=int, default=None)
@click.option("--state_embedding_load_key", type=str, default="default_image")
@click.option("--eval_offline_policy_loss", type=bool, required=False)
@click.option("--eval_offline_action_diversity", type=bool, required=False,default=False)
@click.option("--eval_offline_state_diversity", type=bool, required=False,default=False)
@click.option("--eval_online_state_similarity", type=bool, required=False,default=False)
@click.option("--eval_online_demo_score", type=bool, required=False,default=False)
@click.option("--eval_online_trak_influence", type=bool, required=False,default=True)
def main(
    exp_name: str,
    eval_dir: str,
    pretrained_dir: str,
    train_dataset_path: str,
    holdout_dataset_path: Optional[str],
    result_date: str,
    overwrite: bool,
    device: str,
    seed: int,
    use_half_precision: bool,
    compute_holdout: bool,
    policy_embedding_dim: Optional[int],
    state_embedding_load_key: str,
    eval_offline_policy_loss: bool,
    eval_offline_action_diversity: bool,
    eval_offline_state_diversity: bool,
    eval_online_state_similarity: bool,
    eval_online_demo_score: bool,
    eval_online_trak_influence: bool,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device_t = torch.device(device)

    pretrained_path = pathlib.Path(pretrained_dir).expanduser()
    policy, _ = load_act_policy(pretrained_path, device_t)
    if policy_embedding_dim is None:
        policy_embedding_dim = _infer_policy_embedding_dim(policy)

    train_set_metadata = scoring.load_dataset_metadata(pathlib.Path(train_dataset_path))
    if compute_holdout:
        if holdout_dataset_path is None:
            raise ValueError("Provide --holdout_dataset_path when compute_holdout=True")
        holdout_set_metadata = scoring.load_dataset_metadata(pathlib.Path(holdout_dataset_path))
    else:
        holdout_set_metadata = scoring.build_metadata_from_lengths([])

    test_set = scoring.EvalPklEpisodeDataset(
        dataset_path=pathlib.Path(eval_dir) / "episodes",
        exec_horizon=1,
        sample_history=0,
    )
    test_set_metadata = scoring.build_metadata_from_lengths(
        test_set.episode_lengths,
        success_mask=test_set.episode_successes,
    )
    del test_set

    save_dir = pathlib.Path(eval_dir) / exp_name
    save_dir.mkdir(exist_ok=True, parents=True)
    return_dtype = np.float16 if use_half_precision else np.float32

    def save_result(result: Dict[str, Any], result_name: str) -> None:
        save_path = save_dir / f"{result_name}.pkl"
        if save_path.exists() and not overwrite:
            raise ValueError(f"Output path {save_path} already exists.")
        with open(save_path, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Results saved [{result_name}]")

    if eval_offline_policy_loss:
        save_result(
            scoring.offline_policy_loss_method(
                load_key=ACT_LOAD_KEY,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            "offline_policy_loss",
        )

    if eval_offline_action_diversity:
        save_result(
            scoring.offline_action_diversity_method(
                load_key=ACT_LOAD_KEY,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            "offline_action_diversity",
        )

    if eval_offline_state_diversity:
        if policy_embedding_dim is None:
            raise ValueError("policy_embedding_dim is required for offline_state_diversity.")
        save_result(
            scoring.offline_state_diversity_method(
                load_key=state_embedding_load_key,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
                policy_embedding_dim=policy_embedding_dim,
            ),
            "offline_state_diversity",
        )

    if eval_online_state_similarity:
        if policy_embedding_dim is None:
            raise ValueError("policy_embedding_dim is required for online_state_similarity.")
        save_result(
            scoring.online_state_similarity_method(
                load_key=state_embedding_load_key,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
                policy_embedding_dim=policy_embedding_dim,
            ),
            "online_state_similarity",
        )

    if eval_online_demo_score:
        save_result(
            scoring.online_demo_score_method(
                load_key="default",
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            "online_demo_score",
        )

    if eval_online_trak_influence:
        save_result(
            scoring.online_trak_influence_method(
                load_key=ACT_LOAD_KEY,
                eval_dir=eval_dir,
                exp_date=result_date,
                compute_holdout=compute_holdout,
                return_dtype=return_dtype,
                train_set_metadata=train_set_metadata,
                holdout_set_metadata=holdout_set_metadata,
                test_set_metadata=test_set_metadata,
            ),
            "online_trak_influence",
        )

    print(f"Experiment Complete: All results saved to {save_dir}.")


if __name__ == "__main__":
    main()
