#!/usr/bin/env python3
"""对齐偏好组和随机组的 can_ph sample 数，保证最终训练样本数一致。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对齐两组 can_ph 补充 sample 数。")
    parser.add_argument("--pref-parts", type=Path, required=True)
    parser.add_argument("--rand-parts", type=Path, required=True)
    parser.add_argument("--pref-output", type=Path, required=True)
    parser.add_argument("--rand-output", type=Path, required=True)
    parser.add_argument("--pref-summary", type=Path, required=True)
    parser.add_argument("--rand-summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_parts(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        can_mh = np.asarray(data["can_mh_sample_indices"], dtype=np.int64)
        can_ph = np.asarray(data["can_ph_sample_indices"], dtype=np.int64)
    return can_mh, can_ph


def choose_subset(values: np.ndarray, target_count: int, seed: int) -> np.ndarray:
    if values.shape[0] <= target_count:
        return np.sort(values)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(values, size=target_count, replace=False)
    return np.sort(chosen.astype(np.int64))


def write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    pref_can_mh, pref_can_ph = load_parts(args.pref_parts)
    rand_can_mh, rand_can_ph = load_parts(args.rand_parts)
    if pref_can_mh.shape[0] != rand_can_mh.shape[0]:
        raise ValueError(
            f"can_mh 样本数不一致，无法公平对齐: pref={pref_can_mh.shape[0]}, rand={rand_can_mh.shape[0]}"
        )

    target_can_ph = int(min(pref_can_ph.shape[0], rand_can_ph.shape[0]))
    pref_can_ph_eq = choose_subset(pref_can_ph, target_can_ph, args.seed)
    rand_can_ph_eq = choose_subset(rand_can_ph, target_can_ph, args.seed + 1)

    pref_final = np.sort(np.concatenate([pref_can_mh, pref_can_ph_eq]).astype(np.int64))
    rand_final = np.sort(np.concatenate([rand_can_mh, rand_can_ph_eq]).astype(np.int64))
    np.save(args.pref_output, pref_final)
    np.save(args.rand_output, rand_final)

    pref_summary = {
        "mode": "preference",
        "pref_parts": str(args.pref_parts),
        "target_can_ph_samples": target_can_ph,
        "kept_can_mh_samples": int(pref_can_mh.shape[0]),
        "kept_can_ph_samples_before_equalize": int(pref_can_ph.shape[0]),
        "kept_can_ph_samples_after_equalize": int(pref_can_ph_eq.shape[0]),
        "final_total_samples": int(pref_final.shape[0]),
        "final_indices_path": str(args.pref_output),
    }
    rand_summary = {
        "mode": "random",
        "rand_parts": str(args.rand_parts),
        "target_can_ph_samples": target_can_ph,
        "kept_can_mh_samples": int(rand_can_mh.shape[0]),
        "kept_can_ph_samples_before_equalize": int(rand_can_ph.shape[0]),
        "kept_can_ph_samples_after_equalize": int(rand_can_ph_eq.shape[0]),
        "final_total_samples": int(rand_final.shape[0]),
        "final_indices_path": str(args.rand_output),
    }
    write_summary(args.pref_summary, pref_summary)
    write_summary(args.rand_summary, rand_summary)

    print(json.dumps({"pref": pref_summary, "rand": rand_summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
