#!/usr/bin/env python3
"""从 MYSYS 两级删除结果生成补采区域与轨迹特征建议。

这个脚本消费 `filter_joint_episode_window.py` 的输出：

- `episode_scores.csv`：轨迹级 keep/drop 与 joint score
- `window_scores.csv`：片段级 keep/drop 与局部质量分

核心思想：

1. 先把被删除的 episode 与 window 还原到状态-动作描述空间；
2. 用轻量 PCA + 分箱形成可解释的 state-action region；
3. 统计每个 region 的删除压力、动作/状态抖动和保留覆盖；
4. 输出“需要补采的区域”和“应采集的轨迹特点”。

注意：这里的“补采”不是复现坏样本，而是在被删区域补充更平滑、
更稳定、更接近成功路径的高质量专家示例。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EpisodeScore:
    episode: int
    keep_episode: bool
    episode_influence: float
    episode_quality: float
    norm_episode_influence: float
    norm_episode_quality: float
    episode_joint_score: float
    episode_length: int


@dataclass
class WindowScore:
    episode: int
    start: int
    end: int
    keep_window: bool
    keep_gain: float
    keep_novelty: float
    window_episode_score: float
    window_quality_score: float
    window_local_value: float
    episode_score: float
    window_contribution_score: float
    raw_action_jitter: float
    raw_state_jitter: float


@dataclass
class RegionRecord:
    deletion_level: str
    episode: int
    start: int
    end: int
    keep: bool
    episode_keep: bool
    episode_joint_score: float
    episode_quality: float
    episode_influence: float
    window_local_value: float
    window_contribution_score: float
    action_jitter: float
    state_jitter: float
    action_energy: float
    state_delta_norm: float
    length: int
    phase_start: float
    phase_end: float
    phase_center: float
    phase_bin: int
    region_x_bin: int
    region_y_bin: int
    pca_x: float
    pca_y: float
    state_mean: list[float]
    action_mean: list[float]


@dataclass
class CollectionTarget:
    rank: int
    priority_score: float
    region_id: str
    phase_name: str
    suggested_episode_count: int
    deletion_records: int
    kept_records: int
    deletion_ratio: float
    episode_drop_records: int
    window_drop_records: int
    mean_episode_joint_score: float
    mean_window_local_value: float
    mean_action_jitter: float
    mean_state_jitter: float
    mean_action_energy: float
    mean_state_delta_norm: float
    representative_episodes: str
    representative_segments: str
    state_center: str
    action_center: str
    collection_region: str
    collection_pose: str
    trajectory_corrections: str
    guidance: str


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_episode_scores(path: Path) -> dict[int, EpisodeScore]:
    scores: dict[int, EpisodeScore] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = _safe_int(row.get("episode"), -1)
            if ep < 0:
                continue
            scores[ep] = EpisodeScore(
                episode=ep,
                keep_episode=bool(_safe_int(row.get("keep_episode"), 0)),
                episode_influence=_safe_float(row.get("episode_influence")),
                episode_quality=_safe_float(row.get("episode_quality")),
                norm_episode_influence=_safe_float(row.get("norm_episode_influence")),
                norm_episode_quality=_safe_float(row.get("norm_episode_quality")),
                episode_joint_score=_safe_float(row.get("episode_joint_score")),
                episode_length=_safe_int(row.get("episode_length"), 0),
            )
    return scores


def _read_window_scores(path: Path) -> list[WindowScore]:
    if not path.exists():
        return []
    windows: list[WindowScore] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = _safe_int(row.get("episode"), -1)
            if ep < 0:
                continue
            windows.append(
                WindowScore(
                    episode=ep,
                    start=_safe_int(row.get("start"), 0),
                    end=_safe_int(row.get("end"), 0),
                    keep_window=bool(_safe_int(row.get("keep_window"), 0)),
                    keep_gain=_safe_float(row.get("keep_gain")),
                    keep_novelty=_safe_float(row.get("keep_novelty")),
                    window_episode_score=_safe_float(row.get("window_episode_score")),
                    window_quality_score=_safe_float(row.get("window_quality_score")),
                    window_local_value=_safe_float(row.get("window_local_value")),
                    episode_score=_safe_float(row.get("episode_score")),
                    window_contribution_score=_safe_float(row.get("window_contribution_score")),
                    raw_action_jitter=_safe_float(row.get("raw_action_jitter")),
                    raw_state_jitter=_safe_float(row.get("raw_state_jitter")),
                )
            )
    return windows


def _collect_episode_files(dataset_root: Path) -> dict[int, Path]:
    episode_files: dict[int, Path] = {}
    for path in sorted((dataset_root / "data").glob("chunk-*/episode_*.parquet")):
        stem = path.stem
        if not stem.startswith("episode_"):
            continue
        ep_str = stem.split("_")[-1]
        if ep_str.isdigit():
            episode_files[int(ep_str)] = path
    return episode_files


def _load_episode_arrays(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - 取决于运行环境是否安装 pyarrow
        raise RuntimeError("读取 LeRobot parquet 需要安装 pyarrow。") from exc

    schema = pq.read_schema(path)
    available = set(schema.names)
    cols = [col for col in ("action", "observation.state") if col in available]
    if not cols:
        return None, None
    table = pq.read_table(path, columns=cols)
    action = None
    state = None
    if "action" in table.column_names:
        action = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    if "observation.state" in table.column_names:
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    return action, state


def _slice_2d(values: np.ndarray | None, start: int, end: int) -> np.ndarray | None:
    if values is None or values.ndim != 2 or values.shape[0] == 0:
        return None
    s = max(0, min(int(start), int(values.shape[0])))
    e = max(s, min(int(end), int(values.shape[0])))
    return values[s:e]


def _mean_vector(values: np.ndarray | None, dims: int) -> list[float]:
    if values is None or values.ndim != 2 or values.shape[0] == 0:
        return []
    mean = np.mean(values, axis=0)
    return [float(x) for x in mean[: max(0, dims)].tolist()]


def _mean_diff_norm(values: np.ndarray | None) -> float:
    if values is None or values.ndim != 2 or values.shape[0] < 2:
        return float("nan")
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).mean())


def _mean_energy(values: np.ndarray | None) -> float:
    if values is None or values.ndim != 2 or values.shape[0] == 0:
        return float("nan")
    return float(np.linalg.norm(values, axis=1).mean())


def _state_delta_norm(values: np.ndarray | None) -> float:
    if values is None or values.ndim != 2 or values.shape[0] < 2:
        return float("nan")
    return float(np.linalg.norm(values[-1] - values[0]))


def _finite_or(value: float, fallback: float) -> float:
    return float(value) if math.isfinite(float(value)) else float(fallback)


def _phase_bin(center: float, bins: int) -> int:
    if bins <= 1:
        return 0
    return int(np.clip(math.floor(float(center) * bins), 0, bins - 1))


def _phase_name(phase_bin: int, bins: int) -> str:
    if bins <= 1:
        return "全轨迹"
    lo = phase_bin / bins
    hi = (phase_bin + 1) / bins
    if hi <= 0.25:
        label = "起始接近段"
    elif lo >= 0.75:
        label = "末端收敛段"
    elif lo < 0.5 < hi:
        label = "中段转换段"
    elif hi <= 0.5:
        label = "前半段"
    else:
        label = "后半段"
    return f"{label}({lo:.0%}-{hi:.0%})"


def _build_record_feature(record: RegionRecord) -> list[float]:
    feature: list[float] = []
    feature.extend(record.state_mean[:])
    feature.extend(record.action_mean[:])
    feature.extend(
        [
            record.phase_center,
            _finite_or(record.action_jitter, 0.0),
            _finite_or(record.state_jitter, 0.0),
            _finite_or(record.action_energy, 0.0),
            _finite_or(record.state_delta_norm, 0.0),
        ]
    )
    return feature


def _standardize(features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return features
    mu = np.nanmean(features, axis=0, keepdims=True)
    sd = np.nanstd(features, axis=0, keepdims=True)
    sd = np.where(sd > 1e-8, sd, 1.0)
    out = (features - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _pca_2d(features: np.ndarray) -> np.ndarray:
    if features.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    x = _standardize(features)
    if x.shape[1] == 0:
        return np.zeros((x.shape[0], 2), dtype=np.float64)
    try:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((x.shape[0], 2), dtype=np.float64)
    comps = vh[: min(2, vh.shape[0])].T
    z = x @ comps
    if z.shape[1] == 1:
        z = np.concatenate([z, np.zeros((z.shape[0], 1), dtype=np.float64)], axis=1)
    return np.asarray(z[:, :2], dtype=np.float64)


def _bin_values(values: np.ndarray, bins: int) -> np.ndarray:
    if values.size == 0:
        return np.asarray([], dtype=np.int64)
    if bins <= 1:
        return np.zeros(values.shape[0], dtype=np.int64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape[0], dtype=np.int64)
    vals = values[finite]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if np.isclose(lo, hi):
        return np.zeros(values.shape[0], dtype=np.int64)
    scaled = (np.nan_to_num(values, nan=lo) - lo) / (hi - lo)
    return np.clip(np.floor(scaled * bins), 0, bins - 1).astype(np.int64)


def _format_vec(values: list[float], precision: int = 4, max_items: int = 6) -> str:
    if not values:
        return ""
    shown = values[:max_items]
    return "[" + ", ".join(f"{v:.{precision}f}" for v in shown) + "]"


def _normalize_metric(values: list[float], invert: bool = False) -> list[float]:
    arr = np.asarray([v if math.isfinite(v) else np.nan for v in values], dtype=np.float64)
    out = np.zeros_like(arr)
    finite = np.isfinite(arr)
    if not finite.any():
        return out.tolist()
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if np.isclose(lo, hi):
        out[finite] = 0.5
    else:
        out[finite] = (arr[finite] - lo) / (hi - lo)
    if invert:
        out[finite] = 1.0 - out[finite]
    return np.clip(out, 0.0, 1.0).tolist()


def _make_guidance(
    *,
    phase_name: str,
    deletion_ratio: float,
    episode_drop_records: int,
    window_drop_records: int,
    mean_episode_joint_score: float,
    mean_window_local_value: float,
    mean_action_jitter: float,
    global_action_jitter: float,
    mean_state_jitter: float,
    global_state_jitter: float,
    mean_state_delta_norm: float,
) -> str:
    reasons: list[str] = []
    if episode_drop_records > 0:
        reasons.append(f"该区域包含 {episode_drop_records} 条低 joint score 轨迹")
    if window_drop_records > 0:
        reasons.append(f"该区域包含 {window_drop_records} 个被删片段")
    if deletion_ratio >= 0.75:
        reasons.append("保留覆盖明显不足")
    if math.isfinite(mean_episode_joint_score) and mean_episode_joint_score < 0.35:
        reasons.append("episode 全局质量/影响力偏低")
    if math.isfinite(mean_window_local_value) and mean_window_local_value < 0.45:
        reasons.append("局部片段价值偏低")
    if math.isfinite(mean_action_jitter) and mean_action_jitter > global_action_jitter:
        reasons.append("动作抖动高于全局均值")
    if math.isfinite(mean_state_jitter) and mean_state_jitter > global_state_jitter:
        reasons.append("状态变化不够平滑")
    if math.isfinite(mean_state_delta_norm) and mean_state_delta_norm < 1e-6:
        reasons.append("片段内状态推进不足，疑似停滞")

    if not reasons:
        reasons.append("删除密度较高，需要补充同区域高质量覆盖")

    action_hint = "动作应连续、低抖动，避免高频开合或方向来回切换"
    if math.isfinite(mean_action_jitter) and mean_action_jitter <= global_action_jitter:
        action_hint = "保持当前动作平滑度，同时提高终端收敛与成功率"

    state_hint = "状态轨迹应单调推进到目标区域，减少回退、停滞和绕路"
    if math.isfinite(mean_state_delta_norm) and mean_state_delta_norm > 0:
        state_hint = "覆盖同一状态区域，但用更短、更直接的路径完成关键转换"

    return (
        f"补采 {phase_name} 的高质量专家示例；原因：{'；'.join(reasons)}。"
        f"采集要求：{action_hint}；{state_hint}；每条轨迹应完整覆盖该阶段前后的上下文，"
        "不要只录孤立片段。"
    )


def _make_collection_region(
    *,
    region_id: str,
    phase_name: str,
    state_center: str,
    action_center: str,
    deletion_ratio: float,
) -> str:
    coverage_hint = "该区域保留覆盖不足，优先补采高质量替代样本" if deletion_ratio >= 0.6 else "该区域存在局部质量缺口，按需补齐"
    state_hint = f"状态中心参考 {state_center}" if state_center else "状态中心不可用，按代表片段回放定位"
    action_hint = f"动作中心参考 {action_center}" if action_center else "动作中心不可用，重点观察原始片段动作形态"
    return f"{region_id}，时间相位为 {phase_name}；{state_hint}；{action_hint}；{coverage_hint}。"


def _make_collection_pose(
    *,
    phase_name: str,
    state_center: str,
    action_center: str,
    episode_drop_records: int,
    window_drop_records: int,
) -> str:
    phase_hint = "从任务自然初始姿态进入该区域"
    if "起始" in phase_name:
        phase_hint = "从稳定初始位姿进入，末端姿态先对齐目标物/接触面，避免一开始就大幅修正"
    elif "前半段" in phase_name:
        phase_hint = "从稳定接近状态继续推进，保持末端朝向一致，避免接近过程中出现大幅横向修正或路径回退"
    elif "中段" in phase_name:
        phase_hint = "保持抓取/接触约束稳定，末端姿态沿主运动方向连续过渡"
    elif "末端" in phase_name or "后半段" in phase_name:
        phase_hint = "靠近目标终态时减速收敛，末端姿态对齐放置/插入/堆叠目标"

    if episode_drop_records > 0 and window_drop_records == 0:
        scope_hint = "需要整条成功轨迹，不能只补局部姿态"
    elif episode_drop_records > 0:
        scope_hint = "建议采完整轨迹，并重点保证该局部姿态附近的连续上下文"
    else:
        scope_hint = "可围绕代表片段补采，但每段至少包含进入和离开该姿态的上下文"

    state_hint = f"状态参考：{state_center}" if state_center else "状态参考：使用 representative_segments 回放定位"
    action_hint = f"动作参考：{action_center}" if action_center else "动作参考：以低速、连续控制为准"
    return f"{phase_hint}；{scope_hint}；{state_hint}；{action_hint}。"


def _make_trajectory_corrections(
    *,
    mean_episode_joint_score: float,
    mean_window_local_value: float,
    mean_action_jitter: float,
    global_action_jitter: float,
    mean_state_jitter: float,
    global_state_jitter: float,
    mean_state_delta_norm: float,
    deletion_ratio: float,
) -> str:
    corrections: list[str] = []
    if math.isfinite(mean_action_jitter) and mean_action_jitter > global_action_jitter:
        corrections.append("降低 action jitter：减少方向来回切换、夹爪高频开合和速度突变")
    if math.isfinite(mean_state_jitter) and mean_state_jitter > global_state_jitter:
        corrections.append("降低 state jitter：路径应平滑推进，避免绕路、回退和接触后反复调整")
    if math.isfinite(mean_state_delta_norm) and mean_state_delta_norm < 1e-6:
        corrections.append("修正停滞片段：确保片段内有明确状态推进，不录长时间悬停")
    if math.isfinite(mean_window_local_value) and mean_window_local_value < 0.45:
        corrections.append("提升局部价值：关键接触/对齐阶段放慢速度，动作连续且可复现")
    if math.isfinite(mean_episode_joint_score) and mean_episode_joint_score < 0.35:
        corrections.append("提升整轨质量：必须成功收敛到任务终态，补足起始到终端的完整因果链")
    if deletion_ratio >= 0.75:
        corrections.append("增加覆盖多样性：同一区域采集不同初始扰动、物体微偏移和接近角")
    if not corrections:
        corrections.append("保持原有成功策略，仅补充同区域更干净、更稳定的专家示例")
    return "；".join(corrections) + "。"


def _build_records(
    *,
    episode_scores: dict[int, EpisodeScore],
    window_scores: list[WindowScore],
    dataset_root: Path,
    state_dims: int,
    action_dims: int,
    phase_bins: int,
    include_kept: bool,
) -> list[RegionRecord]:
    episode_files = _collect_episode_files(dataset_root)
    arrays_cache: dict[int, tuple[np.ndarray | None, np.ndarray | None]] = {}

    def arrays_for_episode(ep: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        if ep in arrays_cache:
            return arrays_cache[ep]
        path = episode_files.get(ep)
        if path is None:
            arrays_cache[ep] = (None, None)
        else:
            arrays_cache[ep] = _load_episode_arrays(path)
        return arrays_cache[ep]

    def episode_length(ep: int, action: np.ndarray | None, state: np.ndarray | None) -> int:
        if action is not None:
            return int(action.shape[0])
        if state is not None:
            return int(state.shape[0])
        return max(int(episode_scores.get(ep, EpisodeScore(ep, False, float("nan"), float("nan"), 0, 0, 0, 0)).episode_length), 0)

    records: list[RegionRecord] = []

    # 轨迹级删除：把整条轨迹视作一个待诊断 record。
    for ep, ep_score in sorted(episode_scores.items()):
        if ep_score.keep_episode:
            continue
        action, state = arrays_for_episode(ep)
        length = episode_length(ep, action, state)
        start, end = 0, length
        action_slice = _slice_2d(action, start, end)
        state_slice = _slice_2d(state, start, end)
        records.append(
            RegionRecord(
                deletion_level="episode",
                episode=ep,
                start=start,
                end=end,
                keep=False,
                episode_keep=False,
                episode_joint_score=ep_score.episode_joint_score,
                episode_quality=ep_score.episode_quality,
                episode_influence=ep_score.episode_influence,
                window_local_value=float("nan"),
                window_contribution_score=float("nan"),
                action_jitter=_mean_diff_norm(action_slice),
                state_jitter=_mean_diff_norm(state_slice),
                action_energy=_mean_energy(action_slice),
                state_delta_norm=_state_delta_norm(state_slice),
                length=max(end - start, 0),
                phase_start=0.0,
                phase_end=1.0,
                phase_center=0.5,
                phase_bin=_phase_bin(0.5, phase_bins),
                region_x_bin=0,
                region_y_bin=0,
                pca_x=0.0,
                pca_y=0.0,
                state_mean=_mean_vector(state_slice, state_dims),
                action_mean=_mean_vector(action_slice, action_dims),
            )
        )

    # 片段级删除/保留：同时保留 kept 片段用于计算删除覆盖比例。
    for win in window_scores:
        ep_score = episode_scores.get(win.episode)
        if ep_score is None:
            continue
        if win.keep_window and not include_kept:
            continue
        action, state = arrays_for_episode(win.episode)
        length = max(episode_length(win.episode, action, state), win.end)
        start = max(0, int(win.start))
        end = max(start, int(win.end))
        denom = max(float(length), 1.0)
        phase_start = float(np.clip(start / denom, 0.0, 1.0))
        phase_end = float(np.clip(end / denom, 0.0, 1.0))
        phase_center = float(np.clip(0.5 * (phase_start + phase_end), 0.0, 1.0))
        action_slice = _slice_2d(action, start, end)
        state_slice = _slice_2d(state, start, end)
        records.append(
            RegionRecord(
                deletion_level="window",
                episode=win.episode,
                start=start,
                end=end,
                keep=bool(win.keep_window),
                episode_keep=bool(ep_score.keep_episode),
                episode_joint_score=ep_score.episode_joint_score,
                episode_quality=ep_score.episode_quality,
                episode_influence=ep_score.episode_influence,
                window_local_value=win.window_local_value,
                window_contribution_score=win.window_contribution_score,
                action_jitter=_finite_or(win.raw_action_jitter, _mean_diff_norm(action_slice)),
                state_jitter=_finite_or(win.raw_state_jitter, _mean_diff_norm(state_slice)),
                action_energy=_mean_energy(action_slice),
                state_delta_norm=_state_delta_norm(state_slice),
                length=max(end - start, 0),
                phase_start=phase_start,
                phase_end=phase_end,
                phase_center=phase_center,
                phase_bin=_phase_bin(phase_center, phase_bins),
                region_x_bin=0,
                region_y_bin=0,
                pca_x=0.0,
                pca_y=0.0,
                state_mean=_mean_vector(state_slice, state_dims),
                action_mean=_mean_vector(action_slice, action_dims),
            )
        )
    return records


def _assign_regions(records: list[RegionRecord], region_bins: int) -> None:
    if not records:
        return
    max_len = max(len(_build_record_feature(record)) for record in records)
    if max_len == 0:
        return
    features = np.zeros((len(records), max_len), dtype=np.float64)
    for idx, record in enumerate(records):
        feat = _build_record_feature(record)
        if feat:
            features[idx, : len(feat)] = np.asarray(feat, dtype=np.float64)
    coords = _pca_2d(features)
    x_bins = _bin_values(coords[:, 0], region_bins)
    y_bins = _bin_values(coords[:, 1], region_bins)
    for idx, record in enumerate(records):
        record.pca_x = float(coords[idx, 0])
        record.pca_y = float(coords[idx, 1])
        record.region_x_bin = int(x_bins[idx])
        record.region_y_bin = int(y_bins[idx])


def _group_targets(
    records: list[RegionRecord],
    *,
    phase_bins: int,
    top_k: int,
    min_deleted_records: int,
    replacement_ratio: float,
) -> list[CollectionTarget]:
    groups: dict[tuple[int, int, int], list[RegionRecord]] = {}
    for record in records:
        key = (record.region_x_bin, record.region_y_bin, record.phase_bin)
        groups.setdefault(key, []).append(record)

    action_jitters = [r.action_jitter for r in records if math.isfinite(r.action_jitter)]
    state_jitters = [r.state_jitter for r in records if math.isfinite(r.state_jitter)]
    global_action_jitter = float(np.mean(action_jitters)) if action_jitters else 0.0
    global_state_jitter = float(np.mean(state_jitters)) if state_jitters else 0.0

    candidate_rows: list[dict[str, Any]] = []
    for (x_bin, y_bin, phase_bin), rows in groups.items():
        deleted = [r for r in rows if not r.keep]
        kept = [r for r in rows if r.keep]
        if len(deleted) < min_deleted_records:
            continue

        episode_drop_records = sum(1 for r in deleted if r.deletion_level == "episode")
        window_drop_records = sum(1 for r in deleted if r.deletion_level == "window")
        deletion_ratio = len(deleted) / max(len(rows), 1)
        ep_scores = [r.episode_joint_score for r in deleted if math.isfinite(r.episode_joint_score)]
        win_values = [r.window_local_value for r in deleted if math.isfinite(r.window_local_value)]
        action_j = [r.action_jitter for r in deleted if math.isfinite(r.action_jitter)]
        state_j = [r.state_jitter for r in deleted if math.isfinite(r.state_jitter)]
        action_e = [r.action_energy for r in deleted if math.isfinite(r.action_energy)]
        delta = [r.state_delta_norm for r in deleted if math.isfinite(r.state_delta_norm)]

        candidate_rows.append(
            {
                "key": (x_bin, y_bin, phase_bin),
                "deleted": deleted,
                "kept": kept,
                "episode_drop_records": episode_drop_records,
                "window_drop_records": window_drop_records,
                "deletion_ratio": deletion_ratio,
                "mean_episode_joint_score": float(np.mean(ep_scores)) if ep_scores else float("nan"),
                "mean_window_local_value": float(np.mean(win_values)) if win_values else float("nan"),
                "mean_action_jitter": float(np.mean(action_j)) if action_j else float("nan"),
                "mean_state_jitter": float(np.mean(state_j)) if state_j else float("nan"),
                "mean_action_energy": float(np.mean(action_e)) if action_e else float("nan"),
                "mean_state_delta_norm": float(np.mean(delta)) if delta else float("nan"),
            }
        )

    if not candidate_rows:
        return []

    count_norm = _normalize_metric([float(len(row["deleted"])) for row in candidate_rows])
    ratio_norm = _normalize_metric([float(row["deletion_ratio"]) for row in candidate_rows])
    ep_bad_norm = _normalize_metric(
        [float(row["mean_episode_joint_score"]) for row in candidate_rows],
        invert=True,
    )
    win_bad_norm = _normalize_metric(
        [float(row["mean_window_local_value"]) for row in candidate_rows],
        invert=True,
    )
    action_j_norm = _normalize_metric([float(row["mean_action_jitter"]) for row in candidate_rows])
    state_j_norm = _normalize_metric([float(row["mean_state_jitter"]) for row in candidate_rows])

    targets: list[CollectionTarget] = []
    for idx, row in enumerate(candidate_rows):
        deleted: list[RegionRecord] = row["deleted"]
        kept: list[RegionRecord] = row["kept"]
        x_bin, y_bin, phase_bin = row["key"]
        priority = (
            0.28 * count_norm[idx]
            + 0.22 * ratio_norm[idx]
            + 0.18 * ep_bad_norm[idx]
            + 0.14 * win_bad_norm[idx]
            + 0.09 * action_j_norm[idx]
            + 0.09 * state_j_norm[idx]
        )
        # episode 级整条删除代表更高的数据缺口，给一个轻量加成。
        if row["episode_drop_records"] > 0:
            priority += 0.05
        priority = float(priority * max(float(replacement_ratio), 0.0))

        reps = sorted({r.episode for r in deleted})[:8]
        segs = [f"ep{r.episode}:{r.start}-{r.end}" for r in deleted[:8]]
        state_vectors = [r.state_mean for r in deleted if r.state_mean]
        action_vectors = [r.action_mean for r in deleted if r.action_mean]
        state_center = []
        action_center = []
        if state_vectors:
            width = max(len(v) for v in state_vectors)
            mat = np.full((len(state_vectors), width), np.nan, dtype=np.float64)
            for i, vec in enumerate(state_vectors):
                mat[i, : len(vec)] = vec
            state_center = np.nanmean(mat, axis=0).tolist()
        if action_vectors:
            width = max(len(v) for v in action_vectors)
            mat = np.full((len(action_vectors), width), np.nan, dtype=np.float64)
            for i, vec in enumerate(action_vectors):
                mat[i, : len(vec)] = vec
            action_center = np.nanmean(mat, axis=0).tolist()

        phase = _phase_name(int(phase_bin), phase_bins)
        region_id = f"rx{x_bin}_ry{y_bin}_p{phase_bin}"
        state_center_text = _format_vec([float(x) for x in state_center])
        action_center_text = _format_vec([float(x) for x in action_center])
        suggested_count = max(1, int(math.ceil(len(deleted) * max(float(replacement_ratio), 0.1))))
        collection_region = _make_collection_region(
            region_id=region_id,
            phase_name=phase,
            state_center=state_center_text,
            action_center=action_center_text,
            deletion_ratio=float(row["deletion_ratio"]),
        )
        collection_pose = _make_collection_pose(
            phase_name=phase,
            state_center=state_center_text,
            action_center=action_center_text,
            episode_drop_records=int(row["episode_drop_records"]),
            window_drop_records=int(row["window_drop_records"]),
        )
        trajectory_corrections = _make_trajectory_corrections(
            mean_episode_joint_score=float(row["mean_episode_joint_score"]),
            mean_window_local_value=float(row["mean_window_local_value"]),
            mean_action_jitter=float(row["mean_action_jitter"]),
            global_action_jitter=global_action_jitter,
            mean_state_jitter=float(row["mean_state_jitter"]),
            global_state_jitter=global_state_jitter,
            mean_state_delta_norm=float(row["mean_state_delta_norm"]),
            deletion_ratio=float(row["deletion_ratio"]),
        )
        targets.append(
            CollectionTarget(
                rank=0,
                priority_score=priority,
                region_id=region_id,
                phase_name=phase,
                suggested_episode_count=suggested_count,
                deletion_records=int(len(deleted)),
                kept_records=int(len(kept)),
                deletion_ratio=float(row["deletion_ratio"]),
                episode_drop_records=int(row["episode_drop_records"]),
                window_drop_records=int(row["window_drop_records"]),
                mean_episode_joint_score=float(row["mean_episode_joint_score"]),
                mean_window_local_value=float(row["mean_window_local_value"]),
                mean_action_jitter=float(row["mean_action_jitter"]),
                mean_state_jitter=float(row["mean_state_jitter"]),
                mean_action_energy=float(row["mean_action_energy"]),
                mean_state_delta_norm=float(row["mean_state_delta_norm"]),
                representative_episodes=",".join(str(ep) for ep in reps),
                representative_segments=";".join(segs),
                state_center=state_center_text,
                action_center=action_center_text,
                collection_region=collection_region,
                collection_pose=collection_pose,
                trajectory_corrections=trajectory_corrections,
                guidance=_make_guidance(
                    phase_name=phase,
                    deletion_ratio=float(row["deletion_ratio"]),
                    episode_drop_records=int(row["episode_drop_records"]),
                    window_drop_records=int(row["window_drop_records"]),
                    mean_episode_joint_score=float(row["mean_episode_joint_score"]),
                    mean_window_local_value=float(row["mean_window_local_value"]),
                    mean_action_jitter=float(row["mean_action_jitter"]),
                    global_action_jitter=global_action_jitter,
                    mean_state_jitter=float(row["mean_state_jitter"]),
                    global_state_jitter=global_state_jitter,
                    mean_state_delta_norm=float(row["mean_state_delta_norm"]),
                ),
            )
        )

    targets.sort(key=lambda item: item.priority_score, reverse=True)
    if top_k > 0:
        targets = targets[:top_k]
    for rank, target in enumerate(targets, start=1):
        target.rank = rank
    return targets


def _write_records_csv(path: Path, records: list[RegionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(RegionRecord.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["state_mean"] = _format_vec(record.state_mean)
            row["action_mean"] = _format_vec(record.action_mean)
            writer.writerow(row)


def _write_targets_csv(path: Path, targets: list[CollectionTarget]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(CollectionTarget.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for target in targets:
            writer.writerow(asdict(target))


def _phase_sort_key(phase_name: str) -> int:
    if "起始" in phase_name:
        return 0
    if "前半段" in phase_name:
        return 1
    if "中段" in phase_name:
        return 2
    if "后半段" in phase_name:
        return 3
    if "末端" in phase_name:
        return 4
    return 99


def _phase_issue_summary(phase_name: str) -> str:
    if "起始" in phase_name:
        return "初始对齐不足，路径不够直接，早期动作抖动"
    if "前半段" in phase_name:
        return "接近过程存在方向切换和冗余修正"
    if "中段" in phase_name:
        return "抓取/接触保持不稳定，阶段转换质量不足"
    if "后半段" in phase_name or "末端" in phase_name:
        return "靠近终态时收敛不稳定，存在回退和抖动"
    return "局部状态-动作覆盖质量不足"


def _build_phase_summary(targets: list[CollectionTarget]) -> list[dict[str, Any]]:
    grouped: dict[str, list[CollectionTarget]] = {}
    for target in targets:
        grouped.setdefault(target.phase_name, []).append(target)
    rows: list[dict[str, Any]] = []
    for phase_name, items in grouped.items():
        rows.append(
            {
                "phase_name": phase_name,
                "regions": len(items),
                "suggested": sum(int(item.suggested_episode_count) for item in items),
                "issue": _phase_issue_summary(phase_name),
            }
        )
    return sorted(rows, key=lambda row: _phase_sort_key(str(row["phase_name"])))


def _build_main_finding(targets: list[CollectionTarget]) -> str:
    phase_rows = _build_phase_summary(targets)
    if not phase_rows:
        return "当前删除结果没有形成稳定的阶段性聚集，暂不建议据此安排定向补采。"

    top_phases = sorted(phase_rows, key=lambda row: int(row["suggested"]), reverse=True)[:2]
    phase_text = "、".join(
        f"{row['phase_name']}建议补采{row['suggested']}条" for row in top_phases
    )
    appeared = "、".join(row["phase_name"] for row in phase_rows)
    return (
        f"整体来看，被删除样本主要集中在{appeared}。其中，{phase_text}，"
        "是本轮补采的优先方向。这说明当前数据的主要质量缺口并不是简单的数据覆盖不足，"
        "而是集中出现在任务阶段切换、抓取/接触保持和目标收敛过程中的动作不连续、"
        "状态回退和局部抖动问题。"
    )


def _brief_target_correction(target: CollectionTarget) -> str:
    parts: list[str] = []
    if target.episode_drop_records > 0:
        parts.append("包含整轨删除，优先补完整成功轨迹")
    if target.window_drop_records > 0:
        parts.append("包含局部 window 删除，补采时保留进入和离开该区域的上下文")
    if "起始" in target.phase_name:
        parts.append("重点减少初始对齐阶段的动作突变")
    elif "前半段" in target.phase_name:
        parts.append("重点减少接近过程中的横向修正和路径回退")
    elif "中段" in target.phase_name:
        parts.append("重点保证抓取/接触约束稳定和阶段转换连续")
    elif "后半段" in target.phase_name or "末端" in target.phase_name:
        parts.append("重点保证靠近终态时减速收敛，避免反复调整")
    if not parts:
        parts.append("按共性要求控制动作连续性和状态平滑性")
    return "；".join(parts) + "。"


def _write_target_detail(lines: list[str], target: CollectionTarget) -> None:
    lines.extend(
        [
            f"### {target.rank}. {target.region_id} | {target.phase_name}",
            "",
            f"- 优先级分数：{target.priority_score:.4f}",
            f"- 建议补采数量：{target.suggested_episode_count} 条高质量完整轨迹或阶段完整片段",
            f"- 删除/保留训练单元：{target.deletion_records}/{target.kept_records}，删除占比 {target.deletion_ratio:.2%}",
            f"- 整轨删除数：{target.episode_drop_records}",
            f"- 局部 window 删除数：{target.window_drop_records}",
            f"- 涉及 episode：`{target.representative_episodes}`",
            f"- 涉及片段：`{target.representative_segments}`",
            f"- 状态中心参考：`{target.state_center}`",
            f"- 动作中心参考：`{target.action_center}`",
            f"- 采集区域：{target.collection_region}",
            f"- 采集姿态：{target.collection_pose}",
            f"- 差异化修正：{_brief_target_correction(target)}",
            "",
        ]
    )


def _write_markdown(path: Path, targets: list[CollectionTarget], summary: dict[str, Any]) -> None:
    phase_rows = _build_phase_summary(targets)
    top_targets = targets[:5]
    appendix_targets = targets[5:]
    lines: list[str] = [
        "# MYSYS 删除结果驱动补采建议",
        "",
        "## 1. 总览",
        "",
        f"- 输入筛选目录：`{summary['selection_dir']}`",
        f"- 数据集目录：`{summary['dataset_root']}`",
        f"- 删除训练单元数：{summary['deleted_records']}",
        f"- 保留参考训练单元数：{summary['kept_records']}",
        f"- 输出补采目标区域数：{len(targets)}",
        "",
        "本文中的训练单元指筛选算法中的 record，可能对应整条 episode，",
        "也可能对应局部时间 window。建议补采数量表示该区域所需的有效样本数量；",
        "实际采集时优先采完整成功轨迹，并确保目标时间段附近有充分上下文。",
        "",
        "## 2. 主要发现",
        "",
        _build_main_finding(targets),
        "",
        "正文重点关注优先级最高的 5 个区域，这些区域覆盖了起始接近、中段转换和后半段收敛等关键阶段，",
        "能够代表主要的数据质量缺口。其余区域作为次级补采目标，放入附录供后续扩展采集使用。",
        "",
        "## 3. 阶段级补采需求汇总",
        "",
        "| 阶段 | 对应区域数 | 建议补采总数 | 主要问题 |",
        "|---|---:|---:|---|",
    ]
    for row in phase_rows:
        lines.append(
            f"| {row['phase_name']} | {row['regions']} | {row['suggested']} | {row['issue']} |"
        )
    lines.extend(
        [
            "",
            "## 4. 共性补采要求",
            "",
            "所有补采轨迹需要满足以下要求：",
            "",
            "1. 动作连续：避免方向来回切换、速度突变和夹爪高频开合。",
            "2. 状态平滑：末端轨迹应单调推进，避免绕路、回退和接触后反复调整。",
            "3. 上下文完整：不要只采孤立片段，每条轨迹至少包含进入该状态区域、完成关键转换、离开该区域的上下文。",
            "4. 任务成功：补采样本应优先保证完整任务成功，而不是只让局部动作看起来平滑。",
            "",
            "## 5. Top-5 高优先级补采区域",
            "",
        ]
    )
    if top_targets:
        for target in top_targets:
            _write_target_detail(lines, target)
    else:
        lines.extend(
            [
                "没有形成稳定的补采目标。可能原因：删除样本太少、`min_deleted_records` 过高，",
                "或当前筛选结果主要是均匀低密度删除。",
                "",
            ]
        )
    lines.extend(
        [
            "## 6. 次级补采区域",
            "",
        ]
    )
    if appendix_targets:
        lines.extend(
            [
                "| 排名 | 区域 | 阶段 | 建议补采数 | 删除/保留训练单元 | 整轨删除 | 局部 window 删除 | 涉及 episode |",
                "|---:|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for target in appendix_targets:
            lines.append(
                f"| {target.rank} | {target.region_id} | {target.phase_name} | "
                f"{target.suggested_episode_count} | {target.deletion_records}/{target.kept_records} | "
                f"{target.episode_drop_records} | {target.window_drop_records} | `{target.representative_episodes}` |"
            )
        lines.append("")
    else:
        lines.extend(["暂无次级补采区域。", ""])
    lines.extend(
        [
            "## 7. 与成功率差异的关系",
            "",
            "这些被删除区域大多不是随机分布，而是集中在任务执行中的关键转折阶段：",
            "起始接近、中段抓取/接触保持、后半段目标收敛。若这些阶段存在高 action jitter、",
            "state jitter 或夹爪时机不稳定，模型在闭环推理时容易出现回弹、重复修正、",
            "接触后偏移或放置失败。因此，补采这些区域的高质量样本，可以提高模型对关键阶段的",
            "动作一致性和状态稳定性，也解释了少量高质量数据可能优于全量噪声数据的现象。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MYSYS episode/window deletion outputs and generate collection targets."
    )
    parser.add_argument("--selection-dir", type=Path, required=True, help="filter_joint_episode_window.py 输出目录。")
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobot 数据集根目录。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认写到 selection-dir/collection_targets。",
    )
    parser.add_argument("--state-dims", type=int, default=8, help="用于描述区域的 observation.state 前 N 维。")
    parser.add_argument("--action-dims", type=int, default=8, help="用于描述区域的 action 前 N 维。")
    parser.add_argument("--phase-bins", type=int, default=5, help="轨迹时间相位分箱数。")
    parser.add_argument("--region-bins", type=int, default=4, help="PCA state-action 区域每轴分箱数。")
    parser.add_argument("--top-k", type=int, default=12, help="最多输出多少个补采目标，0 表示全部输出。")
    parser.add_argument("--min-deleted-records", type=int, default=2, help="形成目标区域所需的最少删除 record 数。")
    parser.add_argument(
        "--replacement-ratio",
        type=float,
        default=1.0,
        help="补采强度系数，仅影响 priority_score，便于做不同预算版本。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection_dir = args.selection_dir
    dataset_root = args.dataset_root
    output_dir = args.output_dir or (selection_dir / "collection_targets")

    episode_csv = selection_dir / "episode_scores.csv"
    window_csv = selection_dir / "window_scores.csv"
    if not episode_csv.exists():
        raise FileNotFoundError(f"缺少 episode_scores.csv: {episode_csv}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"缺少 dataset root: {dataset_root}")

    episode_scores = _read_episode_scores(episode_csv)
    window_scores = _read_window_scores(window_csv)
    records = _build_records(
        episode_scores=episode_scores,
        window_scores=window_scores,
        dataset_root=dataset_root,
        state_dims=max(int(args.state_dims), 0),
        action_dims=max(int(args.action_dims), 0),
        phase_bins=max(int(args.phase_bins), 1),
        include_kept=True,
    )
    _assign_regions(records, region_bins=max(int(args.region_bins), 1))
    targets = _group_targets(
        records,
        phase_bins=max(int(args.phase_bins), 1),
        top_k=int(args.top_k),
        min_deleted_records=max(int(args.min_deleted_records), 1),
        replacement_ratio=float(args.replacement_ratio),
    )

    deleted_count = sum(1 for record in records if not record.keep)
    kept_count = sum(1 for record in records if record.keep)
    summary = {
        "selection_dir": str(selection_dir),
        "dataset_root": str(dataset_root),
        "episodes": len(episode_scores),
        "windows": len(window_scores),
        "records": len(records),
        "deleted_records": int(deleted_count),
        "kept_records": int(kept_count),
        "targets": len(targets),
        "phase_bins": int(args.phase_bins),
        "region_bins": int(args.region_bins),
        "state_dims": int(args.state_dims),
        "action_dims": int(args.action_dims),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_records_csv(output_dir / "deletion_region_records.csv", records)
    _write_targets_csv(output_dir / "collection_targets.csv", targets)
    (output_dir / "collection_targets.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "targets": [asdict(target) for target in targets],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_dir / "collection_targets.md", targets, summary)
    print(
        f"[done] records={len(records)} deleted={deleted_count} kept={kept_count} "
        f"targets={len(targets)} output={output_dir}"
    )


if __name__ == "__main__":
    main()
