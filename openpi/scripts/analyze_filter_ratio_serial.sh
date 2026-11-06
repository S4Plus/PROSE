#!/usr/bin/env bash
# Analyze adaptive filtering ratios from existing score artifacts or raw joint sources.
#
# Example:
#   bash scripts/analyze_filter_ratio_serial.sh
#   PLOT=1 bash scripts/analyze_filter_ratio_serial.sh
#   SCORE_FILES="/path/a.npz /path/b.npz" bash scripts/analyze_filter_ratio_serial.sh
#   SCORE_PKLS="/path/a.pkl /path/b.pkl" QUALITY_SCORE_CSV="/path/quality.csv" bash scripts/analyze_filter_ratio_serial.sh
#
# Tunable env vars:
#   SCORE_FILES
#   SCORE_PKLS
#   QUALITY_SCORE_PKL
#   QUALITY_SCORE_CSV
#   SCORE_FIELD
#   QUALITY_CSV_EPISODE_COLUMN
#   QUALITY_CSV_SCORE_COLUMN
#   EPISODE_INFLUENCE_FIELD
#   EPISODE_QUALITY_FIELD
#   JOINT_QUALITY_POWER
#   JOINT_INFLUENCE_POWER
#   SAVE_JOINT_NPZ
#   PLOT
#   UV_CACHE_DIR
#
# Example nohup:
#   nohup bash scripts/analyze_filter_ratio_serial.sh \
#     > /workspace/liushuo/cupid/openpi/tmp/analyze_filter_ratio_serial.log 2>&1 &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV_CACHE_DIR="${UV_CACHE_DIR:-./tmp/uv-cache}"
SCORE_FIELD="${SCORE_FIELD:-}"
PLOT="${PLOT:-0}"
QUALITY_SCORE_PKL="${QUALITY_SCORE_PKL:-}"
QUALITY_SCORE_CSV="${QUALITY_SCORE_CSV:-}"
QUALITY_CSV_EPISODE_COLUMN="${QUALITY_CSV_EPISODE_COLUMN:-episode}"
QUALITY_CSV_SCORE_COLUMN="${QUALITY_CSV_SCORE_COLUMN:-quality_score}"
EPISODE_INFLUENCE_FIELD="${EPISODE_INFLUENCE_FIELD:-episode_influence_scores}"
EPISODE_QUALITY_FIELD="${EPISODE_QUALITY_FIELD:-episode_quality_scores}"
JOINT_QUALITY_POWER="${JOINT_QUALITY_POWER:-1}"
JOINT_INFLUENCE_POWER="${JOINT_INFLUENCE_POWER:-1}"
SAVE_JOINT_NPZ="${SAVE_JOINT_NPZ:-0}"

DEFAULT_SCORE_FILES=(
  "/workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/joint_selection_scores.npz"
  "/workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/joint_selection_scores.npz"
  "/workspace/liushuo/cupid/openpi/tmp/can_mh_window_joint_keep90_rank_fusion/joint_window_selection_scores.npz"
)

DEFAULT_SCORE_PKLS=(
  "/workspace/liushuo/cupid/openpi/tmp/can_mh_grass_all/influence_grass.pkl"
  "/workspace/liushuo/cupid/openpi/tmp/can_mh_grass_last4_strict/influence_grass.pkl"
)

if [[ -n "${SCORE_FILES:-}" ]]; then
  INPUT_MODE="score_npz"
  read -r -a SCORE_FILE_LIST <<< "$SCORE_FILES"
elif [[ -n "${SCORE_PKLS:-}" || -n "$QUALITY_SCORE_PKL" || -n "$QUALITY_SCORE_CSV" ]]; then
  INPUT_MODE="joint_sources"
  if [[ -n "${SCORE_PKLS:-}" ]]; then
    read -r -a SCORE_PKL_LIST <<< "$SCORE_PKLS"
  else
    SCORE_PKL_LIST=("${DEFAULT_SCORE_PKLS[@]}")
  fi
else
  INPUT_MODE="score_npz"
  SCORE_FILE_LIST=("${DEFAULT_SCORE_FILES[@]}")
fi

if [[ -n "$QUALITY_SCORE_PKL" && -n "$QUALITY_SCORE_CSV" ]]; then
  echo "Only one of QUALITY_SCORE_PKL or QUALITY_SCORE_CSV can be set." >&2
  exit 1
fi

run_npz() {
  local score_file="$1"
  local report_json
  local report_png
  local -a cmd

  if [[ ! -f "$score_file" ]]; then
    echo "Skipping missing score file: $score_file"
    return 0
  fi

  report_json="${score_file%.npz}_ratio_analysis.json"
  report_png="${score_file%.npz}_ratio_analysis.png"

  cmd=(
    uv run python scripts/analyze_filter_ratio.py
    --score-npz "$score_file"
    --output-json "$report_json"
  )

  if [[ -n "$SCORE_FIELD" ]]; then
    cmd+=(--score-field "$SCORE_FIELD")
  fi
  if [[ "$PLOT" == "1" ]]; then
    cmd+=(--plot --output-png "$report_png")
  fi

  echo "============================================================"
  echo "Analyzing score file: $score_file"
  echo "Report JSON: $report_json"
  if [[ "$PLOT" == "1" ]]; then
    echo "Report PNG: $report_png"
  fi
  echo "Command: ${cmd[*]}"
  echo "============================================================"

  UV_CACHE_DIR="$UV_CACHE_DIR" "${cmd[@]}"
}

run_joint() {
  local score_pkl="$1"
  local report_prefix
  local report_json
  local report_png
  local output_joint_npz
  local -a cmd

  if [[ ! -f "$score_pkl" ]]; then
    echo "Skipping missing score pickle: $score_pkl"
    return 0
  fi
  if [[ -n "$QUALITY_SCORE_PKL" && ! -f "$QUALITY_SCORE_PKL" ]]; then
    echo "Skipping because quality score pickle is missing: $QUALITY_SCORE_PKL"
    return 0
  fi
  if [[ -n "$QUALITY_SCORE_CSV" && ! -f "$QUALITY_SCORE_CSV" ]]; then
    echo "Skipping because quality score csv is missing: $QUALITY_SCORE_CSV"
    return 0
  fi

  if [[ -n "$QUALITY_SCORE_CSV" ]]; then
    report_prefix="${score_pkl%.pkl}__qfrom_$(basename "${QUALITY_SCORE_CSV%.csv}")_joint_ratio_analysis"
  elif [[ -n "$QUALITY_SCORE_PKL" ]]; then
    report_prefix="${score_pkl%.pkl}__qfrom_$(basename "${QUALITY_SCORE_PKL%.pkl}")_joint_ratio_analysis"
  else
    report_prefix="${score_pkl%.pkl}_ratio_analysis"
  fi
  report_json="${report_prefix}.json"
  report_png="${report_prefix}.png"
  output_joint_npz="${report_prefix%.json}_scores.npz"

  cmd=(
    uv run python scripts/analyze_filter_ratio.py
    --score-pkl "$score_pkl"
    --episode-influence-field "$EPISODE_INFLUENCE_FIELD"
    --episode-quality-field "$EPISODE_QUALITY_FIELD"
    --quality-csv-episode-column "$QUALITY_CSV_EPISODE_COLUMN"
    --quality-csv-score-column "$QUALITY_CSV_SCORE_COLUMN"
    --joint-quality-power "$JOINT_QUALITY_POWER"
    --joint-influence-power "$JOINT_INFLUENCE_POWER"
    --output-json "$report_json"
  )

  if [[ -n "$QUALITY_SCORE_PKL" ]]; then
    cmd+=(--quality-score-pkl "$QUALITY_SCORE_PKL")
  fi
  if [[ -n "$QUALITY_SCORE_CSV" ]]; then
    cmd+=(--quality-score-csv "$QUALITY_SCORE_CSV")
  fi
  if [[ "$PLOT" == "1" ]]; then
    cmd+=(--plot --output-png "$report_png")
  fi
  if [[ "$SAVE_JOINT_NPZ" == "1" ]]; then
    cmd+=(--output-joint-npz "$output_joint_npz")
  fi

  echo "============================================================"
  echo "Analyzing joint sources:"
  echo "  Influence PKL: $score_pkl"
  if [[ -n "$QUALITY_SCORE_PKL" ]]; then
    echo "  Quality PKL: $QUALITY_SCORE_PKL"
  fi
  if [[ -n "$QUALITY_SCORE_CSV" ]]; then
    echo "  Quality CSV: $QUALITY_SCORE_CSV"
  fi
  echo "Report JSON: $report_json"
  if [[ "$PLOT" == "1" ]]; then
    echo "Report PNG: $report_png"
  fi
  if [[ "$SAVE_JOINT_NPZ" == "1" ]]; then
    echo "Derived joint NPZ: $output_joint_npz"
  fi
  echo "Command: ${cmd[*]}"
  echo "============================================================"

  UV_CACHE_DIR="$UV_CACHE_DIR" "${cmd[@]}"
}

if [[ "$INPUT_MODE" == "score_npz" ]]; then
  for score_file in "${SCORE_FILE_LIST[@]}"; do
    run_npz "$score_file"
  done
else
  for score_pkl in "${SCORE_PKL_LIST[@]}"; do
    run_joint "$score_pkl"
  done
fi

echo "All ratio analyses finished."
