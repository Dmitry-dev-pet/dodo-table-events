#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
HTML_REPORT_SCRIPT="scripts/html_report_v01.py"
COMBINED_REPORT_SCRIPT="scripts/combined_report.py"
PRESET="${PRESET:-}"
VIDEO_PATH="${VIDEO_PATH:-}"
ROI_JSON_PATH="${ROI_JSON_PATH:-}"
OUT_DIR="${OUT_DIR:-}"
DETECTOR="${DETECTOR:-all}"
BUILD_HTML=1
EXTRA_MAIN_ARGS=()
PRESET_VIDEO_PATH=""
PRESET_ROI_JSON_PATH=""
PRESET_OUT_DIR=""

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run.sh [options] [extra main.py args]

Options:
  --preset NAME        Run a named preset (video1, video2, video3, all)
  --video PATH         Input video path
  --roi-json PATH      ROI preset JSON path
  --out-dir PATH       Output directory
  --detector NAME      Detector mode for main.py (default: all)
  --skip-html          Do not build HTML reports after the run
  -h, --help           Show this help

Examples:
  ./scripts/run.sh --preset video1
  ./scripts/run.sh --preset all
  ./scripts/run.sh --preset video1 --max-seconds 60
  ./scripts/run.sh --video data/video1.mp4 --roi-json roi_presets/video1_table.json
  DETECTOR=motion ./scripts/run.sh --preset video1
EOF
}

resolve_preset() {
  local name="$1"
  case "${name}" in
    video1)
      PRESET_VIDEO_PATH="data/video1.mp4"
      PRESET_ROI_JSON_PATH="roi_presets/video1_table.json"
      PRESET_OUT_DIR="out/video1"
      ;;
    video2)
      PRESET_VIDEO_PATH="data/video2.mp4"
      PRESET_ROI_JSON_PATH="roi_presets/video2_table.json"
      PRESET_OUT_DIR="out/video2"
      ;;
    video3)
      PRESET_VIDEO_PATH="data/video3.mp4"
      PRESET_ROI_JSON_PATH="roi_presets/video3_table.json"
      PRESET_OUT_DIR="out/video3"
      ;;
    *)
      printf '[run] ERROR: unknown preset: %s\n' "${name}" >&2
      exit 1
      ;;
  esac
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --preset)
        PRESET="${2:?missing value for --preset}"
        shift 2
        ;;
      --video)
        VIDEO_PATH="${2:?missing value for --video}"
        shift 2
        ;;
      --roi-json)
        ROI_JSON_PATH="${2:?missing value for --roi-json}"
        shift 2
        ;;
      --out-dir)
        OUT_DIR="${2:?missing value for --out-dir}"
        shift 2
        ;;
      --detector)
        DETECTOR="${2:?missing value for --detector}"
        shift 2
        ;;
      --skip-html)
        BUILD_HTML=0
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        EXTRA_MAIN_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

ensure_python() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf '[run] ERROR: python not found: %s\n' "${PYTHON_BIN}" >&2
    printf '[run] Run ./scripts/setup.sh first.\n' >&2
    exit 1
  fi
}

ensure_single_inputs() {
  if [[ -n "${PRESET}" ]]; then
    resolve_preset "${PRESET}"
    VIDEO_PATH="${VIDEO_PATH:-${PRESET_VIDEO_PATH}}"
    ROI_JSON_PATH="${ROI_JSON_PATH:-${PRESET_ROI_JSON_PATH}}"
    OUT_DIR="${OUT_DIR:-${PRESET_OUT_DIR}}"
  fi

  if [[ -z "${VIDEO_PATH}" ]]; then
    printf '[run] ERROR: video path is required. Use --preset or --video.\n' >&2
    exit 1
  fi

  if [[ -z "${ROI_JSON_PATH}" ]]; then
    printf '[run] ERROR: roi-json path is required. Use --preset or --roi-json.\n' >&2
    exit 1
  fi

  if [[ -z "${OUT_DIR}" ]]; then
    local video_name
    video_name="$(basename "${VIDEO_PATH}")"
    OUT_DIR="out/${video_name%.*}"
  fi

  if [[ ! -f "${VIDEO_PATH}" ]]; then
    printf '[run] ERROR: video not found: %s\n' "${VIDEO_PATH}" >&2
    printf '[run] Download input videos via ./scripts/setup.sh --with-videos\n' >&2
    exit 1
  fi

  if [[ ! -f "${ROI_JSON_PATH}" ]]; then
    printf '[run] ERROR: ROI preset not found: %s\n' "${ROI_JSON_PATH}" >&2
    exit 1
  fi
}

ensure_all_preset_inputs() {
  if [[ -n "${VIDEO_PATH}" || -n "${ROI_JSON_PATH}" || -n "${OUT_DIR}" ]]; then
    printf '[run] ERROR: --preset all does not accept --video/--roi-json/--out-dir overrides.\n' >&2
    exit 1
  fi
}

run_main_once() {
  local video_path="$1"
  local roi_json_path="$2"
  local out_dir="$3"
  local cmd=(
    "${PYTHON_BIN}"
    main.py
    --video "${video_path}"
    --detector "${DETECTOR}"
    --roi-json "${roi_json_path}"
    --out-dir "${out_dir}"
  )
  printf '[run] video=%s roi=%s out=%s detector=%s\n' "${video_path}" "${roi_json_path}" "${out_dir}" "${DETECTOR}"
  if (( ${#EXTRA_MAIN_ARGS[@]} > 0 )); then
    cmd+=("${EXTRA_MAIN_ARGS[@]}")
  fi
  "${cmd[@]}"
}

run_html_report_once() {
  local out_dir="$1"
  if [[ "${BUILD_HTML}" -ne 1 ]]; then
    return
  fi

  "${PYTHON_BIN}" "${HTML_REPORT_SCRIPT}" --out-dir "${out_dir}"
}

run_combined_report() {
  if [[ "${BUILD_HTML}" -ne 1 ]]; then
    return
  fi

  "${PYTHON_BIN}" "${COMBINED_REPORT_SCRIPT}" --out-root out --out out/report.html
}

run_single() {
  ensure_single_inputs
  run_main_once "${VIDEO_PATH}" "${ROI_JSON_PATH}" "${OUT_DIR}"
  run_html_report_once "${OUT_DIR}"
}

run_all_presets() {
  local preset_name
  ensure_all_preset_inputs
  for preset_name in video1 video2 video3; do
    resolve_preset "${preset_name}"
    run_main_once "${PRESET_VIDEO_PATH}" "${PRESET_ROI_JSON_PATH}" "${PRESET_OUT_DIR}"
    run_html_report_once "${PRESET_OUT_DIR}"
  done
}

main() {
  parse_args "$@"
  ensure_python
  if [[ "${PRESET}" == "all" ]]; then
    run_all_presets
  else
    run_single
  fi
  run_combined_report
}

main "$@"
