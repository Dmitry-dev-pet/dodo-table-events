#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

WITH_VIDEOS=0
SKIP_MODELS=0
SKIP_YOLO_WEIGHTS=0
FORCE=0

readonly VIDEO1_PATH="data/video1.mp4"
readonly VIDEO2_PATH="data/video2.mp4"
readonly VIDEO3_PATH="data/video3.mp4"
readonly OPENCV_MODEL_PATH="models/opencv_dnn/person_detection_mediapipe_2023mar.onnx"
readonly OPENCV_ANCHORS_PATH="models/opencv_dnn/person_detection_mediapipe_2023mar.anchors.npy"
readonly YOLO_WEIGHTS_PATH="models/yolov8n.pt"

VIDEO1_STATUS="unknown"
VIDEO2_STATUS="unknown"
VIDEO3_STATUS="unknown"
OPENCV_MODEL_STATUS="unknown"
YOLO_RUNTIME_STATUS="unknown"
YOLO_WEIGHTS_STATUS="unknown"

set_status() {
  local var_name="$1"
  local value="$2"
  printf -v "${var_name}" '%s' "${value}"
}

log() {
  printf '[setup] %s\n' "$*"
}

warn() {
  printf '[setup] WARNING: %s\n' "$*" >&2
}

usage() {
  cat <<'EOF'
Usage:
  ./scripts/setup.sh [options]

Options:
  --with-videos         Download missing input videos into data/
  --skip-models         Skip checking OpenCV DNN model files
  --skip-yolo-weights   Skip checking YOLOv8n weights
  --force               Re-download optional assets even if they already exist
  -h, --help            Show this help

What the script does:
  - checks python3
  - creates .venv if needed
  - installs base pip dependencies if they are missing
  - installs YOLO runtime dependencies (legacy on Python 3.8, ultralytics on Python 3.9+)
  - creates data/, models/, out/
  - checks versioned model files in models/
  - checks whether video1.mp4/video2.mp4/video3.mp4 already exist
  - downloads videos only when --with-videos is passed
EOF
}

require_cmd() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    printf '[setup] ERROR: command not found: %s\n' "${name}" >&2
    exit 1
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --with-videos)
        WITH_VIDEOS=1
        ;;
      --skip-models)
        SKIP_MODELS=1
        ;;
      --skip-yolo-weights)
        SKIP_YOLO_WEIGHTS=1
        ;;
      --force)
        FORCE=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        printf '[setup] ERROR: unknown option: %s\n' "$1" >&2
        usage >&2
        exit 1
        ;;
    esac
    shift
  done
}

ensure_directories() {
  mkdir -p data models out
}

ensure_venv() {
  if [[ -x ".venv/bin/python" ]]; then
    log "Virtual environment already exists: .venv"
    return
  fi
  log "Creating virtual environment in .venv"
  python3 -m venv .venv
}

base_deps_installed() {
  .venv/bin/python - <<'PY' >/dev/null 2>&1
import cv2  # noqa: F401
import numpy  # noqa: F401
import pandas  # noqa: F401
import tqdm  # noqa: F401
PY
}

install_base_requirements() {
  if base_deps_installed; then
    log "Base Python dependencies already installed"
    return
  fi
  log "Installing base Python dependencies from requirements.txt"
  .venv/bin/pip install -r requirements.txt
}

yolo_runtime_installed() {
  .venv/bin/python - <<'PY' >/dev/null 2>&1
import ultralytics  # noqa: F401
PY
}

python_mm() {
  .venv/bin/python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

install_yolo_runtime() {
  local py_mm
  py_mm="$(python_mm)"

  if yolo_runtime_installed; then
    log "YOLO runtime already installed"
    set_status "YOLO_RUNTIME_STATUS" "present"
    return
  fi

  case "${py_mm}" in
    3.8)
      log "Installing legacy YOLO runtime for Python ${py_mm}"
      .venv/bin/python scripts/install_yolo38.py
      ;;
    3.9|3.10|3.11|3.12|3.13)
      log "Installing YOLO runtime (ultralytics) for Python ${py_mm}"
      .venv/bin/pip install "ultralytics>=8,<9"
      ;;
    *)
      warn "Unsupported Python version for automatic YOLO setup: ${py_mm}"
      set_status "YOLO_RUNTIME_STATUS" "unsupported-python"
      return
      ;;
  esac

  if yolo_runtime_installed; then
    set_status "YOLO_RUNTIME_STATUS" "installed"
  else
    set_status "YOLO_RUNTIME_STATUS" "missing"
    warn "YOLO runtime installation finished, but ultralytics import still fails"
  fi
}

check_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then
    log "ffmpeg found: $(command -v ffmpeg)"
  else
    warn "ffmpeg not found in PATH. HTML/video troubleshooting may be limited."
  fi
}

file_present() {
  local path="$1"
  [[ -s "${path}" ]]
}

report_video_status() {
  local path="$1"
  local label="$2"
  if file_present "${path}"; then
    log "${label}: found (${path})"
  else
    warn "${label}: missing (${path})"
  fi
}

resolve_status_after_helper() {
  local path="$1"
  local status_var="$2"
  local had_before="$3"

  if file_present "${path}"; then
    if [[ "${had_before}" -eq 1 && "${FORCE}" -eq 0 ]]; then
      set_status "${status_var}" "present"
    else
      set_status "${status_var}" "downloaded"
    fi
  else
    set_status "${status_var}" "missing"
  fi
}

check_and_optionally_download_videos() {
  local had_video1=0
  local had_video2=0
  local had_video3=0

  report_video_status "${VIDEO1_PATH}" "video1.mp4"
  report_video_status "${VIDEO2_PATH}" "video2.mp4"
  report_video_status "${VIDEO3_PATH}" "video3.mp4"

  if file_present "${VIDEO1_PATH}"; then
    had_video1=1
  fi
  if file_present "${VIDEO2_PATH}"; then
    had_video2=1
  fi
  if file_present "${VIDEO3_PATH}"; then
    had_video3=1
  fi

  if [[ "${WITH_VIDEOS}" -eq 1 ]]; then
    log "Ensuring input videos"
    if [[ "${FORCE}" -eq 1 ]]; then
      .venv/bin/python scripts/download_videos.py --force
    else
      .venv/bin/python scripts/download_videos.py
    fi
  else
    if [[ "${had_video1}" -ne 1 ]]; then
      warn "video1.mp4: missing, run ./scripts/setup.sh --with-videos to download"
    fi
    if [[ "${had_video2}" -ne 1 ]]; then
      warn "video2.mp4: missing, run ./scripts/setup.sh --with-videos to download"
    fi
    if [[ "${had_video3}" -ne 1 ]]; then
      warn "video3.mp4: missing, run ./scripts/setup.sh --with-videos to download"
    fi
  fi

  resolve_status_after_helper "${VIDEO1_PATH}" "VIDEO1_STATUS" "${had_video1}"
  resolve_status_after_helper "${VIDEO2_PATH}" "VIDEO2_STATUS" "${had_video2}"
  resolve_status_after_helper "${VIDEO3_PATH}" "VIDEO3_STATUS" "${had_video3}"
}

report_model_status() {
  if file_present "${OPENCV_MODEL_PATH}" && file_present "${OPENCV_ANCHORS_PATH}"; then
    log "OpenCV DNN model: found (${OPENCV_MODEL_PATH}, ${OPENCV_ANCHORS_PATH})"
  else
    warn "OpenCV DNN model: missing (${OPENCV_MODEL_PATH} and/or ${OPENCV_ANCHORS_PATH})"
  fi

  if file_present "${YOLO_WEIGHTS_PATH}"; then
    log "YOLO weights: found (${YOLO_WEIGHTS_PATH})"
  else
    warn "YOLO weights: missing (${YOLO_WEIGHTS_PATH})"
  fi
}

check_versioned_model_assets() {
  report_model_status

  if [[ "${SKIP_MODELS}" -ne 1 ]]; then
    if file_present "${OPENCV_MODEL_PATH}" && file_present "${OPENCV_ANCHORS_PATH}"; then
      set_status "OPENCV_MODEL_STATUS" "present"
    else
      set_status "OPENCV_MODEL_STATUS" "missing"
      warn "OpenCV DNN model files are expected in git under models/. Recovery: .venv/bin/python scripts/download_models.py"
    fi
  else
    set_status "OPENCV_MODEL_STATUS" "skipped"
  fi

  if [[ "${SKIP_YOLO_WEIGHTS}" -ne 1 ]]; then
    if file_present "${YOLO_WEIGHTS_PATH}"; then
      set_status "YOLO_WEIGHTS_STATUS" "present"
    else
      set_status "YOLO_WEIGHTS_STATUS" "missing"
      warn "YOLO weights are expected in git under models/. Recovery: .venv/bin/python scripts/download_yolo_weights.py"
    fi
  else
    set_status "YOLO_WEIGHTS_STATUS" "skipped"
  fi
}

print_summary() {
  cat <<'EOF'

Setup complete.

Status summary:
EOF
  printf '  - video1.mp4: %s\n' "${VIDEO1_STATUS}"
  printf '  - video2.mp4: %s\n' "${VIDEO2_STATUS}"
  printf '  - video3.mp4: %s\n' "${VIDEO3_STATUS}"
  printf '  - opencv-dnn model: %s\n' "${OPENCV_MODEL_STATUS}"
  printf '  - yolo runtime: %s\n' "${YOLO_RUNTIME_STATUS}"
  printf '  - yolo weights: %s\n' "${YOLO_WEIGHTS_STATUS}"
  cat <<'EOF'

Examples:
  .venv/bin/python main.py --video data/video1.mp4
  .venv/bin/python main.py --video data/video1.mp4 --detector motion
  .venv/bin/python scripts/html_report_v01.py --out-dir out/v01
  ./scripts/setup.sh --with-videos
EOF
}

main() {
  parse_args "$@"
  require_cmd python3
  ensure_directories
  ensure_venv
  install_base_requirements
  install_yolo_runtime
  check_ffmpeg
  check_and_optionally_download_videos
  check_versioned_model_assets
  print_summary
}

main "$@"
