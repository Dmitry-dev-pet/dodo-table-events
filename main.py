"""
Тестовое задание (TZ v0.1): прототип детекции "уборки столика" по видео.

Что делает скрипт:
  - Берёт 1 видео с фиксированной камеры.
  - Пользователь один раз выделяет ROI выбранного столика (cv2.selectROI) и сохраняет в JSON.
  - На каждом кадре оценивает "есть ли человек в зоне" (простая эвристика):
      - motion: вычитание фона (MOG2) внутри ROI
      - opencv-dnn: готовая ONNX модель (MediaPipe Person Detection), через cv2.dnn
      - yolo: Ultralytics YOLO (опционально, зависит от окружения)
  - Строит события:
      - EMPTY      (в зоне нет человека)
      - OCCUPIED   (в зоне есть человек)
      - APPROACH   (первое появление человека после периода пустоты)
  - Сохраняет:
      - out/<run>/output.mp4  (ROI коробка меняет цвет)
      - out/<run>/events.csv  (таблица событий)
      - out/<run>/report.txt  (краткая статистика по метрике ожидания)

Запуск:
  python main.py --video data/video1.mp4

Python: 3.8+
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

#
# Важно (требование ТЗ):
# В исходном ТЗ явно написано "один хорошо прокомментированный main.py".
# Поэтому все core-хелперы (FSM/геометрия/репортинг/OpenCV-DNN) находятся в этом файле.

# Keep caches inside the repo by default (helps avoid permission warnings on locked home dirs).
_OUT_CACHE = Path("out") / ".cache"
os.environ.setdefault("XDG_CACHE_HOME", str(_OUT_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(_OUT_CACHE / "matplotlib"))


# -------------------- Значения по умолчанию --------------------

DEFAULT_OUT_DIR = Path("out/v01")
DEFAULT_DETECTOR = "all"  # run all available detectors; YOLO (if installed) drives the FSM by default

# ROI overlap heuristic:
# in_zone_now = any( intersection(person_bbox, roi) / area(roi) >= overlap_min )
DEFAULT_OVERLAP_MIN = 0.10

# State smoothing (seconds)
DEFAULT_T_ENTER_SEC = 20.0
DEFAULT_T_EXIT_SEC = 20.0
# Debounce for APPROACH: require continuous presence for N seconds before writing APPROACH.
DEFAULT_T_APPROACH_SEC = 1.0
# Minimum duration of empty gap (sec) right before we count an APPROACH.
# Per TZ: "подход" = появление человека после пустоты. По умолчанию: любая пустота (0s).
DEFAULT_T_EMPTY_MIN_SEC = 0.0

# Motion detector inside ROI
DEFAULT_MOTION_FG_FRAC_MIN = 0.015

# DNN (OpenCV) model paths (download via `rye run dodo-download-models`)
DEFAULT_OPENCV_DNN_MODEL = "models/opencv_dnn/person_detection_mediapipe_2023mar.onnx"
DEFAULT_OPENCV_DNN_ANCHORS = "models/opencv_dnn/person_detection_mediapipe_2023mar.anchors.npy"
DEFAULT_OPENCV_DNN_SCORE_MIN = 0.5
DEFAULT_OPENCV_DNN_NMS_IOU = 0.3
DEFAULT_OPENCV_DNN_TOPK = 10

# YOLO (Ultralytics)
DEFAULT_YOLO_WEIGHTS = "models/yolov8n.pt"
DEFAULT_YOLO_DEVICE = "auto"
DEFAULT_YOLO_IMGSZ = 640
DEFAULT_YOLO_ROTATE = "auto"  # auto/none/90cw/90ccw/180 (rotation applied only for YOLO inference)
# Raw YOLO threshold: used only to *collect* candidate boxes for debugging/overlay.
# Final "used" boxes (FSM/overlap) are filtered by DEFAULT_PERSON_CONF_USED.
DEFAULT_PERSON_CONF_MIN = 0.05
# Для выбранного ROI YOLO нередко даёт низкий confidence (особенно сидящих/частично закрытых людей).
# Стабильность обеспечиваем временной фильтрацией (t_enter/t_exit), поэтому порог "used" держим ниже.
DEFAULT_PERSON_CONF_USED = 0.15
DEFAULT_PERSON_MIN_AREA = 1000.0  # helps ignore tiny false positives
DEFAULT_INFER_EVERY = 5
COCO_PERSON_CLASS_ID = 0

# Output codec: try H.264 first (best for HTML5), fallback to mp4v.
DEFAULT_FOURCC = ""


# -------------------- Вспомогательные функции (в этом файле по ТЗ) --------------------


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def clip(self, w: int, h: int) -> "BBox":
        return BBox(
            x1=float(np.clip(self.x1, 0, max(0, w - 1))),
            y1=float(np.clip(self.y1, 0, max(0, h - 1))),
            x2=float(np.clip(self.x2, 0, max(0, w - 1))),
            y2=float(np.clip(self.y2, 0, max(0, h - 1))),
        )

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def as_int_xyxy(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


def intersection_area(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


@dataclass
class TableFSM:
    """
    Простая FSM для состояния столика: EMPTY/OCCUPIED + событие APPROACH.

    - OCCUPIED: in_zone держится >= t_enter
    - EMPTY:    in_zone отсутствует >= t_exit
    - APPROACH: in_zone появился после пустоты, и держится >= t_approach.

    Важно: APPROACH эмитится не чаще одного раза на подтверждённый интервал EMPTY.
    Короткие колебания in_zone внутри того же состояния EMPTY не создают новые APPROACH.
    """

    state: str = "EMPTY"  # EMPTY | OCCUPIED
    enter_hold: float = 0.0
    exit_hold: float = 0.0
    empty_start_ts: float = 0.0
    empty_gap_start_ts: float = 0.0
    approach_emitted: bool = False
    approach_hold: float = 0.0
    prev_in_zone_now: bool = False

    def step(
        self,
        *,
        ts: float,
        dt: float,
        in_zone_now: bool,
        t_enter: float,
        t_exit: float,
        t_approach: float,
        t_empty_min: float,
        events: List[Dict[str, object]],
        frame_idx: int,
    ) -> None:
        dt = float(dt) if (math.isfinite(dt) and dt > 0) else 0.0

        if self.state == "EMPTY":
            if not bool(in_zone_now):
                self.enter_hold = 0.0
                self.approach_hold = 0.0
            else:
                if not bool(self.approach_emitted):
                    self.approach_hold += dt
                    if self.approach_hold >= float(t_approach):
                        empty_dur = max(0.0, float(ts - self.empty_start_ts))
                        if empty_dur >= float(t_empty_min):
                            events.append({"ts_sec": float(ts), "frame_idx": int(frame_idx), "event_type": "APPROACH"})
                            self.approach_emitted = True

                self.enter_hold += dt
                if self.enter_hold >= float(t_enter):
                    self.state = "OCCUPIED"
                    self.exit_hold = 0.0
                    self.enter_hold = 0.0
                    events.append({"ts_sec": float(ts), "frame_idx": int(frame_idx), "event_type": "OCCUPIED"})
        else:
            if in_zone_now:
                self.exit_hold = 0.0
            else:
                self.exit_hold += dt
                if self.exit_hold >= float(t_exit):
                    self.state = "EMPTY"
                    self.exit_hold = 0.0
                    self.enter_hold = 0.0
                    self.empty_start_ts = float(ts)
                    self.empty_gap_start_ts = float(ts)
                    self.approach_emitted = False
                    self.approach_hold = 0.0
                    events.append({"ts_sec": float(ts), "frame_idx": int(frame_idx), "event_type": "EMPTY"})

        self.prev_in_zone_now = bool(in_zone_now)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, topk: int) -> List[int]:
    if boxes.size == 0:
        return []
    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if topk > 0 and len(keep) >= int(topk):
            break
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = np.maximum(areas[i] + areas[order[1:]] - inter, 0.0)
        iou = np.zeros_like(union, dtype=np.float32)
        np.divide(inter, union, out=iou, where=union > 0)
        inds = np.where(iou <= float(iou_thr))[0]
        order = order[inds + 1]
    return keep


@dataclass
class OpenCVDNNPersonDet:
    """
    Детектор людей MediaPipe Person Detection через OpenCV DNN.

    Работает на CPU и подходит для Python 3.8 без torch.
    """

    model_path: Path
    anchors_path: Path

    def __post_init__(self) -> None:
        self.input_size = np.array([224, 224], dtype=np.float32)  # wh
        self.net = cv2.dnn.readNet(str(self.model_path))
        anchors = np.load(str(self.anchors_path))
        if anchors.ndim != 2 or anchors.shape[1] != 2:
            raise SystemExit(f"Invalid anchors file: {self.anchors_path} shape={anchors.shape}")
        self.anchors = anchors.astype(np.float32)

    def _preprocess(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        pad_bias = np.array([0.0, 0.0], dtype=np.float32)  # left, top in resized space
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) * 2.0  # [-1, 1]

        h, w = img.shape[:2]
        ratio = float(min(self.input_size[1] / float(w), self.input_size[0] / float(h)))
        rh = int(h * ratio)
        rw = int(w * ratio)
        if rh != int(self.input_size[0]) or rw != int(self.input_size[1]):
            img = cv2.resize(img, (rw, rh))
            pad_h = int(self.input_size[0]) - rh
            pad_w = int(self.input_size[1]) - rw
            left = pad_w // 2
            top = pad_h // 2
            right = pad_w - left
            bottom = pad_h - top
            pad_bias[0] = float(left)
            pad_bias[1] = float(top)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))

        blob = np.transpose(img, [2, 0, 1])[np.newaxis, :, :, :]  # NCHW
        pad_bias_orig = (pad_bias / max(ratio, 1e-9)).astype(np.float32)  # in original pixels
        return blob, pad_bias_orig, ratio

    def infer(self, image_bgr: np.ndarray, *, score_min: float, nms_iou: float, topk: int) -> List[Tuple[BBox, float]]:
        h, w = image_bgr.shape[:2]
        blob, pad_bias, _ = self._preprocess(image_bgr)
        self.net.setInput(blob)
        out0, out1 = self.net.forward(self.net.getUnconnectedOutLayersNames())
        score_raw = out1[0, :, 0].astype(np.float32)
        score = _sigmoid(np.clip(score_raw.astype(np.float64), -100.0, 100.0)).astype(np.float32)
        mask = score >= float(score_min)
        if not np.any(mask):
            return []
        box_delta = out0[0, :, 0:4].astype(np.float32)[mask]
        score_f = score[mask]
        anchors = self.anchors[mask]

        cxy_delta = box_delta[:, :2] / self.input_size
        wh_delta = box_delta[:, 2:] / self.input_size
        scale = float(max(w, h))
        xy1 = (cxy_delta - wh_delta / 2.0 + anchors) * scale
        xy2 = (cxy_delta + wh_delta / 2.0 + anchors) * scale
        boxes = np.concatenate([xy1, xy2], axis=1)
        boxes -= np.array([pad_bias[0], pad_bias[1], pad_bias[0], pad_bias[1]], dtype=np.float32)
        keep = _nms_xyxy(boxes, score_f, float(nms_iou), int(topk))
        out: List[Tuple[BBox, float]] = []
        for i in keep:
            x1, y1, x2, y2 = boxes[int(i)].tolist()
            out.append((BBox(float(x1), float(y1), float(x2), float(y2)).clip(w, h), float(score_f[int(i)])))
        return out


@dataclass(frozen=True)
class WaitMetric:
    pairs: int
    mean_sec: float
    median_sec: float
    p90_sec: float


@dataclass(frozen=True)
class TouchMetric:
    min_touch_sec: float
    touches: int
    landings: int
    mean_touch_sec: Optional[float]
    mean_gap_sec: Optional[float]


def compute_waits_empty_to_next_approach(df_events: pd.DataFrame) -> List[float]:
    waits: List[float] = []
    if df_events is None or df_events.empty:
        return waits

    if "event_type" not in df_events.columns or "ts_sec" not in df_events.columns:
        return waits

    order_cols = [col for col in ("ts_sec", "frame_idx") if col in df_events.columns]
    df_sorted = df_events.sort_values(order_cols, kind="stable").reset_index(drop=True)

    # TZ asks for "time between guest leaving and the next approach", so we only
    # use EMPTY events that happened after a real occupied period. This excludes
    # the synthetic start-state EMPTY that we add at frame 0 for self-contained logs.
    seen_occupied = False
    empties_list: List[float] = []
    for row in df_sorted.itertuples(index=False):
        event_type = str(getattr(row, "event_type", "") or "")
        if event_type == "OCCUPIED":
            seen_occupied = True
        elif event_type == "EMPTY" and seen_occupied:
            empties_list.append(float(getattr(row, "ts_sec")))

    empties = np.array(empties_list, dtype=np.float64)
    approaches = df_events[df_events["event_type"] == "APPROACH"][["ts_sec"]].to_numpy().reshape(-1)
    approaches = np.sort(approaches.astype(np.float64)) if approaches.size else approaches
    for t_empty in empties:
        if approaches.size == 0:
            break
        idx = int(np.searchsorted(approaches, float(t_empty) + 1e-9, side="left"))
        if idx < int(approaches.size):
            waits.append(float(approaches[idx] - float(t_empty)))
    return waits


def summarize_waits(waits: List[float]) -> WaitMetric:
    arr = np.array(waits, dtype=np.float64)
    return WaitMetric(
        pairs=int(arr.size),
        mean_sec=float(arr.mean()) if arr.size else 0.0,
        median_sec=float(np.median(arr)) if arr.size else 0.0,
        p90_sec=float(np.quantile(arr, 0.9)) if arr.size else 0.0,
    )


def compute_touch_metrics(df_raw: pd.DataFrame, *, min_touch_sec: float) -> TouchMetric:
    if df_raw is None or df_raw.empty:
        return TouchMetric(
            min_touch_sec=float(min_touch_sec),
            touches=0,
            landings=0,
            mean_touch_sec=None,
            mean_gap_sec=None,
        )

    required_cols = {"ts_sec", "in_zone_now", "state"}
    if not required_cols.issubset(set(df_raw.columns)):
        return TouchMetric(
            min_touch_sec=float(min_touch_sec),
            touches=0,
            landings=0,
            mean_touch_sec=None,
            mean_gap_sec=None,
        )

    order_cols = [col for col in ("ts_sec", "frame_idx") if col in df_raw.columns]
    df_sorted = df_raw.sort_values(order_cols, kind="stable").reset_index(drop=True)

    bounced_intervals: List[Tuple[float, float]] = []
    landings = 0

    active = False
    touch_start_ts = 0.0

    for row in df_sorted.itertuples(index=False):
        ts = float(getattr(row, "ts_sec"))
        in_zone = bool(getattr(row, "in_zone_now"))
        state = str(getattr(row, "state", "") or "")

        if not active:
            if state == "EMPTY" and in_zone:
                active = True
                touch_start_ts = ts
            continue

        if state == "OCCUPIED" and in_zone:
            dur = max(0.0, ts - touch_start_ts)
            if dur >= float(min_touch_sec):
                landings += 1
            active = False
            continue

        if (state == "EMPTY" and not in_zone) or (state != "EMPTY" and state != "OCCUPIED"):
            dur = max(0.0, ts - touch_start_ts)
            if dur >= float(min_touch_sec):
                bounced_intervals.append((float(touch_start_ts), float(ts)))
            active = False
            continue

    if active:
        dur = max(0.0, float(df_sorted.iloc[-1]["ts_sec"]) - touch_start_ts)
        if dur >= float(min_touch_sec):
            bounced_intervals.append((float(touch_start_ts), float(df_sorted.iloc[-1]["ts_sec"])))

    touch_durations = [max(0.0, end - start) for start, end in bounced_intervals]
    gaps = [
        max(0.0, bounced_intervals[i][0] - bounced_intervals[i - 1][1])
        for i in range(1, len(bounced_intervals))
    ]

    return TouchMetric(
        min_touch_sec=float(min_touch_sec),
        touches=int(len(bounced_intervals)),
        landings=int(landings),
        mean_touch_sec=float(np.mean(np.array(touch_durations, dtype=np.float64))) if touch_durations else None,
        mean_gap_sec=float(np.mean(np.array(gaps, dtype=np.float64))) if gaps else None,
    )


def build_report_lines(
    *,
    video_path: Path,
    fps: float,
    roi_xyxy: Tuple[int, int, int, int],
    detector_mode: str,
    primary_detector: str,
    enabled: Dict[str, bool],
    yolo_info: Dict[str, object],
    opencv_dnn_info: Dict[str, object],
    motion_info: Dict[str, object],
    overlap_min: float,
    t_enter: float,
    t_exit: float,
    t_approach: float,
    t_empty_min: float,
    infer_every: int,
    waits: List[float],
    touch_metrics: TouchMetric,
    perf_avg_detect_ms: float,
    perf_est_fps: float,
    perf_total_sec: float,
) -> List[str]:
    lines: List[str] = []
    lines.append("Dodo table events report (TZ v0.1)")
    lines.append("")
    lines.append(f"video: {video_path}")
    lines.append(f"fps: {fps:.3f}")
    lines.append(f"roi_xyxy: {roi_xyxy}")
    lines.append(f"detector_mode: {detector_mode}")
    lines.append(f"primary_detector: {primary_detector}")
    lines.append(
        "enabled: "
        f"motion={int(bool(enabled.get('motion')))} "
        f"opencv-dnn={int(bool(enabled.get('opencv-dnn')))} "
        f"yolo={int(bool(enabled.get('yolo')))}"
    )

    if bool(enabled.get("yolo")):
        lines.append(f"  weights: {yolo_info.get('weights', '')}")
        lines.append(f"  device: {yolo_info.get('device', '')}")
        lines.append(f"  torch_mps_built: {int(bool(yolo_info.get('torch_mps_built', False)))}")
        lines.append(f"  torch_mps_available: {int(bool(yolo_info.get('torch_mps_available', False)))}")
        lines.append(f"  imgsz: {int(yolo_info.get('imgsz', 0) or 0)}")
        lines.append(f"  yolo_rotate: {str(yolo_info.get('yolo_rotate', ''))}")
        lines.append(f"  person_conf_min_raw: {float(yolo_info.get('person_conf_min_raw', 0.0) or 0.0):.3f}")
        lines.append(f"  person_conf_used: {float(yolo_info.get('person_conf_used', 0.0) or 0.0):.3f}")
        lines.append(f"  person_min_area: {float(yolo_info.get('person_min_area', 0.0) or 0.0):.1f}")

    if bool(enabled.get("opencv-dnn")):
        lines.append(f"  model: {opencv_dnn_info.get('model', '')}")
        lines.append(f"  anchors: {opencv_dnn_info.get('anchors', '')}")
        lines.append(f"  score_min: {float(opencv_dnn_info.get('score_min', 0.0) or 0.0):.3f}")

    if bool(enabled.get("motion")):
        lines.append(f"  fg_frac_min: {float(motion_info.get('fg_frac_min', 0.0) or 0.0):.4f}")

    lines.append(f"  overlap_min: {float(overlap_min):.3f}")
    lines.append(f"  t_enter_sec: {float(t_enter):.3f}")
    lines.append(f"  t_exit_sec: {float(t_exit):.3f}")
    lines.append(f"  t_approach_sec: {float(t_approach):.3f}")
    lines.append(f"  t_empty_min_sec: {float(t_empty_min):.3f}")
    lines.append(f"  approach_requires_empty_total_sec: {(float(t_exit) + float(t_empty_min)):.3f}")
    lines.append(f"  infer_every: {int(infer_every)}")
    lines.append("")

    if waits:
        m = summarize_waits(waits)
        lines.append("wait_metric_empty_to_next_approach:")
        lines.append(f"  pairs: {int(m.pairs)}")
        lines.append(f"  mean_sec: {float(m.mean_sec):.3f}")
        lines.append(f"  median_sec: {float(m.median_sec):.3f}")
        lines.append(f"  p90_sec: {float(m.p90_sec):.3f}")
    else:
        lines.append("wait_metric_empty_to_next_approach: insufficient events (need both EMPTY and APPROACH)")

    lines.append("")
    lines.append("touch_metrics:")
    lines.append(f"  min_touch_sec: {float(touch_metrics.min_touch_sec):.3f}")
    lines.append(f"  touches: {int(touch_metrics.touches)}")
    lines.append(f"  landings: {int(touch_metrics.landings)}")
    if touch_metrics.mean_touch_sec is not None:
        lines.append(f"  mean_touch_sec: {float(touch_metrics.mean_touch_sec):.3f}")
    else:
        lines.append("  mean_touch_sec: n/a")
    if touch_metrics.mean_gap_sec is not None:
        lines.append(f"  mean_gap_sec: {float(touch_metrics.mean_gap_sec):.3f}")
    else:
        lines.append("  mean_gap_sec: n/a")

    if perf_avg_detect_ms > 0:
        lines.append("")
        lines.append(f"perf_avg_detect_ms: {float(perf_avg_detect_ms):.3f}")
        lines.append(f"perf_est_fps: {float(perf_est_fps):.2f}")
        lines.append(f"perf_total_sec: {float(perf_total_sec):.2f}")

    return lines


def _rotate_frame_for_yolo(frame: np.ndarray, rotate: str) -> np.ndarray:
    r = str(rotate or "none").strip().lower()
    if r in ("none", "0", "off", "false"):
        return frame
    if r in ("180", "rot180"):
        return cv2.rotate(frame, cv2.ROTATE_180)
    if r in ("90cw", "cw", "rot90cw"):
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if r in ("90ccw", "ccw", "rot90ccw"):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _yolo_unrotate_point(xr: float, yr: float, rotate: str, w: int, h: int) -> Tuple[float, float]:
    """
    Переводит точку из координат изображения (после поворота для YOLO-инференса)
    обратно в координаты исходного кадра.

    Важно: используем (w-1)/(h-1), чтобы совпасть с семантикой поворота OpenCV (индексы пикселей).
    Мелкая погрешность на 1 пиксель на границах bbox не критична для нашей эвристики overlap.
    """

    r = str(rotate or "none").strip().lower()
    if r in ("none", "0", "off", "false"):
        return (float(xr), float(yr))
    if r in ("180", "rot180"):
        return (float((w - 1) - xr), float((h - 1) - yr))
    if r in ("90ccw", "ccw", "rot90ccw"):
        # поворот 90°: rotated (w'=h, h'=w) -> исходные координаты
        return (float((w - 1) - yr), float(xr))
    if r in ("90cw", "cw", "rot90cw"):
        return (float(yr), float((h - 1) - xr))
    return (float(xr), float(yr))


def _yolo_unrotate_bbox_xyxy(x1: float, y1: float, x2: float, y2: float, rotate: str, w: int, h: int) -> BBox:
    pts = [
        _yolo_unrotate_point(float(x1), float(y1), rotate, w, h),
        _yolo_unrotate_point(float(x2), float(y1), rotate, w, h),
        _yolo_unrotate_point(float(x2), float(y2), rotate, w, h),
        _yolo_unrotate_point(float(x1), float(y2), rotate, w, h),
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _auto_choose_yolo_rotate(
    *,
    video_path: Path,
    yolo_model,
    fps: float,
    device: str,
    imgsz: int,
    conf_min: float,
    conf_used: float,
    sample_sec: float = 30.0,
    n_samples: int = 12,
) -> str:
    """
    Приблизительное определение ориентации видео, если в файле нет корректных метаданных поворота.

    Запускает YOLO на небольшом наборе кадров в 4 ориентациях и выбирает вариант с лучшим средним
    значением "максимальный confidence по человеку".

    Это эвристика: может не сработать, если в сэмпле вообще нет людей.
    """

    rotations = ["none", "90ccw", "90cw", "180"]
    scores: Dict[str, List[float]] = {r: [] for r in rotations}
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return "none"
        total = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)))
        max_idx = max(0, min(total - 1, int(round(float(fps) * float(sample_sec))))) if total > 0 else 0
        if max_idx <= 0:
            idxs = [0]
        else:
            idxs = np.linspace(0, max_idx, num=max(1, int(n_samples)), dtype=int).tolist()

        predict_kwargs = {
            "conf": float(conf_min),
            "imgsz": int(imgsz),
            "device": str(device),
            "max_det": 50,
            "verbose": False,
        }

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            for r in rotations:
                fr = _rotate_frame_for_yolo(frame, r)
                try:
                    res = yolo_model.predict(fr, cls=[COCO_PERSON_CLASS_ID], **predict_kwargs)  # type: ignore[union-attr]
                except Exception:
                    res = yolo_model.predict(fr, classes=[COCO_PERSON_CLASS_ID], **predict_kwargs)  # type: ignore[union-attr]
                best = 0.0
                for rr in res:
                    if rr.boxes is None or len(rr.boxes) == 0:
                        continue
                    conf_all = rr.boxes.conf.detach().cpu().numpy()
                    if conf_all.size:
                        best = max(best, float(np.max(conf_all)))
                scores[r].append(float(best))
        cap.release()
    except Exception:
        return "none"

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    means = {r: _mean(v) for (r, v) in scores.items()}
    best_r = max(rotations, key=lambda r: float(means.get(r, 0.0)))
    base = float(means.get("none", 0.0))
    best = float(means.get(best_r, 0.0))

    # Защита от ложных переключений: уходим от "none" только если результат заметно лучше.
    # (Если людей в сэмпле нет, все значения ~0 и мы оставляем "none".)
    if best_r != "none":
        if (best < max(float(conf_used), 0.15)) or (best < base + 0.05):
            return "none"

    # Отладочный лог для прозрачности выбора.
    try:
        parts = " ".join([f"{r}={means[r]:.3f}" for r in rotations])
        print(f"YOLO auto-rotate: choose={best_r} ({parts})")
    except Exception:
        pass

    return str(best_r)


# -------------------- ROI: загрузка/сохранение --------------------


def _load_roi(path: Path) -> Optional[BBox]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        xyxy = obj.get("roi_xyxy")
        if not (isinstance(xyxy, list) and len(xyxy) == 4):
            return None
        x1, y1, x2, y2 = map(float, xyxy)
        return BBox(x1, y1, x2, y2)
    except Exception:
        return None


def _save_roi(path: Path, roi: BBox) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"roi_xyxy": list(map(int, roi.as_int_xyxy()))}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _select_roi(frame_bgr: np.ndarray) -> BBox:
    # cv2.selectROI returns (x, y, w, h)
    r = cv2.selectROI("Выберите ROI столика", frame_bgr, fromCenter=False, showCrosshair=True)
    try:
        cv2.destroyWindow("Выберите ROI столика")
    except Exception:
        pass
    x, y, w, h = map(int, r)
    if w <= 0 or h <= 0:
        raise SystemExit("ROI не выбран (пустой прямоугольник).")
    return BBox(float(x), float(y), float(x + w), float(y + h))


def _parse_roi_arg(s: str) -> Optional[BBox]:
    s = str(s).strip()
    if not s:
        return None
    try:
        parts = [float(x.strip()) for x in s.split(",")]
        if len(parts) != 4:
            return None
        return BBox(parts[0], parts[1], parts[2], parts[3])
    except Exception:
        return None


# -------------------- Запись видео --------------------


def _open_video_writer(path: Path, *, fps: float, size: Tuple[int, int], fourcc: str) -> Tuple[cv2.VideoWriter, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = int(size[0]), int(size[1])
    # Если пользователь явно указал кодек — пробуем только его.
    if fourcc:
        c = cv2.VideoWriter_fourcc(*fourcc)
        wr = cv2.VideoWriter(str(path), c, float(fps), (w, h))
        if not wr.isOpened():
            raise SystemExit(f"Не удалось открыть VideoWriter с fourcc={fourcc}.")
        return wr, fourcc

    # Авто-режим: пробуем несколько популярных MP4 кодеков.
    for try_fourcc in ["avc1", "H264", "mp4v"]:
        c = cv2.VideoWriter_fourcc(*try_fourcc)
        wr = cv2.VideoWriter(str(path), c, float(fps), (w, h))
        if wr.isOpened():
            return wr, try_fourcc
    raise SystemExit("Не удалось открыть VideoWriter (пробовали: avc1, H264, mp4v).")


# -------------------- Параметры запуска (CLI) --------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TZ v0.1: детекция событий по столику из видео (Python 3.8+).")
    p.add_argument("--video", required=True, type=str, help="Путь к входному видео")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), type=str, help="Папка вывода (по умолчанию: out/v01)")

    p.add_argument(
        "--detector",
        default=DEFAULT_DETECTOR,
        choices=["all", "auto", "motion", "opencv-dnn", "yolo"],
        help="Детектор присутствия человека: all/auto/motion/opencv-dnn/yolo",
    )

    p.add_argument("--roi-json", default="", type=str, help="Путь к кэшу ROI (по умолчанию: <out-dir>/roi.json)")
    p.add_argument(
        "--roi",
        default="",
        type=str,
        help="Необязательный ROI в пикселях: 'x1,y1,x2,y2' (без cv2.selectROI).",
    )
    p.add_argument(
        "--force-roi",
        action="store_true",
        default=False,
        help="Выбрать ROI заново, даже если roi.json уже существует",
    )

    p.add_argument("--overlap-min", default=DEFAULT_OVERLAP_MIN, type=float, help="Порог overlap для in_zone")
    p.add_argument("--t-enter", default=DEFAULT_T_ENTER_SEC, type=float, help="Секунд, чтобы пометить OCCUPIED")
    p.add_argument("--t-exit", default=DEFAULT_T_EXIT_SEC, type=float, help="Секунд, чтобы пометить EMPTY")
    p.add_argument(
        "--t-approach",
        default=DEFAULT_T_APPROACH_SEC,
        type=float,
        help="Секунд непрерывного присутствия перед событием APPROACH",
    )
    p.add_argument(
        "--t-empty-min",
        default=DEFAULT_T_EMPTY_MIN_SEC,
        type=float,
        help="Мин. длительность устойчивого EMPTY (сек) перед APPROACH",
    )

    p.add_argument("--fourcc", default=DEFAULT_FOURCC, type=str, help="Переопределить fourcc (например: avc1/mp4v)")
    p.add_argument("--no-video", action="store_true", default=False, help="Не записывать выходное видео")
    p.add_argument("--max-seconds", default=0.0, type=float, help="Дебаг: обработать только первые N секунд (0=всё)")

    # motion
    p.add_argument(
        "--motion-fg-frac-min",
        default=DEFAULT_MOTION_FG_FRAC_MIN,
        type=float,
        help="Порог motion внутри ROI",
    )

    # opencv-dnn
    p.add_argument(
        "--opencv-dnn-model",
        default=DEFAULT_OPENCV_DNN_MODEL,
        type=str,
        help="Путь к ONNX модели для OpenCV DNN",
    )
    p.add_argument(
        "--opencv-dnn-anchors",
        default=DEFAULT_OPENCV_DNN_ANCHORS,
        type=str,
        help="Путь к anchors .npy для OpenCV DNN",
    )
    p.add_argument(
        "--opencv-dnn-score-min",
        default=DEFAULT_OPENCV_DNN_SCORE_MIN,
        type=float,
        help="Порог score для DNN",
    )
    p.add_argument("--opencv-dnn-nms-iou", default=DEFAULT_OPENCV_DNN_NMS_IOU, type=float, help="NMS IoU для DNN")
    p.add_argument("--opencv-dnn-topk", default=DEFAULT_OPENCV_DNN_TOPK, type=int, help="Top-K боксов для DNN")

    # yolo
    p.add_argument("--weights", default=DEFAULT_YOLO_WEIGHTS, type=str, help="Путь к весам Ultralytics YOLO")
    p.add_argument(
        "--device",
        default=DEFAULT_YOLO_DEVICE,
        type=str,
        help="Устройство YOLO: auto/cpu/mps (auto использует mps на Apple Silicon, если доступно)",
    )
    p.add_argument("--imgsz", default=DEFAULT_YOLO_IMGSZ, type=int, help="Размер изображения для YOLO-инференса")
    p.add_argument(
        "--yolo-rotate",
        default=DEFAULT_YOLO_ROTATE,
        type=str,
        help="Поворот кадров только для YOLO-инференса: none/90cw/90ccw/180",
    )
    p.add_argument(
        "--person-conf-min",
        default=DEFAULT_PERSON_CONF_MIN,
        type=float,
        help="Порог conf YOLO для raw боксов",
    )
    p.add_argument(
        "--person-conf-used",
        default=DEFAULT_PERSON_CONF_USED,
        type=float,
        help="Порог conf YOLO для 'used' боксов (FSM/overlap).",
    )
    p.add_argument(
        "--person-min-area",
        default=DEFAULT_PERSON_MIN_AREA,
        type=float,
        help="Игнорировать слишком маленькие боксы (px^2)",
    )
    p.add_argument(
        "--infer-every",
        default=DEFAULT_INFER_EVERY,
        type=int,
        help="Запускать детектор раз в N кадров",
    )

    return p.parse_args()


def _resolve_device(device: str) -> str:
    device = str(device).strip().lower()
    if device and device != "auto":
        return device
    # best-effort: используем mps, если доступно
    try:
        import torch  # type: ignore

        # Важно (ТЗ / Python 3.8):
        # В этом репозитории мы поддерживаем legacy torch (1.13.1), чтобы сохранить совместимость с Python 3.8.
        # У такого torch MPS backend часто нестабилен в пост-обработке Ultralytics YOLO
        # (наблюдали «кривые» bbox/конфиденсы ~1.0). Поэтому для корректности AUTO выбирает MPS только на torch>=2.
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            try:
                major = int(str(getattr(torch, "__version__", "0")).split(".", 1)[0])
            except Exception:
                major = 0
            if major >= 2:
                return "mps"
    except Exception:
        pass
    return "cpu"


def _torch_mps_status() -> Dict[str, bool]:
    """
    Возвращает статус MPS backend для диагностики на Apple Silicon.

    На Python 3.8 часто используется legacy сборка torch; доступность MPS может отличаться в разных окружениях.
    """

    built = False
    available = False
    try:
        import torch  # type: ignore

        if getattr(torch.backends, "mps", None) is not None:
            built = bool(torch.backends.mps.is_built())
            available = bool(torch.backends.mps.is_available())
    except Exception:
        pass
    return {"built": built, "available": available}


# -------------------- Основной пайплайн --------------------


def _try_import_yolo():
    """
    Ленивый импорт, чтобы не тянуть тяжёлые зависимости Ultralytics, если YOLO не нужен.

    Возвращает класс YOLO или None.
    """

    try:
        from ultralytics import YOLO as YOLO_CLS  # type: ignore

        return YOLO_CLS
    except Exception:
        return None


def _load_or_select_roi(
    *,
    cap: cv2.VideoCapture,
    frame0: np.ndarray,
    frame_w: int,
    frame_h: int,
    roi_json: Path,
    roi_arg: str,
    force_roi: bool,
) -> BBox:
    roi: Optional[BBox] = None

    roi_arg_box = _parse_roi_arg(str(roi_arg))
    if roi_arg_box is not None:
        roi = roi_arg_box.clip(frame_w, frame_h)
        _save_roi(roi_json, roi)
        print(f"ROI: из --roi {roi.as_int_xyxy()} -> сохранено в {roi_json}")

    if (not bool(force_roi)) and roi_json.exists():
        roi = _load_roi(roi_json)
        if roi is not None:
            roi = roi.clip(frame_w, frame_h)
            print(f"ROI: загружено {roi.as_int_xyxy()} из {roi_json}")

    if roi is None:
        roi = _select_roi(frame0).clip(frame_w, frame_h)
        _save_roi(roi_json, roi)
        print(f"ROI: выбрано {roi.as_int_xyxy()} -> сохранено в {roi_json}")

    return roi


@dataclass
class DetectorContext:
    detector_arg: str
    detector_mode: str
    enabled: Dict[str, bool]
    primary_detector: str
    dnn: Optional[OpenCVDNNPersonDet]
    yolo_model: Optional[Any]
    yolo_device: str
    yolo_rotate: str
    mps_status: Dict[str, bool]


def _setup_detectors(*, args: argparse.Namespace, video_path: Path, fps: float) -> DetectorContext:
    detector_arg = str(args.detector)

    dnn_model = Path(str(args.opencv_dnn_model))
    dnn_anchors = Path(str(args.opencv_dnn_anchors))
    dnn_ready = dnn_model.exists() and dnn_anchors.exists()

    yolo_cls = _try_import_yolo()
    yolo_available = yolo_cls is not None

    detector_mode = detector_arg
    enabled = {"motion": False, "opencv-dnn": False, "yolo": False}

    if detector_arg == "auto":
        # Prefer YOLO when available, fallback to OpenCV-DNN, then motion.
        detector_mode = "yolo" if yolo_available else ("opencv-dnn" if dnn_ready else "motion")

    if detector_arg == "all":
        enabled["motion"] = True
        enabled["opencv-dnn"] = bool(dnn_ready)
        enabled["yolo"] = bool(yolo_available)
        # Primary detector drives FSM/ROI color.
        if enabled["yolo"]:
            primary_detector = "yolo"
        elif enabled["opencv-dnn"]:
            primary_detector = "opencv-dnn"
        else:
            primary_detector = "motion"
    else:
        primary_detector = detector_mode
        enabled["motion"] = detector_mode == "motion"
        enabled["opencv-dnn"] = detector_mode == "opencv-dnn"
        enabled["yolo"] = detector_mode == "yolo"

    # Instantiate detectors that need init.
    dnn: Optional[OpenCVDNNPersonDet] = None
    yolo_model: Optional[Any] = None
    yolo_device = "cpu"
    mps_status = {"built": False, "available": False}
    yolo_rotate_eff = "none"

    if enabled["opencv-dnn"]:
        if not dnn_ready:
            enabled["opencv-dnn"] = False
        else:
            dnn = OpenCVDNNPersonDet(model_path=dnn_model, anchors_path=dnn_anchors)

    if enabled["yolo"]:
        if yolo_cls is None:
            enabled["yolo"] = False
        else:
            yolo_device = _resolve_device(str(args.device))
            mps_status = _torch_mps_status()
            yolo_model = yolo_cls(str(args.weights))  # type: ignore[misc]
            yolo_rotate_eff = str(args.yolo_rotate or DEFAULT_YOLO_ROTATE).strip().lower()
            if yolo_rotate_eff == "auto":
                yolo_rotate_eff = _auto_choose_yolo_rotate(
                    video_path=video_path,
                    yolo_model=yolo_model,
                    fps=float(fps),
                    device=str(yolo_device),
                    imgsz=int(args.imgsz),
                    conf_min=float(args.person_conf_min),
                    conf_used=float(args.person_conf_used),
                )

    # Print config
    if detector_arg == "all":
        parts = []
        if enabled["motion"]:
            parts.append("motion")
        if enabled["opencv-dnn"]:
            parts.append("opencv-dnn")
        if enabled["yolo"]:
            parts.append("yolo")
        extra = f"primary={primary_detector}"
        if primary_detector == "yolo":
            extra += f" device={yolo_device} rotate={yolo_rotate_eff}"
        print(f"Detector: all ({', '.join(parts)}) {extra}")
    else:
        if enabled["opencv-dnn"] and dnn is not None:
            print(f"Detector: opencv-dnn (cpu) model={dnn_model.name}")
        elif enabled["yolo"] and yolo_model is not None:
            print(
                f"Detector: yolo device={yolo_device} weights={args.weights} "
                f"(torch.mps built={int(bool(mps_status['built']))} available={int(bool(mps_status['available']))})"
            )
        else:
            print("Detector: motion (MOG2)")

    return DetectorContext(
        detector_arg=detector_arg,
        detector_mode=detector_mode,
        enabled=enabled,
        primary_detector=primary_detector,
        dnn=dnn,
        yolo_model=yolo_model,
        yolo_device=yolo_device,
        yolo_rotate=yolo_rotate_eff,
        mps_status=mps_status,
    )


@dataclass
class ProcessResult:
    events: List[Dict[str, object]]
    raw_rows: List[Dict[str, object]]
    people_rows: List[Dict[str, object]]
    ms_detect: List[float]
    processed_frames: int
    perf_total_sec: float


def process_video(
    *,
    cap: cv2.VideoCapture,
    fps: float,
    max_frames: int,
    roi: BBox,
    args: argparse.Namespace,
    enabled: Dict[str, bool],
    primary_detector: str,
    dnn: Optional["OpenCVDNNPersonDet"],
    yolo_model,
    yolo_device: str,
    yolo_rotate_eff: str,
    writer: Optional[cv2.VideoWriter],
    out_dir: Path,
) -> ProcessResult:
    """
    Основной цикл по кадрам: читаем кадры, запускаем детекторы, обновляем FSM, сохраняем «сырцы».

    Вынесено в отдельную функцию, чтобы `main()` оставался читабельным для собеседования.
    """

    fsm = TableFSM(
        state="EMPTY",
        empty_start_ts=0.0,
        empty_gap_start_ts=0.0,
        approach_emitted=False,
        prev_in_zone_now=False,
    )
    events: List[Dict[str, object]] = []
    raw_rows: List[Dict[str, object]] = []
    people_rows: List[Dict[str, object]] = []
    ms_detect: List[float] = []

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1, y1, x2, y2 = roi.as_int_xyxy()
    roi_area = max(1e-9, float(roi.area))

    # Motion detector inside ROI (optional)
    bg = None
    if enabled.get("motion"):
        bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    motion_thr = float(args.motion_fg_frac_min)

    infer_every = max(1, int(args.infer_every))
    # Per-detector caches for skipped frames (infer_every > 1)
    yolo_cache = {
        "in_zone": False,
        "best_overlap": 0.0,
        "best_person_score": 0.0,
        "n_people": 0,
        "people": [],
        "people_raw": [],
    }
    dnn_cache = {"in_zone": False, "best_overlap": 0.0, "best_person_score": 0.0, "n_people": 0, "people": []}

    raw_people_jsonl_path = out_dir / "raw_people.jsonl"
    raw_people_f = raw_people_jsonl_path.open("w", encoding="utf-8")

    t0 = time.perf_counter()
    processed = 0
    try:
        for frame_idx in tqdm(range(int(max_frames)), desc="process"):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            processed += 1

            ts = float(frame_idx) / float(fps)
            dt = 1.0 / float(fps)

            t_det0 = time.perf_counter()

            did_infer = (frame_idx % infer_every == 0) or (frame_idx == 0)

            # ----- motion (per-frame) -----
            motion_fg_frac = 0.0
            motion_in_zone = 0
            if enabled.get("motion") and bg is not None:
                roi_crop = frame[y1:y2, x1:x2]
                fg = bg.apply(roi_crop)
                if fg is not None and fg.size > 0:
                    motion_fg_frac = float(cv2.countNonZero(fg)) / float(fg.size)
                motion_in_zone = 1 if (motion_fg_frac >= motion_thr) else 0

            # ----- opencv-dnn (infer every N frames) -----
            dnn_did_infer = 0
            if enabled.get("opencv-dnn") and dnn is not None:
                if did_infer:
                    dnn_did_infer = 1
                    dets = dnn.infer(
                        frame,
                        score_min=float(args.opencv_dnn_score_min),
                        nms_iou=float(args.opencv_dnn_nms_iou),
                        topk=int(args.opencv_dnn_topk),
                    )
                    best_overlap_dnn = 0.0
                    best_score_dnn = 0.0
                    n_people_dnn = 0
                    people_dnn: List[List[float]] = []
                    for pb, sc in dets:
                        if float(pb.area) < float(args.person_min_area):
                            continue
                        n_people_dnn += 1
                        best_score_dnn = max(best_score_dnn, float(sc))
                        ov = intersection_area(pb, roi) / roi_area
                        best_overlap_dnn = max(best_overlap_dnn, float(ov))
                        people_dnn.append([float(pb.x1), float(pb.y1), float(pb.x2), float(pb.y2), float(sc)])
                    dnn_cache = {
                        "in_zone": bool(best_overlap_dnn >= float(args.overlap_min)),
                        "best_overlap": float(best_overlap_dnn),
                        "best_person_score": float(best_score_dnn),
                        "n_people": int(n_people_dnn),
                        "people": people_dnn,
                    }
                # else: keep last dnn_cache
            else:
                dnn_cache = {
                    "in_zone": False,
                    "best_overlap": 0.0,
                    "best_person_score": 0.0,
                    "n_people": 0,
                    "people": [],
                }

            # ----- yolo (infer every N frames) -----
            yolo_did_infer = 0
            if enabled.get("yolo") and yolo_model is not None:
                if did_infer:
                    yolo_did_infer = 1
                    frame_y = _rotate_frame_for_yolo(frame, yolo_rotate_eff)
                    try:
                        res = yolo_model.predict(  # type: ignore[union-attr]
                            frame_y,
                            cls=[COCO_PERSON_CLASS_ID],
                            conf=float(args.person_conf_min),
                            imgsz=int(args.imgsz),
                            device=str(yolo_device),
                            max_det=50,
                            verbose=False,
                        )
                    except Exception:
                        res = yolo_model.predict(  # type: ignore[union-attr]
                            frame_y,
                            classes=[COCO_PERSON_CLASS_ID],
                            conf=float(args.person_conf_min),
                            imgsz=int(args.imgsz),
                            device=str(yolo_device),
                            max_det=50,
                            verbose=False,
                        )
                    best_overlap_y = 0.0
                    best_score_y = 0.0
                    n_people_y = 0
                    people_y_used: List[List[float]] = []
                    people_y_raw: List[List[float]] = []
                    for rr in res:
                        if rr.boxes is None or len(rr.boxes) == 0:
                            continue
                        xyxy = rr.boxes.xyxy.detach().cpu().numpy()
                        confs = rr.boxes.conf.detach().cpu().numpy()
                        for (x1r, y1r, x2r, y2r), conf in zip(xyxy, confs):
                            if str(yolo_rotate_eff).lower() != "none":
                                pb_yolo = _yolo_unrotate_bbox_xyxy(
                                    float(x1r),
                                    float(y1r),
                                    float(x2r),
                                    float(y2r),
                                    yolo_rotate_eff,
                                    frame_w,
                                    frame_h,
                                )
                                pb = pb_yolo.clip(frame_w, frame_h)
                            else:
                                pb = BBox(float(x1r), float(y1r), float(x2r), float(y2r)).clip(frame_w, frame_h)
                            if float(pb.area) < float(args.person_min_area):
                                continue
                            conf = float(conf)
                            row_box = [float(pb.x1), float(pb.y1), float(pb.x2), float(pb.y2), float(conf)]
                            people_y_raw.append(row_box)
                            if float(conf) >= float(args.person_conf_used):
                                people_y_used.append(row_box)
                                n_people_y += 1
                                best_score_y = max(best_score_y, float(conf))
                                ov = intersection_area(pb, roi) / roi_area
                                best_overlap_y = max(best_overlap_y, float(ov))
                    yolo_cache = {
                        "in_zone": bool(best_overlap_y >= float(args.overlap_min)),
                        "best_overlap": float(best_overlap_y),
                        "best_person_score": float(best_score_y),
                        "n_people": int(n_people_y),
                        "people": people_y_used,
                        "people_raw": people_y_raw,
                    }
                # else: keep last yolo_cache
            else:
                yolo_cache = {
                    "in_zone": False,
                    "best_overlap": 0.0,
                    "best_person_score": 0.0,
                    "n_people": 0,
                    "people": [],
                    "people_raw": [],
                }

            # ----- primary detector drives FSM -----
            if primary_detector == "yolo":
                in_zone_now = bool(yolo_cache["in_zone"])
                best_overlap = float(yolo_cache["best_overlap"])
                best_person_score = float(yolo_cache["best_person_score"])
                n_people = int(yolo_cache["n_people"])
                fg_frac = 0.0
            elif primary_detector == "opencv-dnn":
                in_zone_now = bool(dnn_cache["in_zone"])
                best_overlap = float(dnn_cache["best_overlap"])
                best_person_score = float(dnn_cache["best_person_score"])
                n_people = int(dnn_cache["n_people"])
                fg_frac = 0.0
            else:
                in_zone_now = bool(motion_in_zone)
                best_overlap = 0.0
                best_person_score = 0.0
                n_people = int(motion_in_zone)
                fg_frac = float(motion_fg_frac)

            # Init FSM on frame 0 (after we computed in_zone_now).
            # We always emit a start-state event to make the event table self-contained.
            if frame_idx == 0 and not events:
                init_state = "OCCUPIED" if bool(in_zone_now) else "EMPTY"
                fsm.state = str(init_state)
                fsm.empty_start_ts = float(ts)
                fsm.empty_gap_start_ts = float(ts)
                fsm.enter_hold = 0.0
                fsm.exit_hold = 0.0
                fsm.approach_emitted = False
                fsm.approach_hold = 0.0
                fsm.prev_in_zone_now = bool(in_zone_now)
                events.append({"ts_sec": float(ts), "frame_idx": int(frame_idx), "event_type": str(init_state)})

            # Raw people export for HTML overlay (write only on infer frames to keep file size reasonable).
            if did_infer:
                det_payload: Dict[str, object] = {}
                if enabled.get("motion"):
                    det_payload["motion"] = {
                        "did_infer": True,
                        "in_zone": bool(motion_in_zone),
                        "best_ov": 0.0,
                        "best_score": float(motion_fg_frac),
                        "fg_frac": float(motion_fg_frac),
                        "people": [],
                        "people_used": [],
                    }
                if enabled.get("opencv-dnn"):
                    det_payload["opencv-dnn"] = {
                        "did_infer": bool(dnn_did_infer),
                        "in_zone": bool(dnn_cache["in_zone"]),
                        "best_ov": float(dnn_cache["best_overlap"]),
                        "best_score": float(dnn_cache["best_person_score"]),
                        "fg_frac": 0.0,
                        "people": dnn_cache["people"],
                        "people_used": dnn_cache["people"],
                    }
                if enabled.get("yolo"):
                    det_payload["yolo"] = {
                        "did_infer": bool(yolo_did_infer),
                        "in_zone": bool(yolo_cache["in_zone"]),
                        "best_ov": float(yolo_cache["best_overlap"]),
                        "best_score": float(yolo_cache["best_person_score"]),
                        "fg_frac": 0.0,
                        "people": yolo_cache.get("people_raw", []),
                        "people_used": yolo_cache["people"],
                    }

                raw_people_f.write(
                    json.dumps(
                        {
                            "frame_idx": int(frame_idx),
                            "ts_sec": float(ts),
                            "primary_detector": str(primary_detector),
                            "detectors": det_payload,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                for det_name, det_obj in det_payload.items():
                    if not isinstance(det_obj, dict):
                        continue
                    for b in det_obj.get("people_used") or det_obj.get("people") or []:
                        try:
                            people_rows.append(
                                {
                                    "frame_idx": int(frame_idx),
                                    "ts_sec": float(ts),
                                    "detector": str(det_name),
                                    "x1": float(b[0]),
                                    "y1": float(b[1]),
                                    "x2": float(b[2]),
                                    "y2": float(b[3]),
                                    "score": float(b[4]) if len(b) > 4 else 0.0,
                                }
                            )
                        except Exception:
                            pass

            ms_detect.append((time.perf_counter() - t_det0) * 1000.0)

            # FSM + events
            fsm.step(
                ts=float(ts),
                dt=float(dt),
                in_zone_now=bool(in_zone_now),
                t_enter=float(args.t_enter),
                t_exit=float(args.t_exit),
                t_approach=float(args.t_approach),
                t_empty_min=float(args.t_empty_min),
                events=events,
                frame_idx=int(frame_idx),
            )

            # Draw ROI
            color = (0, 200, 0) if fsm.state == "EMPTY" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            if writer is not None:
                writer.write(frame)

            row: Dict[str, object] = {
                "frame_idx": int(frame_idx),
                "ts_sec": float(ts),
                "detector_mode": str(args.detector),
                "primary_detector": str(primary_detector),
                "state": str(fsm.state),
                "in_zone_now": int(bool(in_zone_now)),
                "did_infer": int(bool(did_infer)),
                "best_overlap": float(best_overlap),
                "fg_frac": float(fg_frac),
                "n_people": int(n_people),
                "best_person_score": float(best_person_score),
            }
            if str(args.detector) == "all":
                row.update(
                    {
                        "in_zone_motion": int(bool(motion_in_zone)),
                        "fg_frac_motion": float(motion_fg_frac),
                        "in_zone_yolo": int(bool(yolo_cache["in_zone"])),
                        "best_overlap_yolo": float(yolo_cache["best_overlap"]),
                        "n_people_yolo": int(yolo_cache["n_people"]),
                        "best_person_score_yolo": float(yolo_cache["best_person_score"]),
                        "did_infer_yolo": int(bool(yolo_did_infer)),
                        "in_zone_opencv_dnn": int(bool(dnn_cache["in_zone"])),
                        "best_overlap_opencv_dnn": float(dnn_cache["best_overlap"]),
                        "n_people_opencv_dnn": int(dnn_cache["n_people"]),
                        "best_person_score_opencv_dnn": float(dnn_cache["best_person_score"]),
                        "did_infer_opencv_dnn": int(bool(dnn_did_infer)),
                    }
                )
            raw_rows.append(row)
    finally:
        try:
            raw_people_f.close()
        except Exception:
            pass

    total_s = float(time.perf_counter() - t0)
    return ProcessResult(
        events=events,
        raw_rows=raw_rows,
        people_rows=people_rows,
        ms_detect=ms_detect,
        processed_frames=int(processed),
        perf_total_sec=total_s,
    )


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 1e-6:
        fps = 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = total_frames if total_frames > 0 else 1_000_000_000
    if float(args.max_seconds) > 0:
        max_frames = min(max_frames, int(float(args.max_seconds) * fps))

    # ROI selection / cache
    roi_json = Path(args.roi_json).expanduser() if str(args.roi_json).strip() else (out_dir / "roi.json")
    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        raise SystemExit("Failed to read the first frame.")
    roi = _load_or_select_roi(
        cap=cap,
        frame0=frame0,
        frame_w=frame_w,
        frame_h=frame_h,
        roi_json=roi_json,
        roi_arg=str(args.roi),
        force_roi=bool(args.force_roi),
    )

    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    det = _setup_detectors(args=args, video_path=video_path, fps=fps)
    detector_arg = det.detector_arg
    enabled = det.enabled
    primary_detector = det.primary_detector
    dnn = det.dnn
    yolo_model = det.yolo_model
    yolo_device = det.yolo_device
    yolo_rotate_eff = det.yolo_rotate
    mps_status = det.mps_status

    # Output writer
    writer: Optional[cv2.VideoWriter] = None
    used_fourcc = ""
    out_video = out_dir / "output.mp4"
    if not bool(args.no_video):
        writer, used_fourcc = _open_video_writer(
            out_video,
            fps=float(fps),
            size=(frame_w, frame_h),
            fourcc=str(args.fourcc),
        )
        print(f"Output video: {out_video} fourcc={used_fourcc} fps={fps:.2f} size={frame_w}x{frame_h}")
    else:
        print("Output video: disabled (--no-video)")

    # Main loop (per-frame): detection -> FSM -> raw exports
    try:
        res = process_video(
            cap=cap,
            fps=float(fps),
            max_frames=int(max_frames),
            roi=roi,
            args=args,
            enabled=enabled,
            primary_detector=str(primary_detector),
            dnn=dnn,
            yolo_model=yolo_model,
            yolo_device=str(yolo_device),
            yolo_rotate_eff=str(yolo_rotate_eff),
            writer=writer,
            out_dir=out_dir,
        )
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    events = res.events
    raw_rows = res.raw_rows
    people_rows = res.people_rows
    ms_detect = res.ms_detect
    perf_total_sec = res.perf_total_sec
    infer_every = max(1, int(args.infer_every))

    # Save events.csv (TZ requirement)
    df_events = pd.DataFrame(events)
    events_path = out_dir / "events.csv"
    df_events.to_csv(events_path, index=False)
    print(f"Events -> {events_path} rows={len(df_events)}")

    # Raw exports (used by HTML report / debugging).
    df_raw = pd.DataFrame(raw_rows)
    df_raw.to_csv(out_dir / "raw_frames.csv", index=False)
    df_raw.to_pickle(out_dir / "raw_frames.pkl")

    df_people = pd.DataFrame(people_rows)
    df_people.to_csv(out_dir / "raw_people.csv", index=False)
    df_people.to_pickle(out_dir / "raw_people.pkl")

    # Analytics + report
    waits = compute_waits_empty_to_next_approach(df_events)
    touch_metrics = compute_touch_metrics(df_raw, min_touch_sec=float(args.t_approach))
    if waits:
        m = summarize_waits(waits)
        print(
            "Wait metric (EMPTY -> next APPROACH): "
            f"pairs={int(m.pairs)} mean={float(m.mean_sec):.2f}s "
            f"median={float(m.median_sec):.2f}s p90={float(m.p90_sec):.2f}s"
        )
    else:
        print("Wait metric: insufficient events (need both EMPTY and APPROACH).")
    print(
        "Touch metrics: "
        f"touches={int(touch_metrics.touches)} "
        f"landings={int(touch_metrics.landings)} "
        f"mean_touch_sec={('%.2f' % float(touch_metrics.mean_touch_sec)) if touch_metrics.mean_touch_sec is not None else 'n/a'} "
        f"mean_gap_sec={('%.2f' % float(touch_metrics.mean_gap_sec)) if touch_metrics.mean_gap_sec is not None else 'n/a'} "
        f"(min_touch_sec={float(touch_metrics.min_touch_sec):.2f})"
    )

    avg_ms = float(np.mean(np.array(ms_detect, dtype=np.float64))) if ms_detect else 0.0
    eff_fps = 1000.0 / max(1e-9, avg_ms) if avg_ms > 0 else 0.0
    if avg_ms > 0:
        print(f"Perf: avg detect={avg_ms:.1f}ms (~{eff_fps:.2f} fps). total={float(perf_total_sec):.2f}s")

    report_lines = build_report_lines(
        video_path=video_path,
        fps=float(fps),
        roi_xyxy=roi.as_int_xyxy(),
        detector_mode=str(detector_arg),
        primary_detector=str(primary_detector),
        enabled=enabled,
        yolo_info={
            "weights": str(args.weights),
            "device": str(yolo_device),
            "torch_mps_built": bool(mps_status.get("built")),
            "torch_mps_available": bool(mps_status.get("available")),
            "imgsz": int(args.imgsz),
            "yolo_rotate": str(yolo_rotate_eff),
            "person_conf_min_raw": float(args.person_conf_min),
            "person_conf_used": float(args.person_conf_used),
            "person_min_area": float(args.person_min_area),
        },
        opencv_dnn_info={
            "model": str(args.opencv_dnn_model),
            "anchors": str(args.opencv_dnn_anchors),
            "score_min": float(args.opencv_dnn_score_min),
        },
        motion_info={"fg_frac_min": float(args.motion_fg_frac_min)},
        overlap_min=float(args.overlap_min),
        t_enter=float(args.t_enter),
        t_exit=float(args.t_exit),
        t_approach=float(args.t_approach),
        t_empty_min=float(args.t_empty_min),
        infer_every=int(infer_every),
        waits=waits,
        touch_metrics=touch_metrics,
        perf_avg_detect_ms=float(avg_ms),
        perf_est_fps=float(eff_fps),
        perf_total_sec=float(perf_total_sec),
    )

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
