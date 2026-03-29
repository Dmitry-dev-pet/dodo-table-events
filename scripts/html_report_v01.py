from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _load_raw(out_dir: Path) -> Optional[pd.DataFrame]:
    pkl = out_dir / "raw_frames.pkl"
    csv = out_dir / "raw_frames.csv"
    if pkl.exists():
        return pd.read_pickle(pkl)
    if csv.exists():
        return pd.read_csv(csv)
    return None


def _load_events(out_dir: Path) -> Optional[pd.DataFrame]:
    p = out_dir / "events.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _load_roi(out_dir: Path) -> Optional[Tuple[int, int, int, int]]:
    p = out_dir / "roi.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        xyxy = data.get("roi_xyxy")
        if isinstance(xyxy, list) and len(xyxy) == 4:
            return tuple(int(v) for v in xyxy)
        return (int(data["x1"]), int(data["y1"]), int(data["x2"]), int(data["y2"]))
    except Exception:
        return None


def _roi_from_report_txt(report_txt: str) -> Optional[Tuple[int, int, int, int]]:
    if not report_txt:
        return None
    for line in report_txt.splitlines():
        line = line.strip()
        if not line.startswith("roi_xyxy:"):
            continue
        value = line.split(":", 1)[1].strip()
        if value.startswith("(") and value.endswith(")"):
            try:
                parts = [int(x.strip()) for x in value[1:-1].split(",")]
                if len(parts) == 4:
                    return tuple(parts)
            except Exception:
                return None
    return None


def _fps_from_report_txt(report_txt: str) -> float:
    if not report_txt:
        return 0.0
    for line in report_txt.splitlines():
        line = line.strip()
        if not line.startswith("fps:"):
            continue
        try:
            return float(line.split(":", 1)[1].strip())
        except Exception:
            return 0.0
    return 0.0


def _fsm_params_from_report_txt(report_txt: str) -> Dict[str, float]:
    out = {
        "t_enter_sec": 0.0,
        "t_exit_sec": 0.0,
        "t_approach_sec": 0.0,
        "t_empty_min_sec": 0.0,
    }
    if not report_txt:
        return out
    for line in report_txt.splitlines():
        stripped = line.strip()
        for key in list(out.keys()):
            prefix = f"{key}:"
            if stripped.startswith(prefix):
                try:
                    out[key] = float(stripped.split(":", 1)[1].strip())
                except Exception:
                    pass
    return out


def _overlap_min_from_report_txt(report_txt: str) -> float:
    if not report_txt:
        return 0.0
    for line in report_txt.splitlines():
        stripped = line.strip()
        if not stripped.startswith("overlap_min:"):
            continue
        try:
            return float(stripped.split(":", 1)[1].strip())
        except Exception:
            return 0.0
    return 0.0


def _analytics_from_report_txt(report_txt: str) -> Dict[str, object]:
    info: Dict[str, object] = {
        "wait_pairs": None,
        "wait_mean_sec": None,
        "wait_median_sec": None,
        "wait_p90_sec": None,
        "wait_available": False,
        "touch_min_sec": None,
        "touches": None,
        "landings": None,
        "mean_touch_sec": None,
        "mean_gap_sec": None,
    }
    if not report_txt:
        return info

    section = ""
    for raw in report_txt.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped == "wait_metric_empty_to_next_approach:":
            section = "wait"
            info["wait_available"] = True
            continue
        if stripped.startswith("wait_metric_empty_to_next_approach: insufficient"):
            section = ""
            info["wait_available"] = False
            continue
        if stripped == "touch_metrics:":
            section = "touch"
            continue

        if section == "wait" and raw.startswith("  ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            value = value.strip()
            if key == "pairs":
                info["wait_pairs"] = int(value)
            elif key == "mean_sec":
                info["wait_mean_sec"] = float(value)
            elif key == "median_sec":
                info["wait_median_sec"] = float(value)
            elif key == "p90_sec":
                info["wait_p90_sec"] = float(value)
            continue

        if section == "touch" and raw.startswith("  ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            value = value.strip()
            if key == "min_touch_sec":
                info["touch_min_sec"] = float(value)
            elif key == "touches":
                info["touches"] = int(value)
            elif key == "landings":
                info["landings"] = int(value)
            elif key == "mean_touch_sec":
                info["mean_touch_sec"] = None if value == "n/a" else float(value)
            elif key == "mean_gap_sec":
                info["mean_gap_sec"] = None if value == "n/a" else float(value)
    return info


def _read_text(path: Path, *, max_chars: int = 20000) -> str:
    if not path.exists():
        return ""
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
        if len(txt) > max_chars:
            return txt[:max_chars] + "\n...\n"
        return txt
    except Exception:
        return ""


def _video_from_report_txt(report_txt: str) -> str:
    if not report_txt:
        return ""
    for line in report_txt.splitlines():
        stripped = line.strip()
        if not stripped.startswith("video:"):
            continue
        return stripped.split(":", 1)[1].strip()
    return ""


def _rel_media_src(base_dir: Path, media_path: str) -> str:
    value = str(media_path or "").strip()
    if not value:
        return ""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists():
        return ""
    try:
        return os.path.relpath(candidate, start=base_dir)
    except Exception:
        return str(candidate)


def _prepare_payload(
    *,
    out_dir: Path,
    overlap_min_hint: float,
    max_points: int,
) -> Dict[str, object]:
    roi = _load_roi(out_dir)
    df_ev = _load_events(out_dir)
    df_raw = _load_raw(out_dir)
    report_txt = _read_text(out_dir / "report.txt")
    fps_hint = _fps_from_report_txt(report_txt)
    fsm_params = _fsm_params_from_report_txt(report_txt)
    analytics = _analytics_from_report_txt(report_txt)
    if overlap_min_hint <= 0:
        overlap_min_hint = _overlap_min_from_report_txt(report_txt)
    if roi is None:
        roi = _roi_from_report_txt(report_txt)

    events_payload: List[Dict[str, object]] = []
    if df_ev is not None and not df_ev.empty:
        df_ev = df_ev.sort_values(["ts_sec", "frame_idx"]).reset_index(drop=True)
        for _idx, r in df_ev.iterrows():
            try:
                events_payload.append(
                    {
                        "ts": float(r["ts_sec"]),
                        "frame": int(r["frame_idx"]),
                        "type": str(r["event_type"]),
                    }
                )
            except Exception:
                continue

    raw_payload: Dict[str, object] = {"available": False}
    if df_raw is not None and len(df_raw) > 0 and "ts_sec" in df_raw.columns:
        d = df_raw.copy()
        d = d.sort_values("ts_sec").reset_index(drop=True)
        primary_detector_name = ""
        if "primary_detector" in d.columns and not d["primary_detector"].empty:
            try:
                primary_detector_name = str(d["primary_detector"].iloc[0]).strip()
            except Exception:
                primary_detector_name = ""
        keep_cols = [
            c
            for c in [
                "frame_idx",
                "ts_sec",
                "primary_detector",
                "best_overlap",
                "fg_frac",
                "n_people",
                "best_person_score",
                "in_zone_now",
                "state",
                # Дополнительно (если прогон был с --detector all):
                "in_zone_motion",
                "fg_frac_motion",
                "in_zone_yolo",
                "best_overlap_yolo",
                "n_people_yolo",
                "best_person_score_yolo",
                "in_zone_opencv_dnn",
                "best_overlap_opencv_dnn",
                "n_people_opencv_dnn",
                "best_person_score_opencv_dnn",
            ]
            if c in d.columns
        ]
        d = d[keep_cols]
        if len(d) > int(max_points):
            step = int(np.ceil(len(d) / float(max_points)))
            d = d.iloc[::step].reset_index(drop=True)

        ts = d["ts_sec"].astype(float).to_numpy(copy=False)

        def _col_f(name: str, default: float = 0.0) -> List[float]:
            if name not in d.columns:
                return [float(default) for _ in range(len(d))]
            arr = d[name].to_numpy(dtype=np.float32, copy=False)
            arr = np.nan_to_num(arr, nan=float(default), posinf=float(default), neginf=float(default))
            return [float(x) for x in arr.tolist()]

        def _col_i(name: str) -> List[int]:
            if name not in d.columns:
                return [0 for _ in range(len(d))]
            arr = d[name].to_numpy(dtype=np.float32, copy=False)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return [int(x) for x in arr.tolist()]

        state01: List[int]
        if "state" in d.columns:
            s = d["state"].astype(str).to_list()
            state01 = [1 if x == "OCCUPIED" else 0 for x in s]
        else:
            state01 = [0 for _ in range(len(d))]

        raw_payload = {
            "available": True,
            "n": int(len(d)),
            "frame_idx": _col_i("frame_idx"),
            "ts": [float(x) for x in ts.tolist()],
            "best_overlap": _col_f("best_overlap", 0.0),
            "fg_frac": _col_f("fg_frac", 0.0),
            "n_people": _col_i("n_people"),
            "best_person_score": _col_f("best_person_score", 0.0),
            "in_zone_now": _col_i("in_zone_now"),
            "state01": state01,
            "primary_detector": primary_detector_name,
            "series_note": "Данные прорежены (downsampling) для отображения в HTML.",
        }

        # Если прогон был с --detector all, сохраняем отдельные серии по детекторам,
        # чтобы в HTML можно было сравнивать сигналы, не перезапуская инференс.
        by_detector: Dict[str, Dict[str, object]] = {
            "primary": {
                "best_overlap": raw_payload["best_overlap"],
                "n_people": raw_payload["n_people"],
                "best_person_score": raw_payload["best_person_score"],
                "in_zone_now": raw_payload["in_zone_now"],
            }
        }

        if "best_overlap_yolo" in d.columns or "in_zone_yolo" in d.columns:
            by_detector["yolo"] = {
                "best_overlap": _col_f("best_overlap_yolo", 0.0),
                "n_people": _col_i("n_people_yolo"),
                "best_person_score": _col_f("best_person_score_yolo", 0.0),
                "in_zone_now": _col_i("in_zone_yolo"),
            }

        if "best_overlap_opencv_dnn" in d.columns or "in_zone_opencv_dnn" in d.columns:
            by_detector["opencv-dnn"] = {
                "best_overlap": _col_f("best_overlap_opencv_dnn", 0.0),
                "n_people": _col_i("n_people_opencv_dnn"),
                "best_person_score": _col_f("best_person_score_opencv_dnn", 0.0),
                "in_zone_now": _col_i("in_zone_opencv_dnn"),
            }

        if "fg_frac_motion" in d.columns or "in_zone_motion" in d.columns:
            fg = _col_f("fg_frac_motion", 0.0)
            by_detector["motion"] = {
                # Для motion нет bbox/overlap; в качестве "сигнала" используем fg_frac внутри ROI.
                "best_overlap": fg,
                "n_people": _col_i("in_zone_motion"),
                "best_person_score": fg,
                "in_zone_now": _col_i("in_zone_motion"),
            }

        if len(by_detector) > 1:
            raw_payload["by_detector"] = by_detector
    video_path = out_dir / "output.mp4"
    source_video_path = _video_from_report_txt(report_txt)
    output_video_src = "output.mp4" if video_path.exists() else ""
    original_video_src_local = _rel_media_src(out_dir, source_video_path)
    original_video_src = original_video_src_local

    # Optional: people bboxes per frame (for video overlay). We align it to downsampled raw.ts.
    people_path = out_dir / "raw_people.jsonl"
    if raw_payload.get("available") and people_path.exists():
        try:
            people_ts: List[float] = []
            people_frames: List[int] = []
            people_by_detector: Dict[str, List[list]] = {}
            with people_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    t = float(obj.get("ts_sec", 0.0))
                    frame_idx = int(obj.get("frame_idx", len(people_ts)))
                    idx = len(people_ts)
                    people_ts.append(t)
                    people_frames.append(frame_idx)

                    present: set = set()
                    if isinstance(obj.get("detectors"), dict):
                        det_map = obj.get("detectors") or {}
                        for det_name, det_obj in det_map.items():
                            det_name = str(det_name)
                            if det_name not in people_by_detector:
                                people_by_detector[det_name] = [[] for _ in range(idx)]
                            boxes = []
                            if isinstance(det_obj, dict):
                                boxes = det_obj.get("people_used", []) or det_obj.get("people", []) or []
                            people_by_detector[det_name].append(boxes)
                            present.add(det_name)
                    else:
                        det_name = str(obj.get("detector", "people"))
                        if det_name not in people_by_detector:
                            people_by_detector[det_name] = [[] for _ in range(idx)]
                        people_by_detector[det_name].append(obj.get("people", []) or [])
                        present.add(det_name)

                    # Pad missing detectors with empty boxes for this timestamp.
                    for det_name, seq in list(people_by_detector.items()):
                        if det_name not in present and len(seq) < len(people_ts):
                            seq.append([])
            if people_ts:
                if people_by_detector:
                    primary_name = str(raw_payload.get("primary_detector") or "").strip()
                    if primary_name and primary_name in people_by_detector:
                        people_by_detector["primary"] = people_by_detector[primary_name]
                    raw_payload["people_frames"] = people_frames
                    raw_payload["people_ts"] = people_ts
                    raw_payload["people_by_detector"] = people_by_detector
                    prefer = "primary" if "primary" in people_by_detector else next(iter(people_by_detector.keys()))
                    raw_payload["people"] = people_by_detector.get(prefer, [])
                    raw_payload["people_note"] = "Overlay uses people_used and matches nearest infer frame_idx from raw_people.jsonl"
        except Exception:
            pass

    payload: Dict[str, object] = {
        "meta": {
            "out_dir": str(out_dir),
            "roi_xyxy": list(map(int, roi)) if roi is not None else [],
            "overlap_min": float(overlap_min_hint),
            "fps": float(fps_hint),
            "fsm": fsm_params,
            "analytics": analytics,
            "source_video": source_video_path,
            "report_txt": report_txt,
        },
        "events": events_payload,
        "raw": raw_payload,
        "video_src": output_video_src or original_video_src,
        "video_src_rendered": output_video_src,
        "video_src_original": original_video_src,
        "video_src_original_local": original_video_src_local,
        "video_src_kind": "rendered" if output_video_src else ("original" if original_video_src else ""),
    }
    return payload


def _build_html(payload: Dict[str, object]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    meta = payload.get("meta") or {}
    analytics = meta.get("analytics") or {}

    def _fmt_metric_sec(value: object) -> str:
        try:
            if value is None:
                return "n/a"
            return f"{float(value):.2f}s"
        except Exception:
            return "n/a"

    if analytics.get("wait_available") and analytics.get("wait_mean_sec") is not None:
        wait_header_markup = (
            f'<div class="videoMetricCard primary">'
            f'<div class="videoMetricKicker">ТЗ</div>'
            f'<div class="videoMetricTitle">Ожидание</div>'
            f'<div class="videoMetricValue">mean={_fmt_metric_sec(analytics.get("wait_mean_sec"))}</div>'
            f'<div class="videoMetricMeta">pairs={int(analytics.get("wait_pairs") or 0)} · '
            f'med={_fmt_metric_sec(analytics.get("wait_median_sec"))} · '
            f'p90={_fmt_metric_sec(analytics.get("wait_p90_sec"))}</div>'
            f'</div>'
        )
    else:
        wait_header_markup = (
            '<div class="videoMetricCard primary empty">'
            '<div class="videoMetricKicker">ТЗ</div>'
            '<div class="videoMetricTitle">Ожидание</div>'
            '<div class="videoMetricValue">недостаточно событий</div>'
            '<div class="videoMetricMeta">Нет пары EMPTY→APPROACH</div>'
            '</div>'
        )

    touch_cards: List[str] = []
    touch_min = analytics.get("touch_min_sec")
    touches = analytics.get("touches")
    landings = analytics.get("landings")
    mean_touch_sec = analytics.get("mean_touch_sec")
    mean_gap_sec = analytics.get("mean_gap_sec")
    if touches is not None:
        label = "Касания"
        if touch_min is not None:
            label += f" >= {float(touch_min):.1f}s"
        touch_cards.append(f'<div class="videoMiniMetric"><div class="videoMiniLabel">{label}</div><div class="videoMiniValue">{int(touches)}</div></div>')
    if mean_touch_sec is not None:
        touch_cards.append(f'<div class="videoMiniMetric"><div class="videoMiniLabel">Ср. длит.</div><div class="videoMiniValue">{_fmt_metric_sec(mean_touch_sec)}</div></div>')
    if mean_gap_sec is not None:
        touch_cards.append(f'<div class="videoMiniMetric"><div class="videoMiniLabel">Ср. интервал</div><div class="videoMiniValue">{_fmt_metric_sec(mean_gap_sec)}</div></div>')
    if landings is not None:
        touch_cards.append(f'<div class="videoMiniMetric"><div class="videoMiniLabel">Посадки</div><div class="videoMiniValue">{int(landings)}</div></div>')
    touch_header_markup = (
        '<div class="videoMetricCard secondary">'
        '<div class="videoMetricKicker">Перед посадкой</div>'
        '<div class="videoMetricTitle">Касания</div>'
        f'<div class="videoMiniGrid">{"".join(touch_cards)}</div>'
        '</div>'
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Отчёт (v0.1): занятость стола</title>
  <style>
    :root {{
      --bg: #0b0d12;
      --panel: rgba(255,255,255,0.06);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --yellow: #f2c94c;
      --blue: #4dabf7;
      --green: #20c997;
      --red: #ff4d4f;
    }}
    body {{ margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; background: var(--bg); color: var(--text); }}
    .wrap {{ max-width: 1100px; margin: 24px auto; padding: 0 16px 48px; }}
    h1 {{ font-size: 20px; margin: 0 0 10px; }}
    .meta {{ color: var(--muted); font-size: 13px; line-height: 1.5; }}
    code {{ background: rgba(255,255,255,0.08); padding: 2px 6px; border-radius: 6px; }}
    .row {{ display: grid; grid-template-columns: 1fr; gap: 14px; margin-top: 14px; }}
    .panel {{ background: var(--panel); border-radius: 12px; padding: 14px; }}
    .panel h2 {{ margin: 0 0 8px; font-size: 14px; color: var(--muted); font-weight: 600; }}
    .flex {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .btn {{ cursor:pointer; user-select:none; background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.12); color: var(--text); padding: 6px 10px; border-radius: 10px; font-size: 13px; }}
    .btn.sm {{ padding: 4px 8px; border-radius: 8px; font-size: 12px; }}
    .btn.active {{ border-color: rgba(77,171,247,0.75); background: rgba(77,171,247,0.18); }}
    .btn.danger.active {{ border-color: rgba(255,77,79,0.75); background: rgba(255,77,79,0.16); }}
    .btn.green.active {{ border-color: rgba(32,201,151,0.75); background: rgba(32,201,151,0.15); }}
    .small {{ font-size: 12px; color: var(--muted); }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    @media (max-width: 900px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
    .stageList {{ display: grid; gap: 12px; }}
    .stageCard {{ display: grid; grid-template-columns: 1fr; gap: 12px; align-items: stretch; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 12px; }}
    .stageNum {{ font-size: 12px; color: var(--yellow); margin-bottom: 6px; }}
    .stageTitle {{ font-size: 14px; font-weight: 600; margin-bottom: 6px; }}
    .stageBody {{ font-size: 12px; color: var(--muted); line-height: 1.45; }}
    .stageChartLabel {{ font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
    .legendRow {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:8px; font-size:12px; color: var(--muted); }}
    .legendItem {{ display:flex; align-items:center; gap:6px; }}
    .legendSwatch {{ width:18px; height:3px; border-radius:999px; display:inline-block; }}
    .legendSwatch.step {{ height:0; border-top:3px solid rgba(255,255,255,0.92); width:18px; }}
    .legendSwatch.dashed {{ height:0; border-top:3px dashed rgba(242,201,76,0.75); width:18px; }}
    .btn:disabled {{ opacity: 0.45; cursor: default; }}
    .chartCanvas {{ width: 100%; height: 160px; border-radius: 10px; background: rgba(0,0,0,0.25); }}
    .videoCard {{ display:grid; gap:10px; }}
    .videoHeaderBar {{ display:flex; gap:8px; align-items:center; flex-wrap:nowrap; overflow-x:auto; padding-bottom:2px; }}
    .videoMetricsBar {{ display:grid; grid-template-columns: minmax(280px, 0.95fr) minmax(0, 1.05fr); gap:8px; }}
    .statusBadges {{ display:flex; gap:8px; flex-wrap:nowrap; align-items:center; }}
    .statusBadge {{
      display:inline-flex;
      align-items:center;
      gap:6px;
      min-height:30px;
      padding:6px 10px;
      border-radius:999px;
      font-size:12px;
      border:1px solid rgba(255,255,255,0.12);
      background:rgba(255,255,255,0.06);
      color:var(--text);
      white-space:nowrap;
      line-height:1;
      flex:0 0 auto;
    }}
    .statusBadge.subtle {{ color: var(--muted); }}
    .statusBadge.good {{ border-color: rgba(32,201,151,0.45); background: rgba(32,201,151,0.16); }}
    .statusBadge.bad {{ border-color: rgba(255,77,79,0.45); background: rgba(255,77,79,0.16); }}
    .statusLineCompact {{
      display:flex;
      gap:8px;
      flex-wrap:nowrap;
      align-items:center;
      justify-content:flex-end;
      font-size:11px;
      color:var(--muted);
      margin-left:auto;
      flex:0 0 auto;
    }}
    .metaGroup {{
      display:inline-flex;
      align-items:center;
      gap:6px;
      min-height:30px;
      padding:6px 10px;
      border-radius:999px;
      border:1px solid rgba(255,255,255,0.12);
      background:rgba(255,255,255,0.06);
      white-space:nowrap;
      flex:0 0 auto;
    }}
    .statusChip {{
      display:inline-flex;
      align-items:center;
      min-height:22px;
      padding:0;
      border-radius:0;
      border:none;
      background:transparent;
      color:var(--text);
      white-space:nowrap;
      line-height:1;
    }}
    .statusChip.roi {{ color: var(--text); cursor:pointer; font-weight:600; }}
    .metaGroup .btn.sm {{ padding:4px 8px; white-space:nowrap; }}
    .metaGroup.fsm .statusChip {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .pillIcon {{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      width:16px;
      height:16px;
      border-radius:999px;
      background:rgba(255,255,255,0.12);
      color:rgba(255,255,255,0.78);
      font-size:10px;
      font-weight:700;
      letter-spacing:0.02em;
      flex:0 0 16px;
    }}
    .pillText {{ white-space:nowrap; }}
    .videoStage {{ background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03)); border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:12px; }}
    .videoMetricCard {{
      border:1px solid rgba(255,255,255,0.10);
      border-radius:12px;
      padding:9px 10px;
      background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.04));
    }}
    .videoMetricCard.primary {{
      border-color: rgba(204,90,47,0.35);
      background: linear-gradient(180deg, rgba(255,247,242,0.12), rgba(204,90,47,0.08));
    }}
    .videoMetricCard.secondary {{
      background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.04));
    }}
    .videoMetricCard.empty .videoMetricValue,
    .videoMetricCard.empty .videoMetricMeta {{ color: var(--muted); }}
    .videoMetricKicker {{
      color: rgba(255,255,255,0.66);
      font-size: 9px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 3px;
    }}
    .videoMetricTitle {{ font-size: 12px; font-weight: 700; line-height: 1.15; }}
    .videoMetricValue {{ margin-top: 5px; font-size: 17px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }}
    .videoMetricMeta {{ margin-top: 3px; color: var(--muted); font-size: 11px; line-height: 1.25; }}
    .videoMiniGrid {{ margin-top: 7px; display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:6px; }}
    .videoMiniMetric {{ border:1px solid rgba(255,255,255,0.10); border-radius:10px; padding:7px 8px; background: rgba(255,255,255,0.04); min-width:0; }}
    .videoMiniLabel {{ color: var(--muted); font-size: 10px; line-height: 1.15; }}
    .videoMiniValue {{ margin-top: 3px; font-size: 15px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }}
    .videoWrap {{ position: relative; width: 100%; border-radius: 14px; overflow: hidden; background: rgba(0,0,0,0.34); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06); }}
    video {{ width: 100%; display:block; }}
    #overlay {{ position:absolute; left:0; top:0; border-radius:0; background: transparent; pointer-events:none; }}
    .slider {{ width: 100%; }}
    .controlBar {{ display:grid; grid-template-columns: 1fr; gap:8px; align-items:center; }}
    .controlMain {{ display:grid; gap:6px; min-width:0; }}
    .timeMeta {{ display:flex; justify-content:space-between; gap:10px; align-items:center; font-size:11px; color:var(--muted); }}
    .controlSide {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:flex-start; }}
    .toggleGroup, .detectorGroup, .videoSourceGroup {{ display:flex; gap:6px; align-items:center; flex-wrap:wrap; }}
    .btn.linkish {{ text-decoration:none; display:inline-flex; align-items:center; }}
    .roiRow {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
    .roiField {{ width:auto; min-width:0; max-width:240px; text-align:left; cursor:pointer; padding:4px 8px; font-size:12px; }}
    .decisionLine {{ font-size:12px; color:var(--muted); line-height:1.45; }}
    @media (max-width: 900px) {{
      .videoHeaderBar {{ flex-wrap:nowrap; }}
      .videoMetricsBar {{ grid-template-columns: 1fr; }}
      .videoMiniGrid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
      .statusBadges, .statusLineCompact {{ flex-wrap:nowrap; }}
      .statusLineCompact {{ margin-left:0; }}
      .controlSide {{ justify-content:flex-start; }}
      .roiRow {{ grid-template-columns: 1fr; }}
    }}
    .kv {{ display: grid; grid-template-columns: 180px 1fr; gap: 8px; font-size: 13px; }}
    .kv div:nth-child(odd) {{ color: var(--muted); }}
    details summary {{ cursor: pointer; color: var(--muted); }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: rgba(0,0,0,0.25); padding: 12px; border-radius: 10px; }}
    .topBar {{ display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; margin-bottom:10px; }}
    .backLink {{ text-decoration:none; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topBar">
      <h1>Отчёт (v0.1): события по одному столу</h1>
      <a class="btn sm linkish backLink" href="../report.html">Back to report</a>
    </div>
    <div class="meta">
      <div>Оффлайн HTML (без CDN). Источник данных: <code>events.csv</code>, <code>raw_frames.*</code>, <code>raw_people.jsonl</code>.</div>
    </div>

    <div class="row">
      <div class="panel">
        <h2>Видео</h2>
        <div class="videoCard">
          <div class="videoHeaderBar">
            <div class="statusBadges">
              <span id="badgeDetector" class="statusBadge"><span class="pillIcon">D</span><span class="pillText">-</span></span>
              <span id="badgeState" class="statusBadge"><span class="pillIcon">S</span><span class="pillText">-</span></span>
              <span id="badgeFrameTime" class="statusBadge subtle"><span class="pillIcon">T</span><span class="pillText">0.00s</span></span>
            </div>
            <div class="statusLineCompact">
              <div class="metaGroup roi">
                <span class="pillIcon">R</span>
                <span id="headerRoi" class="statusChip roi">-</span>
                <button id="btnCopyRoi" class="btn sm" type="button">Copy</button>
                <button id="btnResetRoi" class="btn sm" type="button">Reset</button>
              </div>
              <div class="metaGroup">
                <span class="pillIcon">V</span>
                <span id="headerVideoSource" class="statusChip">-</span>
              </div>
            </div>
          </div>

          <div class="videoMetricsBar">
            <div id="waitMetricCardWrap">{wait_header_markup}</div>
            <div id="touchMetricCardWrap">{touch_header_markup}</div>
          </div>

          <div class="videoStage">
            <div class="videoWrap">
              <video id="vid" controls preload="metadata"></video>
              <canvas id="overlay"></canvas>
            </div>
          </div>

          <div class="controlBar">
            <div class="controlMain">
              <input id="timeSlider" class="slider" type="range" min="0" max="1" step="0.001" value="0" />
              <div class="timeMeta">
                <span><span id="tLabel">0.00</span> s</span>
                <span>Hover ROI to move or resize</span>
              </div>
            </div>
            <div class="controlSide">
              <div class="videoSourceGroup">
                <span class="small">Видео:</span>
                <span id="srcRendered" class="btn sm">rendered</span>
                <span id="srcOriginal" class="btn sm">original</span>
              </div>
              <div class="toggleGroup">
                <span class="small">Оверлей:</span>
                <span id="btnRoi" class="btn sm green active">ROI</span>
                <span id="btnPeople" class="btn sm active">Люди</span>
              </div>
              <div class="detectorGroup">
                <span class="small">Детектор:</span>
                <span id="detAll" class="btn sm">all</span>
                <span id="detYolo" class="btn sm">yolo</span>
                <span id="detDnn" class="btn sm">opencv-dnn</span>
                <span id="detMotion" class="btn sm">motion</span>
              </div>
            </div>
          </div>

        </div>
      </div>

      <div class="panel">
        <h2>Как Читать Модель</h2>
        <div class="stageList">
          <div class="stageCard">
            <div>
              <div class="stageNum">1-4. Цепочка Решения</div>
              <div class="stageTitle">От наблюдения до финального состояния</div>
              <div class="stageBody">Верхняя дорожка показывает решение системы: быстрый сигнал <code>in_zone_now</code> и устойчивое состояние <code>state</code>. Средняя дорожка показывает причины срабатывания: <code>best_overlap</code> и <code>best_person_score</code> вместе с порогом <code>overlap_min</code>. Нижняя дорожка показывает <code>n_people</code> в полном кадре, чтобы было видно, нашёл ли детектор кого-то вообще.</div>
            </div>
            <div>
              <div class="stageChartLabel">Один таймлайн: решение системы, причины срабатывания и число найденных людей</div>
              <canvas id="cTimeline" class="chartCanvas"></canvas>
              <div class="legendRow">
                <span class="legendItem"><span class="legendSwatch" style="background: rgba(255,77,79,0.95);"></span><span>state</span></span>
                <span class="legendItem"><span class="legendSwatch" style="background: rgba(32,201,151,0.95);"></span><span>in_zone_now</span></span>
                <span class="legendItem"><span class="legendSwatch" style="background: rgba(242,201,76,0.95);"></span><span>overlap</span></span>
                <span class="legendItem"><span class="legendSwatch" style="background: rgba(77,171,247,0.95);"></span><span>score</span></span>
                <span class="legendItem"><span class="legendSwatch dashed"></span><span>overlap_min</span></span>
                <span class="legendItem"><span class="legendSwatch step"></span><span>n_people</span></span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>События</h2>
        <div id="eventsTable"></div>
      </div>

      <div class="panel">
        <h2>Meta</h2>
        <div id="metaKv" class="kv"></div>
        <details style="margin-top:10px;">
          <summary>report.txt</summary>
          <pre id="reportTxt"></pre>
        </details>
      </div>
    </div>
  </div>

  <script>
    const PAYLOAD = {payload_json};

    const vid = document.getElementById('vid');
    const videoWrap = vid.parentElement;
    const overlay = document.getElementById('overlay');
    const ctxO = overlay.getContext('2d');
    const timeSlider = document.getElementById('timeSlider');
    const tLabel = document.getElementById('tLabel');
    const btnRoi = document.getElementById('btnRoi');
    const btnPeople = document.getElementById('btnPeople');
    const btnEditRoi = document.getElementById('btnEditRoi');
    const btnCopyRoi = document.getElementById('btnCopyRoi');
    const btnResetRoi = document.getElementById('btnResetRoi');
    const headerRoi = document.getElementById('headerRoi');
    const headerVideoSource = document.getElementById('headerVideoSource');
    const headerRoiStatus = document.getElementById('headerRoiStatus');
    const headerFsm = document.getElementById('headerFsm');
    const badgeDetector = document.getElementById('badgeDetector');
    const badgeState = document.getElementById('badgeState');
    const badgeInZone = document.getElementById('badgeInZone');
    const badgeOverlap = document.getElementById('badgeOverlap');
    const badgeFrameTime = document.getElementById('badgeFrameTime');
    const waitMetricCardWrap = document.getElementById('waitMetricCardWrap');
    const touchMetricCardWrap = document.getElementById('touchMetricCardWrap');

    const detBtns = {{
      all: document.getElementById('detAll'),
      yolo: document.getElementById('detYolo'),
      'opencv-dnn': document.getElementById('detDnn'),
      motion: document.getElementById('detMotion'),
    }};
    const sourceBtns = {{
      rendered: document.getElementById('srcRendered'),
      original: document.getElementById('srcOriginal'),
    }};

    let showRoi = true;
    let showPeople = true;
    let peopleDet = '';
    let graphDet = '';
    let currentVideoSourceKind = String(PAYLOAD.video_src_kind || '');
    let renderTime = 0;
    let roiHoverMode = '';
    let overlayScaleX = 1;
    let overlayScaleY = 1;
    let overlayCssW = 0;
    let overlayCssH = 0;
    const analyticsCache = Object.create(null);
    const baseRoi = Array.isArray(PAYLOAD.meta && PAYLOAD.meta.roi_xyxy) ? (PAYLOAD.meta.roi_xyxy.slice(0, 4).map(v => +v || 0)) : [];
    let currentRoi = baseRoi.slice();
    let editRoi = false;
    let roiDrag = null;
    const meta = PAYLOAD.meta || {{}};
    const fsm = meta.fsm || {{}};
    const overlapMin = +((meta.overlap_min != null ? meta.overlap_min : 0) || 0);
    const roiStorageId = String(meta.source_video || '').trim() || String((PAYLOAD.meta && PAYLOAD.meta.roi_xyxy) ? PAYLOAD.meta.roi_xyxy.join(',') : '') || 'detail';
    const roiStorageKey = `dodo-table-events::roi::${{roiStorageId}}`;

    function loadSavedRoi() {{
      try {{
        if (!window.localStorage) return null;
        const raw = window.localStorage.getItem(roiStorageKey);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed) || parsed.length !== 4) return null;
        const vals = parsed.map(v => Math.round(+v || 0));
        if (vals.some(v => !Number.isFinite(v))) return null;
        return vals;
      }} catch (_err) {{
        return null;
      }}
    }}

    function saveCurrentRoi() {{
      try {{
        if (!window.localStorage) return;
        if (!Array.isArray(currentRoi) || currentRoi.length !== 4 || !roiDirty()) {{
          window.localStorage.removeItem(roiStorageKey);
          return;
        }}
        window.localStorage.setItem(roiStorageKey, JSON.stringify(currentRoi.map(v => Math.round(+v || 0))));
      }} catch (_err) {{
        // Ignore storage failures in offline preview environments.
      }}
    }}

    function clearSavedRoi() {{
      try {{
        if (window.localStorage) window.localStorage.removeItem(roiStorageKey);
      }} catch (_err) {{
        // Ignore storage failures in offline preview environments.
      }}
    }}

    function roiHitModeAt(mx, my) {{
      if (!hasEditablePreview() || graphDet === 'motion') return '';
      const roi = overlayRoi();
      if (!roi || roi.length !== 4) return '';
      const r = roiRectCss(roi);
      const hit = 12;
      const corners = [
        ['tl', r.x, r.y],
        ['tr', r.x + r.w, r.y],
        ['bl', r.x, r.y + r.h],
        ['br', r.x + r.w, r.y + r.h],
      ];
      for (const [name, px, py] of corners) {{
        if (Math.abs(mx - px) <= hit && Math.abs(my - py) <= hit) return String(name);
      }}
      if (mx >= r.x && mx <= r.x + r.w && my >= r.y && my <= r.y + r.h) return 'move';
      return '';
    }}

    function cursorForRoiMode(mode) {{
      if (mode === 'move') return 'move';
      if (mode === 'tl' || mode === 'br') return 'nwse-resize';
      if (mode === 'tr' || mode === 'bl') return 'nesw-resize';
      return 'default';
    }}

    function updateOverlayInteractivity() {{
      const interactive = !!roiDrag || !!roiHoverMode;
      overlay.style.pointerEvents = interactive ? 'auto' : 'none';
      overlay.style.cursor = cursorForRoiMode(roiDrag ? roiDrag.mode : roiHoverMode);
    }}

    function availableDetectors() {{
      const raw = PAYLOAD.raw || {{}};
      const by = raw.by_detector || {{}};
      return ['yolo', 'opencv-dnn', 'motion'].filter(name => !!by[name]);
    }}

    function initialDetector() {{
      const available = availableDetectors();
      if (available.includes('yolo')) return 'yolo';
      return preferredDetector();
    }}

    function preferredDetector() {{
      const raw = PAYLOAD.raw || {{}};
      const preferred = String(raw.primary_detector || '');
      const available = availableDetectors();
      if (available.includes(preferred)) return preferred;
      return available[0] || 'motion';
    }}

    function selectedDetectors() {{
      if (graphDet === 'all') return availableDetectors();
      return [graphDet || preferredDetector()].filter(Boolean);
    }}

    function editableDetectors() {{
      return ['yolo', 'opencv-dnn'].filter(name => availableDetectors().includes(name));
    }}

    function roiKey(roi) {{
      return Array.isArray(roi) && roi.length === 4 ? roi.map(v => Math.round(+v || 0)).join(',') : '';
    }}

    function availableVideoSources() {{
      const out = [];
      if (PAYLOAD.video_src_rendered) out.push('rendered');
      if (PAYLOAD.video_src_original) out.push('original');
      return out;
    }}

    function currentVideoSourceSrc() {{
      if (currentVideoSourceKind === 'original' && PAYLOAD.video_src_original) return PAYLOAD.video_src_original;
      if (currentVideoSourceKind === 'rendered' && PAYLOAD.video_src_rendered) return PAYLOAD.video_src_rendered;
      return PAYLOAD.video_src || PAYLOAD.video_src_rendered || PAYLOAD.video_src_original || '';
    }}

    function refreshVideoSourceUi() {{
      const available = availableVideoSources();
      const fallback = available[0] || '';
      if (!available.includes(currentVideoSourceKind)) currentVideoSourceKind = fallback;
      Object.entries(sourceBtns).forEach(([name, el]) => {{
        if (!el) return;
        const enabled = available.includes(name);
        el.style.display = enabled ? '' : 'none';
        el.classList.toggle('active', enabled && currentVideoSourceKind === name);
      }});
      if (headerVideoSource) {{
        headerVideoSource.textContent = `${{currentVideoSourceKind || 'n/a'}}`;
      }}
      if (headerFsm) {{
        const tEnter = +(fsm.t_enter_sec || 0);
        const tExit = +(fsm.t_exit_sec || 0);
        const tApproach = +(fsm.t_approach_sec || 0);
        const tEmpty = +(fsm.t_empty_min_sec || 0);
        headerFsm.textContent = `e${{tEnter.toFixed(0)}} x${{tExit.toFixed(0)}} a${{tApproach.toFixed(0)}} empty${{tEmpty.toFixed(0)}} ov>=${{overlapMin.toFixed(2)}}`;
      }}
    }}

    function hasEditablePreview() {{
      return editableDetectors().length > 0;
    }}

    function isMotionOnlySelection() {{
      return (graphDet || preferredDetector()) === 'motion';
    }}

    function roiDirty() {{
      return roiKey(currentRoi) !== roiKey(baseRoi);
    }}

    function normalizeRoi(roi) {{
      const vw = Math.max(1, vid.videoWidth || 1);
      const vh = Math.max(1, vid.videoHeight || 1);
      let x1 = Math.max(0, Math.min(vw - 1, +(roi[0] || 0)));
      let y1 = Math.max(0, Math.min(vh - 1, +(roi[1] || 0)));
      let x2 = Math.max(1, Math.min(vw, +(roi[2] || 0)));
      let y2 = Math.max(1, Math.min(vh, +(roi[3] || 0)));
      if (x2 < x1) [x1, x2] = [x2, x1];
      if (y2 < y1) [y1, y2] = [y2, y1];
      const minW = 16;
      const minH = 16;
      if (x2 - x1 < minW) x2 = Math.min(vw, x1 + minW);
      if (y2 - y1 < minH) y2 = Math.min(vh, y1 + minH);
      return [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)];
    }}

    function parseRoiText(text) {{
      const parts = String(text || '').split(/[\\s,;]+/).filter(Boolean).map(v => Number(v));
      if (parts.length !== 4 || parts.some(v => !Number.isFinite(v))) return null;
      return normalizeRoi(parts);
    }}

    function overlayRoi() {{
      if ((graphDet || preferredDetector()) === 'motion') return baseRoi;
      return (Array.isArray(currentRoi) && currentRoi.length === 4) ? currentRoi : baseRoi;
    }}

    function roiPreviewDetectors() {{
      if (graphDet === 'all') return editableDetectors();
      if (graphDet === 'motion') return [];
      return editableDetectors().includes(graphDet) ? [graphDet] : [];
    }}

    function overlaySelectedDetectors() {{
      if (editRoi && roiDirty() && graphDet === 'all') return roiPreviewDetectors();
      return selectedDetectors();
    }}

    function binarySearchFloor(arr, target) {{
      let lo = 0, hi = arr.length - 1, ans = 0;
      while (lo <= hi) {{
        const mid = (lo + hi) >> 1;
        if (arr[mid] <= target) {{ ans = mid; lo = mid + 1; }} else {{ hi = mid - 1; }}
      }}
      return ans;
    }}

    function intersectionRatio(box, roi) {{
      if (!box || box.length < 4 || !roi || roi.length !== 4) return 0;
      const x1 = Math.max(+box[0], +roi[0]);
      const y1 = Math.max(+box[1], +roi[1]);
      const x2 = Math.min(+box[2], +roi[2]);
      const y2 = Math.min(+box[3], +roi[3]);
      const iw = Math.max(0, x2 - x1);
      const ih = Math.max(0, y2 - y1);
      const roiArea = Math.max(1e-9, (+roi[2] - +roi[0]) * (+roi[3] - +roi[1]));
      return (iw * ih) / roiArea;
    }}

    function recomputeDetectorSeries(detName, roi) {{
      const raw = PAYLOAD.raw || {{}};
      const ts = raw.ts || [];
      const rawFrames = raw.frame_idx || [];
      const fps = +((PAYLOAD.meta && PAYLOAD.meta.fps) || 0);
      const frames = raw.people_frames || [];
      const byDet = raw.people_by_detector || {{}};
      const seq = byDet[detName] || [];
      const thr = +((PAYLOAD.meta && PAYLOAD.meta.overlap_min) || 0);
      const out = {{
        best_overlap: [],
        n_people: [],
        best_person_score: [],
        in_zone_now: [],
      }};
      if (!ts.length || !frames.length || !(fps > 0)) return out;
      for (let i = 0; i < ts.length; i++) {{
        const targetFrame = rawFrames.length ? Math.max(0, Math.round(+rawFrames[i] || 0)) : Math.max(0, Math.round((+ts[i] || 0) * fps));
        const idx = binarySearchFloor(frames, targetFrame);
        const boxes = (seq && seq[idx]) ? seq[idx] : [];
        let bestOverlap = 0.0;
        let bestScore = 0.0;
        let nPeople = 0;
        for (const b of boxes) {{
          if (!b || b.length < 4) continue;
          nPeople += 1;
          const score = +(b[4] || 0);
          if (score > bestScore) bestScore = score;
          const ov = intersectionRatio(b, roi);
          if (ov > bestOverlap) bestOverlap = ov;
        }}
        out.best_overlap.push(bestOverlap);
        out.n_people.push(nPeople);
        out.best_person_score.push(bestScore);
        out.in_zone_now.push(bestOverlap >= thr ? 1 : 0);
      }}
      return out;
    }}

    function fmtMetricSec(value) {{
      const num = Number(value);
      return Number.isFinite(num) ? `${{num.toFixed(2)}}s` : 'n/a';
    }}

    function simulateEventsFromInzone(ts, inzone) {{
      const fsm = (PAYLOAD.meta && PAYLOAD.meta.fsm) ? PAYLOAD.meta.fsm : {{}};
      const tEnter = +(fsm.t_enter_sec || 0);
      const tExit = +(fsm.t_exit_sec || 0);
      const tApproach = +(fsm.t_approach_sec || 0);
      const tEmptyMin = +(fsm.t_empty_min_sec || 0);
      const frameIdx = (((PAYLOAD.raw || {{}}).frame_idx) || []);
      const events = [];
      if (!Array.isArray(ts) || !ts.length) return events;

      let state = inzone[0] ? 'OCCUPIED' : 'EMPTY';
      let enterHold = 0.0;
      let exitHold = 0.0;
      let emptyStartTs = +ts[0] || 0;
      let approachEmitted = false;
      let approachHold = 0.0;

      events.push({{
        ts_sec: +ts[0] || 0,
        frame_idx: frameIdx.length ? Math.round(+frameIdx[0] || 0) : 0,
        event_type: state,
      }});

      for (let i = 1; i < ts.length; i++) {{
        const t = +ts[i] || 0;
        const dt = Math.max(0, t - (+ts[i - 1] || 0));
        const inNow = !!inzone[i];

        if (state === 'EMPTY') {{
          if (!inNow) {{
            enterHold = 0.0;
            approachHold = 0.0;
          }} else {{
            if (!approachEmitted) {{
              approachHold += dt;
              if (approachHold >= tApproach) {{
                const emptyDur = Math.max(0, t - emptyStartTs);
                if (emptyDur >= tEmptyMin) {{
                  events.push({{
                    ts_sec: t,
                    frame_idx: frameIdx.length ? Math.round(+frameIdx[i] || 0) : i,
                    event_type: 'APPROACH',
                  }});
                  approachEmitted = true;
                }}
              }}
            }}

            enterHold += dt;
            if (enterHold >= tEnter) {{
              state = 'OCCUPIED';
              exitHold = 0.0;
              enterHold = 0.0;
              events.push({{
                ts_sec: t,
                frame_idx: frameIdx.length ? Math.round(+frameIdx[i] || 0) : i,
                event_type: 'OCCUPIED',
              }});
            }}
          }}
        }} else {{
          if (inNow) {{
            exitHold = 0.0;
          }} else {{
            exitHold += dt;
            if (exitHold >= tExit) {{
              state = 'EMPTY';
              exitHold = 0.0;
              enterHold = 0.0;
              emptyStartTs = t;
              approachEmitted = false;
              approachHold = 0.0;
              events.push({{
                ts_sec: t,
                frame_idx: frameIdx.length ? Math.round(+frameIdx[i] || 0) : i,
                event_type: 'EMPTY',
              }});
            }}
          }}
        }}
      }}
      return events;
    }}

    function computeWaitMetricsFromEvents(events) {{
      const waits = [];
      if (!Array.isArray(events) || !events.length) {{
        return {{ available: false, pairs: 0, mean_sec: null, median_sec: null, p90_sec: null }};
      }}
      let seenOccupied = false;
      const empties = [];
      const approaches = [];
      for (const ev of events) {{
        const type = String((ev && ev.event_type) || '');
        const ts = +(ev && ev.ts_sec || 0);
        if (type === 'OCCUPIED') seenOccupied = true;
        else if (type === 'EMPTY' && seenOccupied) empties.push(ts);
        else if (type === 'APPROACH') approaches.push(ts);
      }}
      let approachIdx = 0;
      for (const tEmpty of empties) {{
        while (approachIdx < approaches.length && !(approaches[approachIdx] >= tEmpty + 1e-9)) approachIdx += 1;
        if (approachIdx < approaches.length) waits.push(approaches[approachIdx] - tEmpty);
      }}
      if (!waits.length) {{
        return {{ available: false, pairs: 0, mean_sec: null, median_sec: null, p90_sec: null }};
      }}
      const arr = waits.slice().sort((a, b) => a - b);
      const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
      const median = arr.length % 2 ? arr[(arr.length - 1) >> 1] : (arr[arr.length / 2 - 1] + arr[arr.length / 2]) / 2;
      const q = (p) => {{
        if (arr.length === 1) return arr[0];
        const pos = (arr.length - 1) * p;
        const lo = Math.floor(pos);
        const hi = Math.ceil(pos);
        if (lo === hi) return arr[lo];
        const t = pos - lo;
        return arr[lo] * (1 - t) + arr[hi] * t;
      }};
      return {{
        available: true,
        pairs: arr.length,
        mean_sec: mean,
        median_sec: median,
        p90_sec: q(0.9),
      }};
    }}

    function computeTouchMetricsFromSeries(ts, inzone) {{
      const fsm = (PAYLOAD.meta && PAYLOAD.meta.fsm) ? PAYLOAD.meta.fsm : {{}};
      const minTouchSec = +(fsm.t_approach_sec || 0);
      if (!Array.isArray(ts) || !ts.length) {{
        return {{ min_touch_sec: minTouchSec, touches: 0, landings: 0, mean_touch_sec: null, mean_gap_sec: null }};
      }}
      const state01 = computeState01(ts, inzone || []);
      const bounced = [];
      let landings = 0;
      let active = false;
      let touchStartTs = 0.0;

      for (let i = 0; i < ts.length; i++) {{
        const t = +ts[i] || 0;
        const inNow = !!((inzone || [])[i]);
        const state = state01[i] ? 'OCCUPIED' : 'EMPTY';

        if (!active) {{
          if (state === 'EMPTY' && inNow) {{
            active = true;
            touchStartTs = t;
          }}
          continue;
        }}

        if (state === 'OCCUPIED' && inNow) {{
          const dur = Math.max(0, t - touchStartTs);
          if (dur >= minTouchSec) landings += 1;
          active = false;
          continue;
        }}

        if ((state === 'EMPTY' && !inNow) || (state !== 'EMPTY' && state !== 'OCCUPIED')) {{
          const dur = Math.max(0, t - touchStartTs);
          if (dur >= minTouchSec) bounced.push([touchStartTs, t]);
          active = false;
        }}
      }}

      if (active) {{
        const lastTs = +ts[ts.length - 1] || 0;
        const dur = Math.max(0, lastTs - touchStartTs);
        if (dur >= minTouchSec) bounced.push([touchStartTs, lastTs]);
      }}

      const durations = bounced.map(([start, end]) => Math.max(0, end - start));
      const gaps = [];
      for (let i = 1; i < bounced.length; i++) gaps.push(Math.max(0, bounced[i][0] - bounced[i - 1][1]));
      const avg = (arr) => arr.length ? (arr.reduce((s, v) => s + v, 0) / arr.length) : null;

      return {{
        min_touch_sec: minTouchSec,
        touches: bounced.length,
        landings,
        mean_touch_sec: avg(durations),
        mean_gap_sec: avg(gaps),
      }};
    }}

    function buildAnalyticsForSeries(src) {{
      const raw = PAYLOAD.raw || {{}};
      const ts = raw.ts || [];
      const inzone = (src && src.in_zone_now) ? src.in_zone_now : [];
      const events = simulateEventsFromInzone(ts, inzone);
      const wait = computeWaitMetricsFromEvents(events);
      const touch = computeTouchMetricsFromSeries(ts, inzone);
      return {{
        wait_available: !!wait.available,
        wait_pairs: wait.pairs,
        wait_mean_sec: wait.mean_sec,
        wait_median_sec: wait.median_sec,
        wait_p90_sec: wait.p90_sec,
        touch_min_sec: touch.min_touch_sec,
        touches: touch.touches,
        landings: touch.landings,
        mean_touch_sec: touch.mean_touch_sec,
        mean_gap_sec: touch.mean_gap_sec,
      }};
    }}

    function analyticsForCurrentSelection() {{
      const detName = graphDet || preferredDetector();
      const preview = roiDirty() && detName !== 'motion';
      const cacheKey = `${{preview ? 'preview' : 'base'}}::${{detName}}::${{preview ? roiKey(currentRoi) : 'base'}}`;
      if (analyticsCache[cacheKey]) return analyticsCache[cacheKey];
      if (!preview) {{
        const base = (PAYLOAD.meta && PAYLOAD.meta.analytics) ? PAYLOAD.meta.analytics : {{}};
        if (detName === preferredDetector()) {{
          analyticsCache[cacheKey] = base;
          return base;
        }}
      }}
      let src = null;
      if (detName === 'all') {{
        const previewSeries = preview ? getRoiPreviewSeries() : null;
        src = previewSeries && previewSeries.all ? previewSeries.all : baseSource('all');
      }} else if (preview) {{
        src = detectorSource(detName);
      }} else {{
        src = baseSource(detName);
      }}
      const analytics = buildAnalyticsForSeries(src || (PAYLOAD.raw || {{}}));
      analyticsCache[cacheKey] = analytics;
      return analytics;
    }}

    function waitMetricMarkup(analytics) {{
      if (analytics && analytics.wait_available && analytics.wait_mean_sec != null) {{
        return `<div class="videoMetricCard primary"><div class="videoMetricKicker">ТЗ</div><div class="videoMetricTitle">Ожидание</div><div class="videoMetricValue">mean=${{fmtMetricSec(analytics.wait_mean_sec)}}</div><div class="videoMetricMeta">pairs=${{Number(analytics.wait_pairs || 0)}} · med=${{fmtMetricSec(analytics.wait_median_sec)}} · p90=${{fmtMetricSec(analytics.wait_p90_sec)}}</div></div>`;
      }}
      return `<div class="videoMetricCard primary empty"><div class="videoMetricKicker">ТЗ</div><div class="videoMetricTitle">Ожидание</div><div class="videoMetricValue">недостаточно событий</div><div class="videoMetricMeta">Нет пары EMPTY→APPROACH</div></div>`;
    }}

    function touchMetricMarkup(analytics) {{
      const cards = [];
      let label = 'Касания';
      if (analytics && analytics.touch_min_sec != null) label += ` >= ${{Number(analytics.touch_min_sec).toFixed(1)}}s`;
      if (analytics && analytics.touches != null) cards.push(`<div class="videoMiniMetric"><div class="videoMiniLabel">${{label}}</div><div class="videoMiniValue">${{Number(analytics.touches || 0)}}</div></div>`);
      if (analytics && analytics.mean_touch_sec != null) cards.push(`<div class="videoMiniMetric"><div class="videoMiniLabel">Ср. длит.</div><div class="videoMiniValue">${{fmtMetricSec(analytics.mean_touch_sec)}}</div></div>`);
      if (analytics && analytics.mean_gap_sec != null) cards.push(`<div class="videoMiniMetric"><div class="videoMiniLabel">Ср. интервал</div><div class="videoMiniValue">${{fmtMetricSec(analytics.mean_gap_sec)}}</div></div>`);
      if (analytics && analytics.landings != null) cards.push(`<div class="videoMiniMetric"><div class="videoMiniLabel">Посадки</div><div class="videoMiniValue">${{Number(analytics.landings || 0)}}</div></div>`);
      return `<div class="videoMetricCard secondary"><div class="videoMetricKicker">Перед посадкой</div><div class="videoMetricTitle">Касания</div><div class="videoMiniGrid">${{cards.join('')}}</div></div>`;
    }}

    function renderHeaderMetrics() {{
      const analytics = analyticsForCurrentSelection();
      if (waitMetricCardWrap) waitMetricCardWrap.innerHTML = waitMetricMarkup(analytics);
      if (touchMetricCardWrap) touchMetricCardWrap.innerHTML = touchMetricMarkup(analytics);
    }}

    function getRoiPreviewSeries() {{
      if (!roiDirty()) return null;
      if (!hasEditablePreview()) return null;
      const by = {{}};
      editableDetectors().forEach(name => {{
        by[name] = recomputeDetectorSeries(name, currentRoi);
      }});
      const names = roiPreviewDetectors();
      const raw = PAYLOAD.raw || {{}};
      const ts = raw.ts || [];
      const n = ts.length;
      const all = {{
        best_overlap: Array.from({{length: n}}, (_, i) => names.length ? Math.max(...names.map(name => +(((by[name] || {{}}).best_overlap || [])[i] || 0))) : 0),
        n_people: Array.from({{length: n}}, (_, i) => names.length ? Math.max(...names.map(name => +(((by[name] || {{}}).n_people || [])[i] || 0))) : 0),
        best_person_score: Array.from({{length: n}}, (_, i) => names.length ? Math.max(...names.map(name => +(((by[name] || {{}}).best_person_score || [])[i] || 0))) : 0),
        in_zone_now: Array.from({{length: n}}, (_, i) => names.some(name => !!((((by[name] || {{}}).in_zone_now || [])[i]) || 0)) ? 1 : 0),
      }};
      return {{ by_detector: by, all }};
    }}

    function detectorSource(name) {{
      const raw = PAYLOAD.raw || {{}};
      const by = raw.by_detector || null;
      const preview = getRoiPreviewSeries();
      if (preview) {{
        if (name === 'all') return preview.all;
        if (preview.by_detector && preview.by_detector[name]) return preview.by_detector[name];
      }}
      return by && by[name] ? by[name] : null;
    }}

    function currentDecisionSummary(cursorIdx) {{
      const raw = PAYLOAD.raw || {{}};
      const ts = raw.ts || [];
      let detName = graphDet || preferredDetector();
      let src = raw;
      if (detName === 'all') {{
        src = aggregateAllSeries();
      }} else {{
        const detSrc = detectorSource(detName);
        if (detSrc) src = detSrc;
      }}
      const overlap = +(((src.best_overlap || [])[cursorIdx]) || 0);
      const inZone = !!((((src.in_zone_now || [])[cursorIdx]) || 0));
      const stateSeries = computeState01(ts, src.in_zone_now || raw.in_zone_now || []);
      const state = stateSeries[cursorIdx] ? 'OCCUPIED' : 'EMPTY';
      return {{
        detector: detName,
        overlap,
        inZone,
        state,
      }};
    }}

    function baseSource(name) {{
      const raw = PAYLOAD.raw || {{}};
      const by = raw.by_detector || {{}};
      if (name === 'all') {{
        const dets = availableDetectors().map(n => by[n]).filter(Boolean);
        const n = (raw.ts || []).length;
        if (!dets.length) return raw;
        return {{
          best_overlap: Array.from({{length: n}}, (_, i) => Math.max(...dets.map(d => +((d.best_overlap && d.best_overlap[i]) || 0)))),
          n_people: Array.from({{length: n}}, (_, i) => Math.max(...dets.map(d => +((d.n_people && d.n_people[i]) || 0)))),
          best_person_score: Array.from({{length: n}}, (_, i) => Math.max(...dets.map(d => +((d.best_person_score && d.best_person_score[i]) || 0)))),
          in_zone_now: Array.from({{length: n}}, (_, i) => dets.some(d => !!((d.in_zone_now && d.in_zone_now[i]) || 0)) ? 1 : 0),
        }};
      }}
      return by[name] || null;
    }}

    function previewDiffSummary() {{
      if (!roiDirty()) return '';
      const detName = graphDet || preferredDetector();
      if (detName === 'motion') return 'preview diff unavailable for motion';
      const srcPreview = detName === 'all' ? aggregateAllSeries() : detectorSource(detName);
      const srcBase = baseSource(detName);
      const raw = PAYLOAD.raw || {{}};
      const ts = raw.ts || [];
      if (!srcPreview || !srcBase || !ts.length) return '';
      const aO = srcBase.best_overlap || [];
      const bO = srcPreview.best_overlap || [];
      const aI = srcBase.in_zone_now || [];
      const bI = srcPreview.in_zone_now || [];
      let diffOverlap = 0;
      let diffInZone = 0;
      for (let i = 0; i < ts.length; i++) {{
        if (Math.abs((+aO[i] || 0) - (+bO[i] || 0)) > 1e-6) diffOverlap += 1;
        if ((+aI[i] || 0) !== (+bI[i] || 0)) diffInZone += 1;
      }}
      const sA = computeState01(ts, aI);
      const sB = computeState01(ts, bI);
      let diffState = 0;
      for (let i = 0; i < ts.length; i++) {{
        if ((+sA[i] || 0) !== (+sB[i] || 0)) diffState += 1;
      }}
      return {{
        diffOverlap,
        diffInZone,
        diffState,
        baseSource: srcBase,
      }};
    }}

    function renderDecisionReadout(cursorIdx) {{
      const d = currentDecisionSummary(cursorIdx);
      const preview = roiDirty() && d.detector !== 'motion';
      const label = d.detector === 'all' ? 'all' : d.detector;
      const diff = preview ? previewDiffSummary() : null;
      let extra = '';
      if (diff && typeof diff === 'object' && diff.baseSource) {{
        const baseInZone = !!((((diff.baseSource.in_zone_now || [])[cursorIdx]) || 0));
        const raw = PAYLOAD.raw || {{}};
        const ts = raw.ts || [];
        const baseStateSeries = computeState01(ts, diff.baseSource.in_zone_now || raw.in_zone_now || []);
        const baseState = baseStateSeries[cursorIdx] ? 'OCCUPIED' : 'EMPTY';
        extra = ` · base@t: in_zone_now=${{baseInZone ? 1 : 0}} state=${{baseState}} · diff-total: overlap=${{diff.diffOverlap}} in_zone=${{diff.diffInZone}} state=${{diff.diffState}}`;
      }}
      if (badgeDetector) badgeDetector.innerHTML = `<span class="pillIcon">D</span><span class="pillText">${{label}}</span>`;
      if (badgeState) {{
        badgeState.innerHTML = `<span class="pillIcon">S</span><span class="pillText">${{d.state}}</span>`;
        badgeState.className = `statusBadge ${{d.state === 'OCCUPIED' ? 'bad' : 'good'}}`;
      }}
      if (badgeInZone) {{
        badgeInZone.innerHTML = `<span class="pillIcon">Z</span><span class="pillText">${{d.inZone ? 1 : 0}}</span>`;
        badgeInZone.className = `statusBadge ${{d.inZone ? 'bad' : 'good'}}`;
      }}
      if (badgeOverlap) badgeOverlap.innerHTML = `<span class="pillIcon">O</span><span class="pillText">${{d.overlap.toFixed(3)}}</span>`;
      if (badgeFrameTime) badgeFrameTime.innerHTML = `<span class="pillIcon">T</span><span class="pillText">${{(renderTime || 0).toFixed(2)}}s</span>`;
    }}

    function updateRoiUi() {{
      if ((!hasEditablePreview() || graphDet === 'motion') && editRoi) {{
        editRoi = false;
        roiDrag = null;
      }}
      const roi = overlayRoi();
      if (headerRoi) headerRoi.textContent = roi && roi.length === 4 ? `${{roi[0]}},${{roi[1]}},${{roi[2]}},${{roi[3]}}` : '-';
      let status = '';
      if (!hasEditablePreview()) {{
        status = 'n/a';
      }} else if (graphDet === 'motion') {{
        status = 'motion fixed';
      }} else if (graphDet === 'all' && roiDirty()) {{
        status = 'yolo+dnn';
      }} else if (roiDirty()) {{
        status = `${{graphDet || preferredDetector()}}`;
      }} else {{
        status = 'base';
      }}
      if (headerRoiStatus) headerRoiStatus.textContent = status;
      if (btnEditRoi) {{
        btnEditRoi.style.display = 'none';
        btnEditRoi.classList.toggle('active', !!editRoi);
        btnEditRoi.disabled = !hasEditablePreview() || graphDet === 'motion';
      }}
      if (btnResetRoi) btnResetRoi.disabled = !roiDirty();
      refreshVideoSourceUi();
      updateOverlayInteractivity();
    }}

    function detLabel(name) {{
      if (name === 'opencv-dnn') return 'DNN';
      if (name === 'yolo') return 'YOLO';
      if (name === 'motion') return 'M';
      return String(name || '').toUpperCase();
    }}

    function detState(name, cursorIdx) {{
      const raw = PAYLOAD.raw || {{}};
      const src = detectorSource(name);
      const ts = raw.ts || [];
      const inzoneSeries = src && src.in_zone_now ? src.in_zone_now : [];
      const stateSeries = computeState01(ts, inzoneSeries);
      return {{
        name,
        inZone: src && src.in_zone_now ? !!src.in_zone_now[cursorIdx] : false,
        stateOn: !!stateSeries[cursorIdx],
        overlap: src && src.best_overlap ? +(src.best_overlap[cursorIdx] || 0) : 0,
        score: src && src.best_person_score ? +(src.best_person_score[cursorIdx] || 0) : 0,
        nPeople: src && src.n_people ? +(src.n_people[cursorIdx] || 0) : 0,
      }};
    }}

    function aggregateAllSeries() {{
      const preview = getRoiPreviewSeries();
      if (preview && preview.all) return preview.all;
      const raw = PAYLOAD.raw || {{}};
      const by = raw.by_detector || {{}};
      const dets = availableDetectors().map(name => by[name]).filter(Boolean);
      const n = (raw.ts || []).length;
      if (!dets.length) return raw;
      return {{
        best_overlap: Array.from({{length: n}}, (_, i) => Math.max(...dets.map(d => +((d.best_overlap && d.best_overlap[i]) || 0)))),
        n_people: Array.from({{length: n}}, (_, i) => Math.max(...dets.map(d => +((d.n_people && d.n_people[i]) || 0)))),
        best_person_score: Array.from({{length: n}}, (_, i) => Math.max(...dets.map(d => +((d.best_person_score && d.best_person_score[i]) || 0)))),
        in_zone_now: Array.from({{length: n}}, (_, i) => dets.some(d => !!((d.in_zone_now && d.in_zone_now[i]) || 0)) ? 1 : 0),
      }};
    }}

    function aggregateAll(cursorIdx) {{
      const dets = availableDetectors().map(name => detState(name, cursorIdx));
      return {{
        best_overlap: dets.length ? Math.max(...dets.map(d => d.overlap)) : 0,
        n_people: dets.length ? Math.max(...dets.map(d => d.nPeople)) : 0,
        best_person_score: dets.length ? Math.max(...dets.map(d => d.score)) : 0,
        in_zone_now: dets.some(d => d.inZone) ? 1 : 0,
      }};
    }}

    function setDet(name) {{
      if (!detBtns[name]) return;
      peopleDet = name;
      graphDet = name;
      Object.entries(detBtns).forEach(([k, el]) => {{
        const enabled = k === 'all' ? availableDetectors().length > 1 : availableDetectors().includes(k);
        el.classList.toggle('active', k === name);
        el.style.display = enabled ? '' : 'none';
      }});
      updateRoiUi();
      setSig(sigIndex());
    }}
    Object.entries(detBtns).forEach(([k, el]) => {{
      el.addEventListener('click', () => setDet(k));
    }});

    btnRoi.addEventListener('click', () => {{
      showRoi = !showRoi;
      btnRoi.classList.toggle('active', showRoi);
      drawPeople(sigIndex());
    }});
    btnPeople.addEventListener('click', () => {{
      showPeople = !showPeople;
      btnPeople.classList.toggle('active', showPeople);
      drawPeople(sigIndex());
    }});
    if (btnEditRoi) {{
      btnEditRoi.addEventListener('click', () => {{
        return;
      }});
    }}
    function copyRoiValue() {{
      const value = headerRoi ? String(headerRoi.textContent || '').replace(/^ROI\s+/, '').trim() : '';
      if (!value) return;
      const done = () => {{
        if (headerRoiStatus) headerRoiStatus.textContent = `ROI copied: ${{value}}`;
      }};
      try {{
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          navigator.clipboard.writeText(value).then(done).catch(() => done());
        }} else {{
          done();
        }}
      }} catch (_err) {{
        done();
      }}
    }}
    if (headerRoi) {{
      headerRoi.addEventListener('click', copyRoiValue);
    }}
    if (btnCopyRoi) {{
      btnCopyRoi.addEventListener('click', copyRoiValue);
    }}
    btnResetRoi.addEventListener('click', () => {{
      currentRoi = baseRoi.slice();
      clearSavedRoi();
      roiDrag = null;
      updateRoiUi();
      setSig(sigIndex());
    }});

    function fmtTime(sec) {{
      sec = Math.max(0, sec || 0);
      const m = Math.floor(sec / 60);
      const s = sec - m * 60;
      if (m > 0) return `${{m}}:${{s.toFixed(0).padStart(2,'0')}}`;
      return `${{s.toFixed(2)}}`;
    }}

    function sigIndexAt(t) {{
      const raw = PAYLOAD.raw || {{}};
      if (!raw.available) return 0;
      const ts = raw.ts || [];
      let lo = 0, hi = ts.length - 1, ans = 0;
      while (lo <= hi) {{
        const mid = (lo + hi) >> 1;
        if (ts[mid] <= t) {{ ans = mid; lo = mid + 1; }} else {{ hi = mid - 1; }}
      }}
      return ans;
    }}

    function sigIndex() {{
      return sigIndexAt(renderTime || 0);
    }}

    function resizeOverlay() {{
      const vw = vid.videoWidth || 0;
      const vh = vid.videoHeight || 0;
      if (!vw || !vh) return;
      const rect = vid.getBoundingClientRect();
      const wrapRect = overlay.parentElement.getBoundingClientRect();
      const cssW = Math.max(1, Math.round(rect.width));
      const cssH = Math.max(1, Math.round(rect.height));
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      overlay.style.left = `${{Math.round(rect.left - wrapRect.left)}}px`;
      overlay.style.top = `${{Math.round(rect.top - wrapRect.top)}}px`;
      overlay.style.width = `${{cssW}}px`;
      overlay.style.height = `${{cssH}}px`;
      overlay.width = Math.max(1, Math.round(cssW * dpr));
      overlay.height = Math.max(1, Math.round(cssH * dpr));
      ctxO.setTransform(dpr, 0, 0, dpr, 0, 0);
      overlayScaleX = cssW / vw;
      overlayScaleY = cssH / vh;
      overlayCssW = cssW;
      overlayCssH = cssH;
      updateRoiUi();
      drawPeople(sigIndex());
    }}

    function roiRectCss(roi) {{
      return {{
        x: roi[0] * overlayScaleX,
        y: roi[1] * overlayScaleY,
        w: (roi[2] - roi[0]) * overlayScaleX,
        h: (roi[3] - roi[1]) * overlayScaleY,
      }};
    }}

    function drawRoiHandles(roi) {{
      if (!(roiDrag || roiHoverMode) || !roi || roi.length !== 4) return;
      const r = roiRectCss(roi);
      const pts = [
        [r.x, r.y],
        [r.x + r.w, r.y],
        [r.x, r.y + r.h],
        [r.x + r.w, r.y + r.h],
      ];
      ctxO.save();
      ctxO.fillStyle = 'rgba(255,255,255,0.95)';
      ctxO.strokeStyle = 'rgba(12,14,18,0.92)';
      ctxO.lineWidth = 1.5;
      pts.forEach(([px, py]) => {{
        ctxO.beginPath();
        ctxO.rect(px - 5, py - 5, 10, 10);
        ctxO.fill();
        ctxO.stroke();
      }});
      ctxO.restore();
    }}

    function drawRoi() {{
      const roi = overlayRoi();
      if (!roi || roi.length !== 4) return;
      const cursorIdx = sigIndex();
      const x = roi[0] * overlayScaleX;
      const y = roi[1] * overlayScaleY;
      const w = (roi[2] - roi[0]) * overlayScaleX;
      const h = (roi[3] - roi[1]) * overlayScaleY;
      const dets = overlaySelectedDetectors().map(name => detState(name, cursorIdx));
      const labelSlots = [
        {{ax: 'left', ay: 'top'}},
        {{ax: 'right', ay: 'top'}},
        {{ax: 'left', ay: 'bottom'}},
        {{ax: 'right', ay: 'bottom'}},
      ];
      dets.forEach((det, idx) => {{
        const offset = idx * Math.max(6, Math.round(overlayCssW / 220));
        const dx = x + offset;
        const dy = y + offset;
        const dw = Math.max(16, w - offset * 2);
        const dh = Math.max(16, h - offset * 2);
        const strokeColor = det.stateOn ? 'rgba(255,77,79,1.0)' : 'rgba(32,201,151,1.0)';
        const fillColor = det.inZone ? 'rgba(255,77,79,0.16)' : 'rgba(32,201,151,0.12)';
        const labelText = detLabel(det.name);
        ctxO.save();
        ctxO.lineWidth = Math.max(2, overlayCssW / 700);
        ctxO.fillStyle = fillColor;
        ctxO.fillRect(dx, dy, dw, dh);
        ctxO.strokeStyle = strokeColor;
        ctxO.setLineDash([]);
        ctxO.strokeRect(dx, dy, dw, dh);
        const fontPx = Math.max(12, Math.round(overlayCssW / 55));
        const padX = Math.max(8, Math.round(fontPx * 0.55));
        const padY = Math.max(6, Math.round(fontPx * 0.35));
        ctxO.font = `${{fontPx}}px ui-sans-serif, system-ui`;
        const textW = ctxO.measureText(labelText).width;
        const boxW = textW + padX * 2;
        const boxH = fontPx + padY * 2;
        const slot = labelSlots[idx % labelSlots.length];
        const boxX = slot.ax === 'right'
          ? Math.max(8, dx + dw - boxW - 8)
          : dx + 8;
        const boxY = slot.ay === 'bottom'
          ? Math.max(8, dy + dh - boxH - 8)
          : Math.max(8, dy + 8);
        ctxO.fillStyle = 'rgba(12,14,18,0.88)';
        ctxO.fillRect(boxX, boxY, boxW, boxH);
        ctxO.strokeStyle = strokeColor;
        ctxO.lineWidth = 2;
        ctxO.strokeRect(boxX, boxY, boxW, boxH);
        ctxO.fillStyle = 'rgba(255,245,230,1.0)';
        ctxO.fillText(labelText, boxX + padX, boxY + fontPx + padY - 2);
        ctxO.restore();
      }});
      drawRoiHandles(roi);
    }}

    function _drawBoxes(boxes, color) {{
      if (!Array.isArray(boxes)) return;
      ctxO.save();
      ctxO.lineWidth = Math.max(2, overlayCssW / 900);
      ctxO.strokeStyle = color;
      ctxO.setLineDash([]);
      boxes.forEach(b => {{
        if (!b || b.length < 4) return;
        const x1 = +b[0], y1 = +b[1], x2 = +b[2], y2 = +b[3];
        ctxO.strokeRect(
          x1 * overlayScaleX,
          y1 * overlayScaleY,
          (x2 - x1) * overlayScaleX,
          (y2 - y1) * overlayScaleY
        );
      }});
      ctxO.restore();
    }}

    function detectorBoxColor(name) {{
      if (name === 'yolo') return 'rgba(77,171,247,0.98)';
      if (name === 'opencv-dnn') return 'rgba(255,146,43,0.98)';
      if (name === 'motion') return 'rgba(242,201,76,0.98)';
      return 'rgba(255,255,255,0.96)';
    }}

    function peopleIndexAt(t) {{
      const raw = PAYLOAD.raw || {{}};
      const frames = raw.people_frames || [];
      if (!frames.length) return 0;
      const fps = +((PAYLOAD.meta && PAYLOAD.meta.fps) || 0);
      if (!(fps > 0)) return 0;
      const target = Math.max(0, Math.round((t || 0) * fps));
      let lo = 0, hi = frames.length - 1, ans = 0;
      while (lo <= hi) {{
        const mid = (lo + hi) >> 1;
        if (frames[mid] <= target) {{ ans = mid; lo = mid + 1; }} else {{ hi = mid - 1; }}
      }}
      if (ans + 1 < frames.length && Math.abs(frames[ans + 1] - target) < Math.abs(frames[ans] - target)) {{
        return ans + 1;
      }}
      return ans;
    }}

    function peopleIndex() {{
      return peopleIndexAt(renderTime || 0);
    }}

    function drawPeople(_i) {{
      ctxO.clearRect(0, 0, overlayCssW, overlayCssH);
      if (showRoi) drawRoi();
      if (!showPeople) return;
      const raw = PAYLOAD.raw || {{}};
      if (!raw.available) return;
      const i = peopleIndex();
      const byDet = raw.people_by_detector || null;
      if (byDet) {{
        overlaySelectedDetectors().forEach(name => {{
          const seq = byDet[name];
          const boxes = (seq && seq[i]) ? seq[i] : [];
          _drawBoxes(boxes, detectorBoxColor(name));
        }});
        return;
      }}
      const seq = raw.people || null;
      if (seq && seq[i]) _drawBoxes(seq[i], 'rgba(77,171,247,0.9)');
    }}

    function mkEventRow(e) {{
      const t = fmtTime(e.ts);
      const typ = (e.type || '').toUpperCase();
      return `<div class="flex" style="justify-content:space-between; border-bottom: 1px solid rgba(255,255,255,0.08); padding: 6px 0;">
        <div><code>${{t}}</code> <span class="small">${{typ}}</span></div>
        <div class="small">frame=${{e.frame}}</div>
      </div>`;
    }}

    function renderEvents() {{
      const ev = PAYLOAD.events || [];
      const el = document.getElementById('eventsTable');
      if (!ev.length) {{ el.innerHTML = '<div class="small">events.csv не найден или пустой.</div>'; return; }}
      el.innerHTML = ev.map(mkEventRow).join('');
    }}

    function renderMeta() {{
      const m = PAYLOAD.meta || {{}};
      const el = document.getElementById('metaKv');
      const rows = [
        ['out_dir', m.out_dir || ''],
        ['roi_xyxy', (m.roi_xyxy && m.roi_xyxy.length) ? JSON.stringify(m.roi_xyxy) : ''],
        ['overlap_min (hint)', (m.overlap_min || 0).toFixed(3)],
        ['raw.available', String((PAYLOAD.raw && PAYLOAD.raw.available) ? true : false)],
      ];
      el.innerHTML = rows.map(([k,v]) => `<div>${{k}}</div><div><code>${{String(v)}}</code></div>`).join('');
      document.getElementById('reportTxt').textContent = m.report_txt || '';
    }}

    function ctxFor(canvasId) {{
      const c = document.getElementById(canvasId);
      c.width = Math.floor(c.clientWidth * devicePixelRatio);
      c.height = Math.floor(c.clientHeight * devicePixelRatio);
      const ctx = c.getContext('2d');
      ctx.scale(devicePixelRatio, devicePixelRatio);
      return ctx;
    }}

    function drawSeries(ctx, ts, ys, color, yMin, yMax) {{
      const w = ctx.canvas.clientWidth;
      const h = ctx.canvas.clientHeight;
      const padX = 0;
      const padY = 8;
      ctx.clearRect(0,0,w,h);
      ctx.fillStyle = 'rgba(0,0,0,0.18)';
      ctx.fillRect(0,0,w,h);
      if (!ts.length) return;
      const t0 = ts[0], t1 = ts[ts.length-1];
      const toX = (t) => padX + (t - t0) / Math.max(1e-9, (t1 - t0)) * Math.max(1, (w - 2*padX));
      const toY = (y) => h - padY - (y - yMin) / Math.max(1e-9, (yMax - yMin)) * (h - 2*padY);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (let i=0;i<ts.length;i++) {{
        const x = toX(ts[i]);
        const y = toY(ys[i]);
        if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }}
      ctx.stroke();
      return {{toX, toY, t0, t1, w, h, padX, padY}};
    }}

    function drawVLine(ctx, x, color) {{
      const w = ctx.canvas.clientWidth;
      const h = ctx.canvas.clientHeight;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
      ctx.restore();
    }}

    function drawHLine(ctx, y, color) {{
      const w = ctx.canvas.clientWidth;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([6,4]);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.restore();
    }}

    function drawSeriesOn(ctx, ts, ys, color, toX, toY) {{
      if (!ts.length || !ys.length) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i < ts.length; i++) {{
        const x = toX(ts[i]);
        const y = toY(ys[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
      ctx.stroke();
      ctx.restore();
    }}

    function drawStepSeriesOn(ctx, ts, ys, color, toX, toY) {{
      if (!ts.length || !ys.length) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i < ts.length; i++) {{
        const x = toX(ts[i]);
        const y = toY(ys[i]);
        if (i === 0) {{
          ctx.moveTo(x, y);
        }} else {{
          const px = toX(ts[i]);
          const py = toY(ys[i - 1]);
          ctx.lineTo(px, py);
          ctx.lineTo(x, y);
        }}
      }}
      ctx.stroke();
      ctx.restore();
    }}

    function smoothSeries(ys, alpha) {{
      if (!ys.length) return [];
      const out = [+(ys[0] || 0)];
      let prev = out[0];
      for (let i = 1; i < ys.length; i++) {{
        const cur = +(ys[i] || 0);
        prev = alpha * cur + (1 - alpha) * prev;
        out.push(prev);
      }}
      return out;
    }}

    function drawSmoothSeriesOn(ctx, ts, ys, color, toX, toY) {{
      if (!ts.length || !ys.length) return;
      const smoothed = smoothSeries(ys, 0.35);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.beginPath();
      for (let i = 0; i < ts.length; i++) {{
        const x = toX(ts[i]);
        const y = toY(smoothed[i]);
        if (i === 0) {{
          ctx.moveTo(x, y);
        }} else {{
          const px = toX(ts[i - 1]);
          const py = toY(smoothed[i - 1]);
          const mx = (px + x) / 2;
          const my = (py + y) / 2;
          ctx.quadraticCurveTo(px, py, mx, my);
        }}
      }}
      const last = ts.length - 1;
      if (last > 0) {{
        ctx.lineTo(toX(ts[last]), toY(smoothed[last]));
      }}
      ctx.stroke();
      ctx.restore();
    }}

    function drawBinaryBandOn(ctx, ts, ys, toX, bandY, bandH, fillColor, strokeColor) {{
      if (!ts.length || !ys.length) return;
      const totalW = ctx.canvas.clientWidth;
      let start = -1;
      const flush = (fromIdx, toIdxExclusive) => {{
        if (fromIdx < 0) return;
        const x1 = toX(ts[fromIdx]);
        let x2 = totalW;
        if (toIdxExclusive > 0 && toIdxExclusive < ts.length) {{
          x2 = toX(ts[toIdxExclusive]);
        }}
        const width = Math.max(2, x2 - x1);
        ctx.fillStyle = fillColor;
        ctx.fillRect(x1, bandY, width, bandH);
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x1, bandY, width, bandH);
      }};
      ctx.save();
      for (let i = 0; i < ys.length; i++) {{
        const on = !!ys[i];
        if (on && start < 0) start = i;
        if (!on && start >= 0) {{
          flush(start, i);
          start = -1;
        }}
      }}
      if (start >= 0) flush(start, ts.length);
      ctx.restore();
    }}

    function computeState01(ts, inzone) {{
      const fsm = (PAYLOAD.meta && PAYLOAD.meta.fsm) ? PAYLOAD.meta.fsm : {{}};
      const tEnter = +(fsm.t_enter_sec || 0);
      const tExit = +(fsm.t_exit_sec || 0);
      const state01 = [];
      if (!ts.length) return state01;

      let state = inzone[0] ? 'OCCUPIED' : 'EMPTY';
      let enterHold = 0.0;
      let exitHold = 0.0;
      state01.push(state === 'OCCUPIED' ? 1 : 0);

      for (let i = 1; i < ts.length; i++) {{
        const dt = Math.max(0, (+ts[i] || 0) - (+ts[i - 1] || 0));
        const inNow = !!inzone[i];
        if (state === 'EMPTY') {{
          if (!inNow) {{
            enterHold = 0.0;
          }} else {{
            enterHold += dt;
            if (enterHold >= tEnter) {{
              state = 'OCCUPIED';
              exitHold = 0.0;
              enterHold = 0.0;
            }}
          }}
        }} else {{
          if (inNow) {{
            exitHold = 0.0;
          }} else {{
            exitHold += dt;
            if (exitHold >= tExit) {{
              state = 'EMPTY';
              exitHold = 0.0;
              enterHold = 0.0;
            }}
          }}
        }}
        state01.push(state === 'OCCUPIED' ? 1 : 0);
      }}
      return state01;
    }}

    function renderAll(cursorIdx) {{
      const raw = PAYLOAD.raw || {{}};
      if (!raw.available) return;
      const ts = raw.ts || [];
      const by = raw.by_detector || null;
      let src = raw;
      if (graphDet === 'all' && by) {{
        src = aggregateAllSeries();
      }} else {{
        const detSrc = detectorSource(graphDet);
        if (detSrc) src = detSrc;
      }}

      const overlap = src.best_overlap || raw.best_overlap || [];
      const nppl = src.n_people || raw.n_people || [];
      const score = src.best_person_score || raw.best_person_score || [];
      const inzone = src.in_zone_now || raw.in_zone_now || [];
      const state01 = computeState01(ts, inzone);

      // Unified timeline
      {{
        const ctx = ctxFor('cTimeline');
        const w = ctx.canvas.clientWidth;
        const h = ctx.canvas.clientHeight;
        const padX = 0;
        const padY = 10;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = 'rgba(0,0,0,0.18)';
        ctx.fillRect(0, 0, w, h);
        if (ts.length) {{
          const t0 = ts[0], t1 = ts[ts.length - 1];
          const maxPeople = Math.max(1, ...nppl);
          const innerH = Math.max(1, h - padY * 2);
          const gap = 10;
          const topH = innerH * 0.30;
          const midH = innerH * 0.38;
          const botH = Math.max(24, innerH - topH - midH - gap * 2);
          const topY0 = padY;
          const topY1 = topY0 + topH;
          const midY0 = topY1 + gap;
          const midY1 = midY0 + midH;
          const botY0 = midY1 + gap;
          const botY1 = botY0 + botH;
          const toX = (t) => padX + (t - t0) / Math.max(1e-9, (t1 - t0)) * Math.max(1, (w - 2 * padX));
          const toYTop = (y) => topY1 - (y / Math.max(1e-9, 1)) * topH;
          const toYMid = (y) => midY1 - (y / Math.max(1e-9, 1)) * midH;
          const toYBot = (y) => botY1 - (y / Math.max(1e-9, maxPeople)) * botH;

          ctx.save();
          ctx.strokeStyle = 'rgba(255,255,255,0.10)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, topY1 + gap * 0.5);
          ctx.lineTo(w, topY1 + gap * 0.5);
          ctx.moveTo(0, midY1 + gap * 0.5);
          ctx.lineTo(w, midY1 + gap * 0.5);
          ctx.stroke();
          ctx.fillStyle = 'rgba(255,255,255,0.42)';
          ctx.font = '12px ui-sans-serif, system-ui';
          ctx.fillText('decision', 10, topY0 + 12);
          ctx.fillText('signals', 10, midY0 + 12);
          ctx.fillText('people', 10, botY0 + 12);
          ctx.restore();

          ctx.save();
          ctx.fillStyle = 'rgba(255,255,255,0.03)';
          ctx.fillRect(0, topY0, w, topH);
          ctx.fillRect(0, midY0, w, midH);
          ctx.fillRect(0, botY0, w, botH);
          ctx.restore();

          const stateBandY = topY0 + topH * 0.12;
          const stateBandH = topH * 0.34;
          const inzoneBandY = topY0 + topH * 0.58;
          const inzoneBandH = topH * 0.18;
          drawBinaryBandOn(ctx, ts, state01, toX, stateBandY, stateBandH, 'rgba(255,77,79,0.28)', 'rgba(255,77,79,0.95)');
          drawBinaryBandOn(ctx, ts, inzone, toX, inzoneBandY, inzoneBandH, 'rgba(32,201,151,0.24)', 'rgba(32,201,151,0.90)');

          drawSmoothSeriesOn(ctx, ts, overlap, 'rgba(242,201,76,0.95)', toX, toYMid);
          drawSmoothSeriesOn(ctx, ts, score, 'rgba(77,171,247,0.92)', toX, toYMid);
          const thr = (PAYLOAD.meta && PAYLOAD.meta.overlap_min) ? +PAYLOAD.meta.overlap_min : 0;
          if (thr > 0) drawHLine(ctx, toYMid(thr), 'rgba(242,201,76,0.45)');
          drawStepSeriesOn(ctx, ts, nppl, 'rgba(255,255,255,0.92)', toX, toYBot);
          drawVLine(ctx, toX(ts[cursorIdx] || 0), 'rgba(255,255,255,0.35)');
        }}
      }}
    }}

    function setSig(i) {{
      renderAll(i);
      drawPeople(i);
      renderDecisionReadout(i);
      renderHeaderMetrics();
    }}

    function attachVideo() {{
      const src = currentVideoSourceSrc();
      if (!src) return;
      vid.src = src;
    }}

    function switchVideoSource(kind) {{
      const available = availableVideoSources();
      if (!available.includes(kind)) return;
      if (currentVideoSourceKind === kind && vid.getAttribute('src')) return;
      const keepTime = renderTime || vid.currentTime || 0;
      currentVideoSourceKind = kind;
      refreshVideoSourceUi();
      const src = currentVideoSourceSrc();
      if (!src) return;
      vid.src = src;
      vid.addEventListener('loadedmetadata', () => {{
        const t = Math.max(0, Math.min(keepTime, vid.duration || keepTime || 0));
        renderTime = t;
        vid.currentTime = t;
        timeSlider.max = String(vid.duration || 0);
        timeSlider.value = String(t);
        tLabel.textContent = String(t.toFixed(2));
        resizeOverlay();
        setSig(sigIndexAt(t));
      }}, {{ once: true }});
    }}

    function hook() {{
      refreshVideoSourceUi();
      attachVideo();
      renderEvents();
      renderMeta();
      setDet(initialDetector());
      updateRoiUi();

      vid.addEventListener('loadedmetadata', () => {{
        if (Array.isArray(currentRoi) && currentRoi.length === 4) {{
          currentRoi = normalizeRoi(currentRoi);
        }}
        timeSlider.max = String(vid.duration || 0);
        resizeOverlay();
        updateRoiUi();
        setSig(sigIndex());
      }});
      vid.addEventListener('timeupdate', () => {{
        const t = vid.currentTime || 0;
        renderTime = t;
        timeSlider.value = String(t);
        tLabel.textContent = String(t.toFixed(2));
        setSig(sigIndexAt(t));
      }});
      vid.addEventListener('seeked', () => {{
        renderTime = vid.currentTime || renderTime || 0;
        setSig(sigIndexAt(renderTime));
      }});
      window.addEventListener('resize', () => {{
        resizeOverlay();
        setSig(sigIndex());
      }});

      timeSlider.addEventListener('input', () => {{
        const t = +timeSlider.value;
        renderTime = t;
        tLabel.textContent = String(t.toFixed(2));
        vid.currentTime = t;
        setSig(sigIndexAt(t));
      }});

      Object.entries(sourceBtns).forEach(([kind, el]) => {{
        if (!el) return;
        el.addEventListener('click', () => switchVideoSource(kind));
      }});

      overlay.addEventListener('pointerdown', (ev) => {{
        const roi = overlayRoi();
        if (!roi || roi.length !== 4) return;
        const rect = overlay.getBoundingClientRect();
        const mx = ev.clientX - rect.left;
        const my = ev.clientY - rect.top;
        const mode = roiHitModeAt(mx, my);
        if (!mode) return;
        roiDrag = {{
          mode,
          startX: mx,
          startY: my,
          startRoi: roi.slice(),
        }};
        editRoi = true;
        roiHoverMode = mode;
        updateOverlayInteractivity();
        overlay.setPointerCapture(ev.pointerId);
        ev.preventDefault();
      }});

      const finishRoiDrag = (ev = null, commitEdit = false) => {{
        if (!roiDrag) return;
        try {{
          if (ev && typeof ev.pointerId === 'number' && overlay.hasPointerCapture(ev.pointerId)) {{
            overlay.releasePointerCapture(ev.pointerId);
          }}
        }} catch (_err) {{}}
        roiDrag = null;
        roiHoverMode = '';
        if (commitEdit) {{
          editRoi = false;
          saveCurrentRoi();
          updateRoiUi();
        }} else {{
          updateOverlayInteractivity();
        }}
      }};

      const syncRoiHover = (clientX, clientY, sourceEl) => {{
        if (roiDrag) return;
        const rect = sourceEl.getBoundingClientRect();
        const mx = clientX - rect.left;
        const my = clientY - rect.top;
        roiHoverMode = roiHitModeAt(mx, my);
        updateOverlayInteractivity();
        drawPeople(sigIndex());
      }};

      videoWrap.addEventListener('pointermove', (ev) => {{
        syncRoiHover(ev.clientX, ev.clientY, videoWrap);
      }});
      videoWrap.addEventListener('pointerleave', () => {{
        if (roiDrag) return;
        roiHoverMode = '';
        updateOverlayInteractivity();
        drawPeople(sigIndex());
      }});

      overlay.addEventListener('pointermove', (ev) => {{
        if (!roiDrag) {{
          syncRoiHover(ev.clientX, ev.clientY, overlay);
          return;
        }}
        const rect = overlay.getBoundingClientRect();
        const mx = ev.clientX - rect.left;
        const my = ev.clientY - rect.top;
        const dx = (mx - roiDrag.startX) / Math.max(1e-9, overlayScaleX);
        const dy = (my - roiDrag.startY) / Math.max(1e-9, overlayScaleY);
        let [x1, y1, x2, y2] = roiDrag.startRoi.slice();
        if (roiDrag.mode === 'move') {{
          x1 += dx; x2 += dx; y1 += dy; y2 += dy;
        }} else if (roiDrag.mode === 'tl') {{
          x1 += dx; y1 += dy;
        }} else if (roiDrag.mode === 'tr') {{
          x2 += dx; y1 += dy;
        }} else if (roiDrag.mode === 'bl') {{
          x1 += dx; y2 += dy;
        }} else if (roiDrag.mode === 'br') {{
          x2 += dx; y2 += dy;
        }}
        currentRoi = normalizeRoi([x1, y1, x2, y2]);
        updateRoiUi();
        setSig(sigIndex());
        const outOfBounds = mx < 0 || my < 0 || mx > rect.width || my > rect.height;
        if (outOfBounds) finishRoiDrag(ev, true);
      }});

      overlay.addEventListener('pointerup', (ev) => finishRoiDrag(ev, true));
      overlay.addEventListener('pointercancel', (ev) => finishRoiDrag(ev, true));
      overlay.addEventListener('lostpointercapture', (ev) => finishRoiDrag(ev, true));
      overlay.addEventListener('pointerleave', (ev) => finishRoiDrag(ev, true));
    }}

    const savedRoi = loadSavedRoi();
    if (savedRoi && savedRoi.length === 4) {{
      currentRoi = savedRoi.slice();
    }}

    hook();
    renderAll(0);
    resizeOverlay();
    drawPeople(0);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Собрать оффлайн HTML-отчёт из out/v01 (видео + raw_frames + events).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", default="out/v01", type=str, help="Папка с output.mp4/events.csv/raw_frames.*")
    p.add_argument("--out", default="", type=str, help="Куда сохранить HTML (по умолчанию: <out-dir>/report.html)")
    p.add_argument("--overlap-min", default=0.0, type=float, help="Подсказка порога overlap_min (горизонтальная линия)")
    p.add_argument("--max-points", default=5000, type=int, help="Макс. точек, встраиваемых в графики (downsampling)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(str(args.out)).expanduser() if str(args.out).strip() else (out_dir / "report.html")
    payload = _prepare_payload(
        out_dir=out_dir,
        overlap_min_hint=float(args.overlap_min),
        max_points=int(args.max_points),
    )
    html = _build_html(payload)
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML report -> {out_path}")


if __name__ == "__main__":
    main()
