from __future__ import annotations

import argparse
import csv
import html
import os
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RunSummary:
    label: str
    rel_dir: str
    video: str
    detector_mode: str
    primary_detector: str
    roi_xyxy: str
    roi_xyxy_list: Optional[List[int]]
    fps: float
    event_counts: Dict[str, int]
    total_events: int
    last_event_sec: Optional[float]
    wait_pairs: Optional[int]
    wait_mean_sec: Optional[float]
    wait_median_sec: Optional[float]
    wait_p90_sec: Optional[float]
    touch_min_sec: Optional[float]
    touches: Optional[int]
    landings: Optional[int]
    mean_touch_sec: Optional[float]
    mean_gap_sec: Optional[float]
    report_excerpt: str
    output_mp4_rel: Optional[str]
    source_video_rel: Optional[str]
    detail_report_rel: Optional[str]
    events_csv_rel: Optional[str]
    report_txt_rel: Optional[str]
    overlay_frames: List[int]
    overlay_boxes: List[List[list]]
    overlay_roi_xyxy: Optional[List[int]]
    overlay_payload_json: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a single HTML index for all run outputs under out/.")
    p.add_argument("--out-root", default="out", type=str, help="Root output directory to scan")
    p.add_argument("--out", default="", type=str, help="Where to save combined HTML (default: <out-root>/report.html)")
    return p.parse_args()


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _find_run_dirs(out_root: Path) -> List[Path]:
    candidates = {p.parent for p in out_root.rglob("events.csv")} | {p.parent for p in out_root.rglob("report.txt")}
    result = []
    for path in candidates:
        if path == out_root:
            continue
        rel_parts = path.relative_to(out_root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if "previews" in rel_parts:
            continue
        result.append(path)

    preferred = {"video1": 0, "video2": 1, "video3": 2}
    return sorted(
        result,
        key=lambda p: (preferred.get(p.relative_to(out_root).parts[0], 999), p.relative_to(out_root).as_posix()),
    )


def _parse_report(path: Path) -> Dict[str, object]:
    text = _read_text(path)
    info: Dict[str, object] = {
        "video": "",
        "fps": 0.0,
        "detector_mode": "",
        "primary_detector": "",
        "roi_xyxy": "",
        "roi_xyxy_list": None,
        "wait_pairs": None,
        "wait_mean_sec": None,
        "wait_median_sec": None,
        "wait_p90_sec": None,
        "touch_min_sec": None,
        "touches": None,
        "landings": None,
        "mean_touch_sec": None,
        "mean_gap_sec": None,
        "excerpt": "",
    }
    if not text:
        return info

    excerpt_top_keys = {
        "video",
        "fps",
        "roi_xyxy",
        "detector_mode",
        "primary_detector",
    }
    excerpt_nested_keys = {
        "overlap_min",
        "t_enter_sec",
        "t_exit_sec",
        "t_approach_sec",
        "t_empty_min_sec",
        "infer_every",
        "perf_avg_detect_ms",
        "perf_est_fps",
        "perf_total_sec",
    }

    excerpt_lines: List[str] = []
    section = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            if len(excerpt_lines) < 24 and excerpt_lines and excerpt_lines[-1] != "":
                excerpt_lines.append("")
            continue

        if stripped == "wait_metric_empty_to_next_approach:":
            section = "wait"
            if len(excerpt_lines) < 24:
                excerpt_lines.append(stripped)
            continue

        if stripped == "touch_metrics:":
            section = "touch_metrics"
            if len(excerpt_lines) < 24:
                excerpt_lines.append(stripped)
            continue

        if stripped.startswith("wait_metric_empty_to_next_approach: insufficient"):
            section = ""
            if len(excerpt_lines) < 24:
                excerpt_lines.append(stripped)
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
            if len(excerpt_lines) < 24:
                excerpt_lines.append(stripped)
            continue

        if section == "touch_metrics" and raw.startswith("  ") and ":" in stripped:
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
            if len(excerpt_lines) < 24:
                excerpt_lines.append(stripped)
            continue

        if ":" in stripped and not raw.startswith("  "):
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            section = ""
            if key == "video":
                info["video"] = value
            elif key == "fps":
                try:
                    info["fps"] = float(value)
                except Exception:
                    pass
            elif key == "detector_mode":
                info["detector_mode"] = value
            elif key == "primary_detector":
                info["primary_detector"] = value
            elif key == "roi_xyxy":
                info["roi_xyxy"] = value
                try:
                    if value.startswith("(") and value.endswith(")"):
                        parts = [int(x.strip()) for x in value[1:-1].split(",")]
                        if len(parts) == 4:
                            info["roi_xyxy_list"] = parts
                except Exception:
                    pass
            if (key in excerpt_top_keys or stripped.startswith("Dodo table events report")) and len(excerpt_lines) < 24:
                excerpt_lines.append(stripped)
            continue

        if raw.startswith("  ") and ":" in stripped:
            key, _value = stripped.split(":", 1)
            if key in excerpt_nested_keys and len(excerpt_lines) < 24:
                excerpt_lines.append(stripped)
            continue

    info["excerpt"] = "\n".join(excerpt_lines).strip()
    return info


def _extract_events_from_detail_html(path: Path) -> Dict[str, object]:
    payload = _extract_detail_payload(path)
    if payload is None:
        return {"counts": {}, "total": 0, "last_ts": None}

    counts: Dict[str, int] = {}
    total = 0
    last_ts: Optional[float] = None
    for event in payload.get("events", []) or []:
        event_type = str(event.get("type", "")).strip() or "UNKNOWN"
        counts[event_type] = counts.get(event_type, 0) + 1
        total += 1
        try:
            ts = float(event.get("ts", ""))
            last_ts = ts if last_ts is None else max(last_ts, ts)
        except Exception:
            pass
    return {"counts": counts, "total": total, "last_ts": last_ts}


def _extract_detail_payload(path: Path) -> Optional[Dict[str, object]]:
    counts: Dict[str, int] = {}
    if not path.exists():
        return None

    text = _read_text(path)
    marker = "const PAYLOAD = "
    start = text.find(marker)
    if start < 0:
        return None
    start = text.find("{", start)
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None

    try:
        return json.loads(text[start:end])
    except Exception:
        return None


def _parse_events(path: Path, detail_report_path: Optional[Path] = None) -> Dict[str, object]:
    counts: Dict[str, int] = {}
    total = 0
    last_ts: Optional[float] = None
    if not path.exists():
        if detail_report_path is not None:
            return _extract_events_from_detail_html(detail_report_path)
        return {"counts": counts, "total": total, "last_ts": last_ts}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_type = str(row.get("event_type", "")).strip() or "UNKNOWN"
            counts[event_type] = counts.get(event_type, 0) + 1
            total += 1
            try:
                ts = float(row.get("ts_sec", ""))
                last_ts = ts if last_ts is None else max(last_ts, ts)
            except Exception:
                pass
    return {"counts": counts, "total": total, "last_ts": last_ts}


def _load_primary_overlay(run_dir: Path, primary_detector: str) -> Dict[str, object]:
    people_path = run_dir / "raw_people.jsonl"
    if not people_path.exists():
        return {"frames": [], "boxes": []}

    frames: List[int] = []
    boxes: List[List[list]] = []
    det_name = str(primary_detector or "").strip()
    try:
        with people_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                frames.append(int(obj.get("frame_idx", len(frames))))
                cur_boxes: List[list] = []
                if isinstance(obj.get("detectors"), dict):
                    det_map = obj.get("detectors") or {}
                    det_obj = det_map.get(det_name) if det_name else None
                    if isinstance(det_obj, dict):
                        cur_boxes = det_obj.get("people_used", []) or det_obj.get("people", []) or []
                else:
                    obj_det = str(obj.get("detector", "")).strip()
                    if not det_name or obj_det == det_name:
                        cur_boxes = obj.get("people", []) or []
                boxes.append(cur_boxes)
    except Exception:
        return {"frames": [], "boxes": []}

    return {"frames": frames, "boxes": boxes}


def _load_overlay_from_detail_report(path: Path) -> Dict[str, object]:
    payload = _extract_detail_payload(path)
    if payload is None:
        return {"frames": [], "boxes": [], "roi_xyxy": None, "payload_json": "{}"}

    raw = payload.get("raw") or {}
    meta = payload.get("meta") or {}
    source_video = str(meta.get("source_video", "") or "").strip()
    signal_frame_idx = raw.get("frame_idx") or []
    signal_ts = raw.get("ts") or []
    people_by_detector = raw.get("people_by_detector") or {}
    people_frames = raw.get("people_frames") or []
    primary = str(raw.get("primary_detector") or meta.get("primary_detector") or "").strip()
    boxes = []
    if isinstance(people_by_detector, dict) and primary and isinstance(people_by_detector.get(primary), list):
        boxes = people_by_detector.get(primary) or []
    elif isinstance(raw.get("people"), list):
        boxes = raw.get("people") or []

    roi_xyxy = None
    if isinstance(meta.get("roi_xyxy"), list) and len(meta.get("roi_xyxy")) >= 4:
        try:
            roi_xyxy = [int(v) for v in meta.get("roi_xyxy")[:4]]
        except Exception:
            roi_xyxy = None

    overlay_payload = {
        "frame_idx": signal_frame_idx if isinstance(signal_frame_idx, list) else [],
        "ts": signal_ts if isinstance(signal_ts, list) else [],
        "people_frames": people_frames if isinstance(people_frames, list) else [],
        "people": boxes if isinstance(boxes, list) else [],
        "roi_xyxy": roi_xyxy or [],
        "fps": float(meta.get("fps", 0.0) or 0.0),
        "primary_detector": primary or str(meta.get("primary_detector", "") or ""),
        "overlap_min": float(meta.get("overlap_min", 0.0) or 0.0),
        "fsm": meta.get("fsm") if isinstance(meta.get("fsm"), dict) else {},
        "analytics": meta.get("analytics") if isinstance(meta.get("analytics"), dict) else {},
        "source_video": source_video,
    }
    return {
        "frames": overlay_payload["frame_idx"],
        "boxes": overlay_payload["people"],
        "roi_xyxy": roi_xyxy,
        "payload_json": json.dumps(overlay_payload, ensure_ascii=False, separators=(",", ":")),
    }


def _make_summary(out_root: Path, run_dir: Path) -> RunSummary:
    report_meta = _parse_report(run_dir / "report.txt")
    event_meta = _parse_events(run_dir / "events.csv", run_dir / "report.html")
    detail_overlay_meta = _load_overlay_from_detail_report(run_dir / "report.html")
    overlay_meta = detail_overlay_meta
    if not detail_overlay_meta["frames"] and not detail_overlay_meta["boxes"]:
        overlay_meta = _load_primary_overlay(run_dir, str(report_meta.get("primary_detector", "")))
        detail_overlay_meta = {
            "frames": overlay_meta["frames"],
            "boxes": overlay_meta["boxes"],
            "roi_xyxy": report_meta.get("roi_xyxy_list"),
            "payload_json": json.dumps(
                {
                    "frame_idx": overlay_meta["frames"],
                    "ts": [],
                    "people_frames": overlay_meta["frames"],
                    "people": overlay_meta["boxes"],
                    "roi_xyxy": report_meta.get("roi_xyxy_list") or [],
                    "fps": float(report_meta.get("fps", 0.0) or 0.0),
                    "primary_detector": str(report_meta.get("primary_detector", "") or ""),
                    "overlap_min": 0.0,
                    "fsm": {},
                    "analytics": {
                        "wait_pairs": report_meta.get("wait_pairs"),
                        "wait_mean_sec": report_meta.get("wait_mean_sec"),
                        "wait_median_sec": report_meta.get("wait_median_sec"),
                        "wait_p90_sec": report_meta.get("wait_p90_sec"),
                        "touch_min_sec": report_meta.get("touch_min_sec"),
                        "touches": report_meta.get("touches"),
                        "landings": report_meta.get("landings"),
                        "mean_touch_sec": report_meta.get("mean_touch_sec"),
                        "mean_gap_sec": report_meta.get("mean_gap_sec"),
                    },
                    "source_video": str(report_meta.get("video", "") or ""),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    rel_dir = run_dir.relative_to(out_root).as_posix()

    def rel_if_exists(name: str) -> Optional[str]:
        path = run_dir / name
        if path.exists():
            return path.relative_to(out_root).as_posix()
        return None

    def rel_video_from_report() -> Optional[str]:
        video_value = str(report_meta.get("video", "")).strip()
        if not video_value:
            return None
        candidate = Path(video_value).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.exists():
            return None
        try:
            return os.path.relpath(candidate, start=out_root)
        except Exception:
            return str(candidate)

    return RunSummary(
        label=rel_dir,
        rel_dir=rel_dir,
        video=str(report_meta.get("video", "")),
        detector_mode=str(report_meta.get("detector_mode", "")),
        primary_detector=str(report_meta.get("primary_detector", "")),
        roi_xyxy=str(report_meta.get("roi_xyxy", "")),
        roi_xyxy_list=report_meta.get("roi_xyxy_list"),
        fps=float(report_meta.get("fps", 0.0) or 0.0),
        event_counts=dict(event_meta["counts"]),
        total_events=int(event_meta["total"]),
        last_event_sec=event_meta["last_ts"],
        wait_pairs=report_meta["wait_pairs"],
        wait_mean_sec=report_meta["wait_mean_sec"],
        wait_median_sec=report_meta["wait_median_sec"],
        wait_p90_sec=report_meta["wait_p90_sec"],
        touch_min_sec=report_meta["touch_min_sec"],
        touches=report_meta["touches"],
        landings=report_meta["landings"],
        mean_touch_sec=report_meta["mean_touch_sec"],
        mean_gap_sec=report_meta["mean_gap_sec"],
        report_excerpt=str(report_meta.get("excerpt", "")),
        output_mp4_rel=rel_if_exists("output.mp4"),
        source_video_rel=rel_video_from_report(),
        detail_report_rel=rel_if_exists("report.html"),
        events_csv_rel=rel_if_exists("events.csv"),
        report_txt_rel=rel_if_exists("report.txt"),
        overlay_frames=list(overlay_meta["frames"]),
        overlay_boxes=list(overlay_meta["boxes"]),
        overlay_roi_xyxy=detail_overlay_meta["roi_xyxy"],
        overlay_payload_json=str(detail_overlay_meta["payload_json"]),
    )


def _fmt_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def _build_html(out_root: Path, runs: List[RunSummary]) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nav_items = "\n".join(
        f'<a class="pill" href="#run-{html.escape(run.rel_dir).replace("/", "-")}">{html.escape(run.label)}</a>'
        for run in runs
    )

    sections: List[str] = []
    for run in runs:
        anchor = html.escape(run.rel_dir).replace("/", "-")
        video_label = Path(run.video).stem if run.video else run.label
        counts_markup = "".join(
            f'<span class="metric"><strong>{html.escape(name)}</strong> {count}</span>'
            for name, count in sorted(run.event_counts.items())
        )
        event_summary_items = []
        for short, key in (("APP", "APPROACH"), ("EMP", "EMPTY"), ("OCC", "OCCUPIED")):
            count = int(run.event_counts.get(key, 0))
            event_summary_items.append(
                f'<div class="summary-stat"><div class="summary-stat-label">{short}</div><div class="summary-stat-value">{count}</div></div>'
            )
        event_summary_markup = "".join(event_summary_items)
        wait_markup = (
            f'<div class="analysis-card analysis-primary">'
            f'<div class="analysis-kicker">Главная метрика ТЗ</div>'
            f'<div class="analysis-title">Ожидание до следующего подхода</div>'
            f'<div class="analysis-value">mean={_fmt_seconds(run.wait_mean_sec)}</div>'
            f'<div class="analysis-meta">pairs={run.wait_pairs} · median={_fmt_seconds(run.wait_median_sec)} · p90={_fmt_seconds(run.wait_p90_sec)}</div>'
            f'</div>'
            if run.wait_pairs is not None
            else '<div class="analysis-card analysis-primary analysis-empty"><div class="analysis-kicker">Главная метрика ТЗ</div><div class="analysis-title">Ожидание до следующего подхода</div><div class="analysis-value">недостаточно событий</div><div class="analysis-meta">Нет валидной пары EMPTY -> APPROACH.</div></div>'
        )
        touch_cards: List[str] = []
        if run.touches is not None:
            label = "Касания"
            if run.touch_min_sec is not None:
                label += f" >= {run.touch_min_sec:.1f}s"
            touch_cards.append(
                f'<div class="mini-metric"><div class="mini-label">{label}</div><div class="mini-value">{int(run.touches)}</div></div>'
            )
        if run.mean_touch_sec is not None:
            touch_cards.append(
                f'<div class="mini-metric"><div class="mini-label">Ср. длительность</div><div class="mini-value">{_fmt_seconds(run.mean_touch_sec)}</div></div>'
            )
        if run.mean_gap_sec is not None:
            touch_cards.append(
                f'<div class="mini-metric"><div class="mini-label">Ср. интервал</div><div class="mini-value">{_fmt_seconds(run.mean_gap_sec)}</div></div>'
            )
        if run.landings is not None:
            touch_cards.append(
                f'<div class="mini-metric"><div class="mini-label">Посадки</div><div class="mini-value">{int(run.landings)}</div></div>'
            )
        touch_markup = (
            f'<div class="analysis-card analysis-secondary">'
            f'<div class="analysis-kicker">Поведение перед посадкой</div>'
            f'<div class="analysis-title">Касания и посадки</div>'
            f'<div class="mini-metric-grid">{"".join(touch_cards)}</div>'
            f'</div>'
            if touch_cards
            else '<div class="analysis-card analysis-secondary analysis-empty"><div class="analysis-kicker">Поведение перед посадкой</div><div class="analysis-title">Касания и посадки</div><div class="analysis-meta">n/a</div></div>'
        )
        playable_video_rel = run.output_mp4_rel or run.source_video_rel
        detail_link = (
            f'<a class="detail-link" href="{html.escape(run.detail_report_rel)}">open report / edit ROI</a>'
            if run.detail_report_rel
            else ""
        )
        source_controls = ""
        if run.output_mp4_rel and run.source_video_rel:
            source_controls = """
      <div class="video-source-group">
        <span class="video-source-label">source:</span>
        <button class="source-btn active" type="button" data-kind="rendered">rendered</button>
        <button class="source-btn" type="button" data-kind="original">original</button>
        {detail_link}
      </div>
""".rstrip().format(detail_link=detail_link)
        elif run.output_mp4_rel:
            source_controls = """
      <div class="video-source-group">
        <span class="video-source-label">source:</span>
        <span class="source-pill">rendered</span>
        {detail_link}
      </div>
""".rstrip().format(detail_link=detail_link)
        elif run.source_video_rel:
            source_controls = """
      <div class="video-source-group">
        <span class="video-source-label">source:</span>
        <span class="source-pill">original</span>
        {detail_link}
      </div>
""".rstrip().format(detail_link=detail_link)
        video_markup = (
            (
                f'{source_controls}\n'
                f'<div class="video-stage" data-overlay=\'{html.escape(run.overlay_payload_json)}\'>'
                f'<video controls preload="metadata" '
                f'data-rendered="{html.escape(run.output_mp4_rel or "")}" '
                f'data-original="{html.escape(run.source_video_rel or "")}" '
                f'data-kind="{"rendered" if run.output_mp4_rel else "original"}" '
                f'src="{html.escape(playable_video_rel)}"></video>'
                f'<canvas class="video-overlay"></canvas>'
                f'</div>'
            )
            if playable_video_rel
            else '<div class="video-missing">output.mp4 and original video not found</div>'
        )
        sections.append(
            f"""
<section class="run-card" id="run-{anchor}">
    <div class="run-head">
    <div>
      <h2>{html.escape(run.label)}</h2>
      <div class="meta">video: <code>{html.escape(run.video or "n/a")}</code></div>
      <div class="meta">
        detector: <code>{html.escape(run.detector_mode or "n/a")}</code>
        · primary: <code>{html.escape(run.primary_detector or "n/a")}</code>
      </div>
      <div class="meta">roi: <code>{html.escape(run.roi_xyxy or "n/a")}</code></div>
    </div>
    <div class="summary-box">
      <div class="summary-grid">{event_summary_markup}</div>
      <div class="summary-note">last event: {_fmt_seconds(run.last_event_sec)}</div>
    </div>
  </div>
  <div class="run-body">
  <div class="video-panel">
    {video_markup}
  </div>
    <div class="info-panel">
      <div class="metrics">{counts_markup or '<span class="muted">no events.csv data</span>'}</div>
      <div class="analysis-stack">
        <div class="wait-metric-wrap">{wait_markup}</div>
        <div class="touch-metric-wrap">{touch_markup}</div>
      </div>
      <details>
        <summary>report excerpt</summary>
        <pre>{html.escape(run.report_excerpt or "report.txt not found")}</pre>
      </details>
    </div>
  </div>
</section>
""".strip()
        )

    empty_state = """
<section class="empty-state">
  <h2>Пока нет прогонов</h2>
  <p>
    Запусти, например, <code>./scripts/run.sh --preset video1</code>
    или <code>./scripts/run.sh --preset all</code>.
  </p>
</section>
""".strip()

    body_content = "\n".join(sections) if sections else empty_state
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dodo Table Events Report</title>
  <style>
    :root {{
      --bg: #f6efe7;
      --panel: #fffaf4;
      --panel-strong: #fff;
      --text: #1f1a17;
      --muted: #6e6258;
      --accent: #cc5a2f;
      --accent-soft: #f5d0c3;
      --border: #e5d7ca;
      --shadow: 0 18px 45px rgba(70, 44, 29, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, #fff8ee 0, transparent 34%),
        linear-gradient(180deg, #f7f1ea 0%, #efe4d8 100%);
    }}
    .page {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px 24px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(204, 90, 47, 0.12), rgba(255, 255, 255, 0.92));
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 24px 26px;
      box-shadow: var(--shadow);
    }}
    h1, h2, p {{ margin: 0; }}
    .hero h1 {{
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1;
      letter-spacing: -0.03em;
      margin-bottom: 10px;
    }}
    .hero p {{
      color: var(--muted);
      max-width: 760px;
      line-height: 1.5;
    }}
    .hero-meta {{
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
    }}
    .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 18px 0 26px;
    }}
    .pill {{
      text-decoration: none;
      color: var(--text);
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 10px 14px;
    }}
    .run-card, .empty-state {{
      background: rgba(255, 250, 244, 0.92);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 20px;
      box-shadow: var(--shadow);
      margin-bottom: 18px;
    }}
    .run-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 18px;
    }}
    .run-head h2 {{
      font-size: 28px;
      margin-bottom: 8px;
    }}
    .meta {{
      color: var(--muted);
      line-height: 1.5;
      margin-top: 2px;
    }}
    .summary-box {{
      min-width: 220px;
      background: var(--panel-strong);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px 16px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .summary-stat {{
      background: rgba(245, 208, 195, 0.32);
      border: 1px solid rgba(204, 90, 47, 0.12);
      border-radius: 14px;
      padding: 10px 8px;
      text-align: center;
    }}
    .summary-stat-label {{
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .summary-stat-value {{
      margin-top: 4px;
      font-size: 24px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }}
    .summary-note {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }}
    .run-body {{
      display: grid;
      grid-template-columns: minmax(320px, 1.35fr) minmax(280px, 0.9fr);
      gap: 18px;
    }}
    .video-panel, .info-panel {{
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 14px;
    }}
    .video-stage {{
      position: relative;
    }}
    video, .video-missing {{
      width: 100%;
      border-radius: 14px;
      background: #120f0d;
      min-height: 220px;
    }}
    .video-overlay {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      border-radius: 14px;
      background: transparent;
    }}
    .video-missing {{
      color: #fff;
      display: grid;
      place-items: center;
    }}
    .links {{
      margin-top: 10px;
      color: var(--muted);
    }}
    .video-source-group {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .video-source-label {{
      color: var(--muted);
      font-size: 13px;
    }}
    .source-btn, .source-pill {{
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.86);
      color: var(--text);
    }}
    .detail-link {{
      margin-left: auto;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.86);
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }}
    .source-btn {{
      cursor: pointer;
    }}
    .source-btn.active {{
      border-color: var(--accent);
      background: var(--accent-soft);
      color: #6f2a11;
    }}
    .links a {{
      color: var(--accent);
    }}
    .metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .metric {{
      background: var(--accent-soft);
      border-radius: 999px;
      padding: 8px 10px;
    }}
    .analysis-stack {{
      display: grid;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .analysis-card {{
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(250,244,238,0.92));
    }}
    .analysis-primary {{
      border-color: rgba(204, 90, 47, 0.28);
      background: linear-gradient(180deg, rgba(255,247,242,0.98), rgba(247,231,221,0.92));
      box-shadow: inset 0 0 0 1px rgba(204, 90, 47, 0.05);
    }}
    .analysis-secondary {{
      background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(245,240,234,0.94));
    }}
    .analysis-empty .analysis-value,
    .analysis-empty .analysis-meta {{
      color: var(--muted);
    }}
    .analysis-kicker {{
      color: var(--accent);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    .analysis-title {{
      font-size: 16px;
      font-weight: 700;
      line-height: 1.2;
    }}
    .analysis-value {{
      margin-top: 8px;
      font-size: 26px;
      font-weight: 800;
      letter-spacing: -0.03em;
    }}
    .analysis-meta {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }}
    .mini-metric-grid {{
      margin-top: 10px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .mini-metric {{
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.72);
    }}
    .analysis-secondary .mini-metric {{
      background: rgba(255, 250, 244, 0.92);
    }}
    .mini-label {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.3;
    }}
    .mini-value {{
      margin-top: 6px;
      font-size: 22px;
      font-weight: 800;
      letter-spacing: -0.02em;
    }}
    .muted {{
      color: var(--muted);
    }}
    details {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 10px 12px;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    pre {{
      margin: 10px 0 0;
      white-space: pre-wrap;
      font-size: 13px;
      line-height: 1.45;
      color: var(--muted);
    }}
    .detail-embed {{
      margin-top: 16px;
      background: rgba(255, 255, 255, 0.78);
    }}
    .detail-embed iframe {{
      width: 100%;
      min-height: 1080px;
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-top: 12px;
      background: #fff;
    }}
    code {{
      font-family: "SFMono-Regular", Menlo, monospace;
      font-size: 0.95em;
    }}
    @media (max-width: 960px) {{
      .run-body {{
        grid-template-columns: 1fr;
      }}
      .run-head {{
        flex-direction: column;
      }}
      .summary-box {{
        text-align: left;
      }}
      .mini-metric-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>Dodo Table Events</h1>
      <p>
        Единая оффлайн-страница по всем найденным прогонам в
        <code>{html.escape(str(out_root))}</code>. Здесь можно открыть видео,
        быстро сверить метрики и при желании перейти в детальный HTML
        конкретного запуска.
      </p>
      <div class="hero-meta">
        <span>runs: <strong>{len(runs)}</strong></span>
        <span>generated: <strong>{html.escape(generated_at)}</strong></span>
        <span>detail html: <strong>{"enabled" if runs else "not found yet"}</strong></span>
      </div>
    </section>
    <nav class="nav">
      {nav_items or '<span class="muted">Появится после первого прогона.</span>'}
    </nav>
    {body_content}
  </main>
  <script>
    function fmtSeconds(value) {{
      const num = Number(value);
      return Number.isFinite(num) ? `${{num.toFixed(2)}}s` : 'n/a';
    }}

    function intersectionRatio(box, roi) {{
      if (!Array.isArray(box) || box.length < 4 || !Array.isArray(roi) || roi.length !== 4) return 0;
      const x1 = Math.max(+box[0] || 0, +roi[0] || 0);
      const y1 = Math.max(+box[1] || 0, +roi[1] || 0);
      const x2 = Math.min(+box[2] || 0, +roi[2] || 0);
      const y2 = Math.min(+box[3] || 0, +roi[3] || 0);
      const iw = Math.max(0, x2 - x1);
      const ih = Math.max(0, y2 - y1);
      const roiArea = Math.max(1e-9, ((+roi[2] || 0) - (+roi[0] || 0)) * ((+roi[3] || 0) - (+roi[1] || 0)));
      return (iw * ih) / roiArea;
    }}

    function binarySearchFloor(arr, target) {{
      let lo = 0, hi = arr.length - 1, ans = 0;
      while (lo <= hi) {{
        const mid = (lo + hi) >> 1;
        if ((+arr[mid] || 0) <= target) {{ ans = mid; lo = mid + 1; }} else {{ hi = mid - 1; }}
      }}
      return ans;
    }}

    function computeSeriesForOverlay(overlayData, roi) {{
      const ts = Array.isArray(overlayData.ts) ? overlayData.ts.map((v) => +v || 0) : [];
      const signalFrames = Array.isArray(overlayData.frame_idx) ? overlayData.frame_idx.map((v) => Math.round(+v || 0)) : [];
      const inferFrames = Array.isArray(overlayData.people_frames) ? overlayData.people_frames.map((v) => Math.round(+v || 0)) : [];
      const people = Array.isArray(overlayData.people) ? overlayData.people : [];
      const fps = +overlayData.fps || 0;
      const thr = +overlayData.overlap_min || 0;
      const n = ts.length || signalFrames.length;
      const out = {{
        ts: ts.length ? ts : signalFrames.map((f) => fps > 0 ? f / fps : 0),
        in_zone_now: [],
      }};
      if (!n || !inferFrames.length || !people.length) return out;
      for (let i = 0; i < n; i += 1) {{
        const targetFrame = signalFrames.length ? signalFrames[i] : (fps > 0 ? Math.round((out.ts[i] || 0) * fps) : i);
        const idx = binarySearchFloor(inferFrames, targetFrame);
        const boxes = Array.isArray(people[idx]) ? people[idx] : [];
        let bestOverlap = 0;
        boxes.forEach((box) => {{
          bestOverlap = Math.max(bestOverlap, intersectionRatio(box, roi));
        }});
        out.in_zone_now.push(bestOverlap >= thr ? 1 : 0);
      }}
      return out;
    }}

    function simulateEvents(ts, inzone, overlayData) {{
      const fsm = (overlayData && typeof overlayData.fsm === 'object' && overlayData.fsm) ? overlayData.fsm : {{}};
      const tEnter = +(fsm.t_enter_sec || 0);
      const tExit = +(fsm.t_exit_sec || 0);
      const tApproach = +(fsm.t_approach_sec || 0);
      const tEmptyMin = +(fsm.t_empty_min_sec || 0);
      const frameIdx = Array.isArray(overlayData.frame_idx) ? overlayData.frame_idx : [];
      const events = [];
      if (!ts.length) return events;

      let state = inzone[0] ? 'OCCUPIED' : 'EMPTY';
      let enterHold = 0.0;
      let exitHold = 0.0;
      let emptyStartTs = +ts[0] || 0;
      let approachEmitted = false;
      let approachHold = 0.0;

      events.push({{ ts_sec: +ts[0] || 0, frame_idx: frameIdx.length ? Math.round(+frameIdx[0] || 0) : 0, event_type: state }});

      for (let i = 1; i < ts.length; i += 1) {{
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
                  events.push({{ ts_sec: t, frame_idx: frameIdx.length ? Math.round(+frameIdx[i] || 0) : i, event_type: 'APPROACH' }});
                  approachEmitted = true;
                }}
              }}
            }}
            enterHold += dt;
            if (enterHold >= tEnter) {{
              state = 'OCCUPIED';
              exitHold = 0.0;
              enterHold = 0.0;
              events.push({{ ts_sec: t, frame_idx: frameIdx.length ? Math.round(+frameIdx[i] || 0) : i, event_type: 'OCCUPIED' }});
            }}
          }}
        }} else if (inNow) {{
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
            events.push({{ ts_sec: t, frame_idx: frameIdx.length ? Math.round(+frameIdx[i] || 0) : i, event_type: 'EMPTY' }});
          }}
        }}
      }}
      return events;
    }}

    function summarizeWait(events) {{
      let seenOccupied = false;
      const empties = [];
      const approaches = [];
      (events || []).forEach((ev) => {{
        const type = String((ev && ev.event_type) || '');
        const ts = +(ev && ev.ts_sec || 0);
        if (type === 'OCCUPIED') seenOccupied = true;
        else if (type === 'EMPTY' && seenOccupied) empties.push(ts);
        else if (type === 'APPROACH') approaches.push(ts);
      }});
      const waits = [];
      let j = 0;
      empties.forEach((tEmpty) => {{
        while (j < approaches.length && !(approaches[j] >= tEmpty + 1e-9)) j += 1;
        if (j < approaches.length) waits.push(approaches[j] - tEmpty);
      }});
      if (!waits.length) return {{ available: false }};
      const arr = waits.slice().sort((a, b) => a - b);
      const mean = arr.reduce((s, v) => s + v, 0) / arr.length;
      const median = arr.length % 2 ? arr[(arr.length - 1) >> 1] : (arr[arr.length / 2 - 1] + arr[arr.length / 2]) / 2;
      const quantile = (p) => {{
        if (arr.length === 1) return arr[0];
        const pos = (arr.length - 1) * p;
        const lo = Math.floor(pos);
        const hi = Math.ceil(pos);
        if (lo === hi) return arr[lo];
        const t = pos - lo;
        return arr[lo] * (1 - t) + arr[hi] * t;
      }};
      return {{ available: true, pairs: arr.length, mean_sec: mean, median_sec: median, p90_sec: quantile(0.9) }};
    }}

    function summarizeTouches(ts, inzone, overlayData) {{
      const fsm = (overlayData && typeof overlayData.fsm === 'object' && overlayData.fsm) ? overlayData.fsm : {{}};
      const minTouchSec = +(fsm.t_approach_sec || 0);
      if (!ts.length) return {{ min_touch_sec: minTouchSec, touches: 0, landings: 0, mean_touch_sec: null, mean_gap_sec: null }};
      const state01 = [];
      let state = inzone[0] ? 'OCCUPIED' : 'EMPTY';
      let enterHold = 0.0;
      let exitHold = 0.0;
      state01.push(state === 'OCCUPIED' ? 1 : 0);
      for (let i = 1; i < ts.length; i += 1) {{
        const dt = Math.max(0, (+ts[i] || 0) - (+ts[i - 1] || 0));
        const inNow = !!inzone[i];
        if (state === 'EMPTY') {{
          if (!inNow) enterHold = 0.0;
          else {{
            enterHold += dt;
            if (enterHold >= (+(fsm.t_enter_sec || 0))) {{
              state = 'OCCUPIED';
              exitHold = 0.0;
              enterHold = 0.0;
            }}
          }}
        }} else if (inNow) {{
          exitHold = 0.0;
        }} else {{
          exitHold += dt;
          if (exitHold >= (+(fsm.t_exit_sec || 0))) {{
            state = 'EMPTY';
            exitHold = 0.0;
            enterHold = 0.0;
          }}
        }}
        state01.push(state === 'OCCUPIED' ? 1 : 0);
      }}
      const bounced = [];
      let landings = 0;
      let active = false;
      let touchStartTs = 0.0;
      for (let i = 0; i < ts.length; i += 1) {{
        const t = +ts[i] || 0;
        const inNow = !!inzone[i];
        const st = state01[i] ? 'OCCUPIED' : 'EMPTY';
        if (!active) {{
          if (st === 'EMPTY' && inNow) {{ active = true; touchStartTs = t; }}
          continue;
        }}
        if (st === 'OCCUPIED' && inNow) {{
          const dur = Math.max(0, t - touchStartTs);
          if (dur >= minTouchSec) landings += 1;
          active = false;
          continue;
        }}
        if ((st === 'EMPTY' && !inNow) || (st !== 'EMPTY' && st !== 'OCCUPIED')) {{
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
      const durations = bounced.map((pair) => Math.max(0, pair[1] - pair[0]));
      const gaps = [];
      for (let i = 1; i < bounced.length; i += 1) gaps.push(Math.max(0, bounced[i][0] - bounced[i - 1][1]));
      const avg = (arr) => arr.length ? (arr.reduce((s, v) => s + v, 0) / arr.length) : null;
      return {{
        min_touch_sec: minTouchSec,
        touches: bounced.length,
        landings,
        mean_touch_sec: avg(durations),
        mean_gap_sec: avg(gaps),
      }};
    }}

    function waitMarkup(analytics) {{
      if (analytics && analytics.wait_available && analytics.wait_mean_sec != null) {{
        return `<div class="analysis-card analysis-primary"><div class="analysis-kicker">Главная метрика ТЗ</div><div class="analysis-title">Ожидание до следующего подхода</div><div class="analysis-value">mean=${{fmtSeconds(analytics.wait_mean_sec)}}</div><div class="analysis-meta">pairs=${{analytics.wait_pairs}} · median=${{fmtSeconds(analytics.wait_median_sec)}} · p90=${{fmtSeconds(analytics.wait_p90_sec)}}</div></div>`;
      }}
      return `<div class="analysis-card analysis-primary analysis-empty"><div class="analysis-kicker">Главная метрика ТЗ</div><div class="analysis-title">Ожидание до следующего подхода</div><div class="analysis-value">недостаточно событий</div><div class="analysis-meta">Нет валидной пары EMPTY -> APPROACH.</div></div>`;
    }}

    function touchMarkup(analytics) {{
      const cards = [];
      if (analytics && analytics.touches != null) {{
        let label = 'Касания';
        if (analytics.touch_min_sec != null) label += ` >= ${{Number(analytics.touch_min_sec).toFixed(1)}}s`;
        cards.push(`<div class="mini-metric"><div class="mini-label">${{label}}</div><div class="mini-value">${{Number(analytics.touches || 0)}}</div></div>`);
      }}
      if (analytics && analytics.mean_touch_sec != null) cards.push(`<div class="mini-metric"><div class="mini-label">Ср. длительность</div><div class="mini-value">${{fmtSeconds(analytics.mean_touch_sec)}}</div></div>`);
      if (analytics && analytics.mean_gap_sec != null) cards.push(`<div class="mini-metric"><div class="mini-label">Ср. интервал</div><div class="mini-value">${{fmtSeconds(analytics.mean_gap_sec)}}</div></div>`);
      if (analytics && analytics.landings != null) cards.push(`<div class="mini-metric"><div class="mini-label">Посадки</div><div class="mini-value">${{Number(analytics.landings || 0)}}</div></div>`);
      if (!cards.length) return `<div class="analysis-card analysis-secondary analysis-empty"><div class="analysis-kicker">Поведение перед посадкой</div><div class="analysis-title">Касания и посадки</div><div class="analysis-meta">n/a</div></div>`;
      return `<div class="analysis-card analysis-secondary"><div class="analysis-kicker">Поведение перед посадкой</div><div class="analysis-title">Касания и посадки</div><div class="mini-metric-grid">${{cards.join('')}}</div></div>`;
    }}

    document.querySelectorAll('.video-panel').forEach((panel) => {{
      const stage = panel.querySelector('.video-stage');
      const runCard = panel.closest('.run-card');
      const video = panel.querySelector('video');
      if (!video) return;
      const canvas = panel.querySelector('.video-overlay');
      const waitWrap = runCard ? runCard.querySelector('.wait-metric-wrap') : null;
      const touchWrap = runCard ? runCard.querySelector('.touch-metric-wrap') : null;
      let overlayData = null;
      try {{
        overlayData = stage ? JSON.parse(stage.dataset.overlay || '{{}}') : null;
      }} catch (_err) {{
        overlayData = null;
      }}
      const roiStorageId = (overlayData && String(overlayData.source_video || '').trim())
        || (overlayData && Array.isArray(overlayData.roi_xyxy) ? overlayData.roi_xyxy.join(',') : '')
        || video.currentSrc
        || video.dataset.original
        || video.dataset.rendered
        || '';
      const roiStorageKey = `dodo-table-events::roi::${{roiStorageId}}`;
      const rendered = video.dataset.rendered || '';
      const original = video.dataset.original || '';
      const buttons = panel.querySelectorAll('.source-btn');
      const renderMetrics = () => {{
        if (!overlayData || !waitWrap || !touchWrap) return;
        let roi = Array.isArray(overlayData.roi_xyxy) ? overlayData.roi_xyxy.slice() : [];
        try {{
          if (window.localStorage) {{
            const rawSavedRoi = window.localStorage.getItem(roiStorageKey);
            if (rawSavedRoi) {{
              const savedRoi = JSON.parse(rawSavedRoi);
              if (Array.isArray(savedRoi) && savedRoi.length === 4) {{
                roi = savedRoi.map((v) => +v || 0);
              }}
            }}
          }}
        }} catch (_err) {{
        }}
        const analyticsBase = (overlayData && typeof overlayData.analytics === 'object' && overlayData.analytics) ? overlayData.analytics : null;
        const roiKeyBase = Array.isArray(overlayData.roi_xyxy) ? overlayData.roi_xyxy.join(',') : '';
        const roiKeyCurrent = Array.isArray(roi) ? roi.join(',') : '';
        let analytics = analyticsBase;
        if (roiKeyCurrent !== roiKeyBase && Array.isArray(roi) && roi.length === 4) {{
          const series = computeSeriesForOverlay(overlayData, roi);
          const events = simulateEvents(series.ts || [], series.in_zone_now || [], overlayData);
          const wait = summarizeWait(events);
          const touch = summarizeTouches(series.ts || [], series.in_zone_now || [], overlayData);
          analytics = {{
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
        waitWrap.innerHTML = waitMarkup(analytics || {{}});
        touchWrap.innerHTML = touchMarkup(analytics || {{}});
      }};
      const drawOverlay = () => {{
        if (!canvas || !overlayData) return;
        const rect = video.getBoundingClientRect();
        const cssW = Math.max(1, Math.round(rect.width));
        const cssH = Math.max(1, Math.round(rect.height));
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(cssW * dpr));
        canvas.height = Math.max(1, Math.round(cssH * dpr));
        canvas.style.width = `${{cssW}}px`;
        canvas.style.height = `${{cssH}}px`;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, cssW, cssH);

        const vw = video.videoWidth || 0;
        const vh = video.videoHeight || 0;
        if (!vw || !vh) return;
        const scaleX = cssW / vw;
        const scaleY = cssH / vh;

        let roi = Array.isArray(overlayData.roi_xyxy) ? overlayData.roi_xyxy.slice() : [];
        try {{
          if (window.localStorage) {{
            const rawSavedRoi = window.localStorage.getItem(roiStorageKey);
            if (rawSavedRoi) {{
              const savedRoi = JSON.parse(rawSavedRoi);
              if (Array.isArray(savedRoi) && savedRoi.length === 4) {{
                roi = savedRoi.map((v) => +v || 0);
              }}
            }}
          }}
        }} catch (_err) {{
        }}
        if (roi.length === 4) {{
          const [x1, y1, x2, y2] = roi.map((v) => +v || 0);
          ctx.strokeStyle = 'rgba(255,77,79,0.95)';
          ctx.fillStyle = 'rgba(255,77,79,0.12)';
          ctx.lineWidth = Math.max(2, cssW / 900);
          ctx.fillRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
          ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
        }}

        const frames = Array.isArray(overlayData.frame_idx) ? overlayData.frame_idx : [];
        const people = Array.isArray(overlayData.people) ? overlayData.people : [];
        if (!frames.length || !people.length) return;
        const fps = +overlayData.fps || 0;
        const frameNow = fps > 0 ? Math.round((video.currentTime || 0) * fps) : 0;
        let bestIdx = 0;
        let bestDist = Number.POSITIVE_INFINITY;
        for (let i = 0; i < frames.length; i += 1) {{
          const dist = Math.abs((+frames[i] || 0) - frameNow);
          if (dist < bestDist) {{
            bestDist = dist;
            bestIdx = i;
          }}
        }}
        const boxes = Array.isArray(people[bestIdx]) ? people[bestIdx] : [];
        ctx.strokeStyle = 'rgba(70,130,255,0.95)';
        ctx.lineWidth = Math.max(2, cssW / 1000);
        boxes.forEach((box) => {{
          if (!Array.isArray(box) || box.length < 4) return;
          const [x1, y1, x2, y2] = box;
          ctx.strokeRect(
            (+x1 || 0) * scaleX,
            (+y1 || 0) * scaleY,
            ((+x2 || 0) - (+x1 || 0)) * scaleX,
            ((+y2 || 0) - (+y1 || 0)) * scaleY,
          );
        }});
      }};
      const setKind = (kind) => {{
        const src = kind === 'original' ? original : rendered;
        if (!src) return;
        const keepTime = video.currentTime || 0;
        const wasPaused = video.paused;
        video.dataset.kind = kind;
        buttons.forEach((btn) => btn.classList.toggle('active', btn.dataset.kind === kind));
        video.src = src;
        video.addEventListener('loadedmetadata', () => {{
          const t = Math.max(0, Math.min(keepTime, video.duration || keepTime || 0));
          video.currentTime = t;
          drawOverlay();
          if (!wasPaused) video.play().catch(() => {{}});
        }}, {{ once: true }});
      }};
      buttons.forEach((btn) => {{
        btn.addEventListener('click', () => setKind(btn.dataset.kind || 'rendered'));
      }});
      ['loadedmetadata', 'timeupdate', 'seeked', 'play', 'pause'].forEach((name) => {{
        video.addEventListener(name, drawOverlay);
      }});
      window.addEventListener('resize', drawOverlay);
      window.addEventListener('storage', (ev) => {{
        if (ev.key && ev.key === roiStorageKey) {{
          drawOverlay();
          renderMetrics();
        }}
      }});
      drawOverlay();
      renderMetrics();
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    out_root = Path(str(args.out_root)).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = Path(str(args.out)).expanduser() if str(args.out).strip() else (out_root / "report.html")
    run_dirs = _find_run_dirs(out_root)
    runs = [_make_summary(out_root, run_dir) for run_dir in run_dirs]
    html_text = _build_html(out_root, runs)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"Combined HTML report -> {out_path}")


if __name__ == "__main__":
    main()
