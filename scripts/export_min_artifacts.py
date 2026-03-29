from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


MINIMAL_FILES = ("raw_frames.pkl", "raw_people.jsonl", "report.txt")


@dataclass
class ExportedRun:
    rel_dir: str
    files: List[str]
    total_bytes: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a minimal publishable bundle without videos. "
        "By default it copies raw_frames.pkl, raw_people.jsonl, and report.txt."
    )
    p.add_argument("--out-root", default="out", type=str, help="Source run root (default: out)")
    p.add_argument(
        "--dest-root",
        default="publish/minimal",
        type=str,
        help="Destination root for the minimal bundle (default: publish/minimal)",
    )
    p.add_argument(
        "--with-html",
        action="store_true",
        help="Also rebuild per-run report.html and combined report.html in the destination root",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove destination root before export",
    )
    return p.parse_args()


def _find_run_dirs(out_root: Path) -> List[Path]:
    candidates = {p.parent for p in out_root.rglob("report.txt")} | {p.parent for p in out_root.rglob("raw_frames.pkl")}
    result = []
    for path in candidates:
        if path == out_root:
            continue
        rel_parts = path.relative_to(out_root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        result.append(path)

    preferred = {"video1": 0, "video2": 1, "video3": 2}
    return sorted(
        result,
        key=lambda p: (preferred.get(p.relative_to(out_root).parts[0], 999), p.relative_to(out_root).as_posix()),
    )


def _copy_minimal_files(run_dir: Path, dest_dir: Path, rel_dir: str) -> ExportedRun:
    copied: List[str] = []
    total_bytes = 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in MINIMAL_FILES:
        src = run_dir / name
        if not src.exists():
            continue
        dst = dest_dir / name
        shutil.copy2(src, dst)
        copied.append(name)
        total_bytes += src.stat().st_size
    return ExportedRun(rel_dir=rel_dir, files=copied, total_bytes=total_bytes)


def _fmt_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _iter_exported_runs(runs: Iterable[ExportedRun]) -> List[str]:
    lines: List[str] = []
    total = 0
    for run in runs:
        total += run.total_bytes
        files_str = ", ".join(run.files) if run.files else "no files copied"
        lines.append(f"- {run.rel_dir}: {_fmt_bytes(run.total_bytes)} ({files_str})")
    lines.append(f"- total: {_fmt_bytes(total)}")
    return lines


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_payload_json(text: str) -> tuple[int, int, dict] | tuple[None, None, None]:
    marker = "const PAYLOAD = "
    start = text.find(marker)
    if start < 0:
        return None, None, None
    start = text.find("{", start)
    if start < 0:
        return None, None, None

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
        return None, None, None

    try:
        payload = json.loads(text[start:end])
    except Exception:
        return None, None, None
    return start, end, payload


def _patch_copied_detail_report(*, repo_root: Path, run_dest: Path) -> None:
    html_path = run_dest / "report.html"
    text = _read_text(html_path)
    start, end, payload = _extract_payload_json(text)
    if payload is None:
        return

    meta = payload.get("meta") or {}
    source_video = str(meta.get("source_video", "")).strip()
    original_rel = ""
    if source_video:
        candidate = Path(source_video).expanduser()
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        if candidate.exists():
            original_rel = os.path.relpath(candidate, start=run_dest)

    rendered_path = run_dest / "output.mp4"
    rendered_rel = "output.mp4" if rendered_path.exists() else ""

    payload["video_src_rendered"] = rendered_rel
    payload["video_src_original"] = original_rel
    payload["video_src_original_local"] = original_rel
    payload["video_src"] = rendered_rel or original_rel
    payload["video_src_kind"] = "rendered" if rendered_rel else ("original" if original_rel else "")
    if isinstance(meta, dict):
        meta["out_dir"] = str(run_dest)
        payload["meta"] = meta

    new_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    patched = text[:start] + new_json + text[end:]
    html_path.write_text(patched, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    out_root = (repo_root / str(args.out_root)).resolve()
    dest_root = (repo_root / str(args.dest_root)).resolve()

    if not out_root.exists():
        raise SystemExit(f"[export] source root not found: {out_root}")

    if args.clean and dest_root.exists():
        shutil.rmtree(dest_root)

    dest_root.mkdir(parents=True, exist_ok=True)

    run_dirs = _find_run_dirs(out_root)
    exported_runs: List[ExportedRun] = []
    for run_dir in run_dirs:
        rel_dir = run_dir.relative_to(out_root)
        exported_runs.append(_copy_minimal_files(run_dir, dest_root / rel_dir, rel_dir.as_posix()))

    if args.with_html:
        for run in exported_runs:
            if not run.files:
                continue
            run_dest = dest_root / run.rel_dir
            run_src = out_root / run.rel_dir
            src_html = run_src / "report.html"
            if src_html.exists():
                shutil.copy2(src_html, run_dest / "report.html")
                _patch_copied_detail_report(repo_root=repo_root, run_dest=run_dest)
            else:
                subprocess.run(
                    [
                        sys.executable,
                        str(repo_root / "scripts" / "html_report_v01.py"),
                        "--out-dir",
                        str(run_dest),
                    ],
                    check=True,
                    cwd=repo_root,
                )

        subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "combined_report.py"),
                "--out-root",
                str(dest_root),
                "--out",
                str(dest_root / "report.html"),
            ],
            check=True,
            cwd=repo_root,
        )

    print(f"[export] destination: {dest_root}")
    for line in _iter_exported_runs(exported_runs):
        print(line)


if __name__ == "__main__":
    main()
