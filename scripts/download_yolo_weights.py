from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

# Стабильный вес YOLOv8n, выложенный на Hugging Face.
# Этот файл совместим со старыми версиями Ultralytics, которые ожидают `ultralytics.nn.modules` как модуль.
YOLOV8N_URL = "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt"
YOLOV8N_SHA256 = "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dodo-table-events/0.1"})
    with urllib.request.urlopen(req) as resp:  # nosec - URL фиксирован в коде
        total = int(resp.headers.get("Content-Length") or 0)
        written = 0
        with tmp.open("wb") as f:
            while True:
                buf = resp.read(1024 * 256)
                if not buf:
                    break
                f.write(buf)
                written += len(buf)
                if total > 0:
                    pct = (100.0 * written) / float(total)
                    sys.stdout.write(f"\rСкачивание {dst.name}: {pct:5.1f}% ({written / 1e6:.1f}/{total / 1e6:.1f} MB)")
                    sys.stdout.flush()
    if total > 0:
        sys.stdout.write("\n")
    os.replace(str(tmp), str(dst))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Скачать веса YOLOv8 в models/ (для legacy Ultralytics на Python 3.8).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dst", default="models/yolov8n.pt", type=str, help="Куда сохранить yolov8n.pt")
    p.add_argument("--force", action="store_true", help="Скачать заново, даже если файл уже есть и sha совпадает")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dst = Path(str(args.dst)).expanduser()
    if dst.exists() and not args.force:
        got = _sha256(dst)
        if got == YOLOV8N_SHA256:
            print(f"OK: {dst} (sha256 проверен)")
            return
        # Делаем бэкап, чтобы не потерять текущий файл пользователя.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = dst.with_name(dst.name + f".bak_{ts}")
        dst.rename(bak)
        print(f"Сделан бэкап существующего файла: {bak}")

    print(f"Скачиваю из {YOLOV8N_URL}")
    _download(YOLOV8N_URL, dst)
    got = _sha256(dst)
    if got != YOLOV8N_SHA256:
        raise SystemExit(f"Не совпал sha256 для {dst}\n  получено: {got}\n  ожидалось: {YOLOV8N_SHA256}")
    print(f"Сохранено: {dst} (sha256 проверен)")


if __name__ == "__main__":
    main()
