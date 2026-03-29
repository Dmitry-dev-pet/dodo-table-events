from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

VIDEO_SPECS = [
    (
        "video1.mp4",
        "https://drive.usercontent.google.com/download?id=1pAPTjESoDgjqhTaqM_graYfMpWcOyzRe&export=download&confirm=t",
    ),
    (
        "video2.mp4",
        "https://drive.usercontent.google.com/download?id=1rYmJB13vvV96JuDUrBvlEXtoKFPWo75A&export=download&confirm=t",
    ),
    (
        "video3.mp4",
        "https://drive.usercontent.google.com/download?id=1xfHTf3vJVlTXXs0Rdq9L_xi816ATX5zD&export=download&confirm=t",
    ),
]


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": "dodo-table-events/0.1"})
    with urllib.request.urlopen(req) as resp:  # nosec - URL is fixed in code
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
        description="Скачать входные видео из ТЗ в data/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dst-dir", default="data", type=str, help="Куда сохранить video1.mp4/video2.mp4/video3.mp4")
    p.add_argument("--force", action="store_true", help="Скачать заново, даже если файл уже есть")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dst_dir = Path(str(args.dst_dir)).expanduser()

    for filename, url in VIDEO_SPECS:
        dst = dst_dir / filename
        if dst.exists() and dst.stat().st_size > 0 and not args.force:
            print(f"OK: {dst}")
            continue
        print(f"Скачиваю {filename}")
        _download(url, dst)
        print(f"Сохранено: {dst}")


if __name__ == "__main__":
    main()
