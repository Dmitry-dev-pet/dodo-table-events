from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Установить legacy YOLO зависимости для Python 3.8 (torch 1.13.1 + ultralytics 8.0.13).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="Показать команды установки, но не выполнять их")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if sys.version_info >= (3, 9):
        raise SystemExit(f"Этот скрипт предназначен для Python 3.8.x. Сейчас: {sys.version.split()[0]}")

    # В виртуальных окружениях Rye может не быть `pip`. Используем `uv pip install` (Rye сам использует uv).
    # Кэш uv держим внутри репозитория, чтобы не упираться в права в ~/.cache.
    repo_root = Path(__file__).resolve().parents[1]
    uv_cache_dir = repo_root / ".uv-cache"

    pkgs = ["torch==1.13.1", "torchvision==0.14.1", "ultralytics==8.0.13"]
    cmd = [
        "uv",
        "pip",
        "install",
        "--python",
        str(sys.executable),
        "--torch-backend",
        "auto",
        *pkgs,
    ]

    env = dict(os.environ)
    env["UV_CACHE_DIR"] = str(uv_cache_dir)

    print("+ " + " ".join(cmd))
    if not args.dry_run:
        subprocess.check_call(cmd, env=env)

    if args.dry_run:
        return

    # Быстрая проверка импорта
    try:
        import torch  # noqa: F401
        import ultralytics  # noqa: F401

        print("OK: torch и ultralytics успешно импортируются")
    except Exception as e:
        raise SystemExit(f"Пакеты установились, но импорт не прошёл: {e}") from e

    # Скачиваем совместимый yolov8n.pt (локальные веса могут быть несовместимы с legacy-структурой Ultralytics).
    try:
        subprocess.check_call(
            [
                sys.executable,
                str((Path(__file__).resolve().parent / "download_yolo_weights.py")),
                "--dst",
                "models/yolov8n.pt",
            ]
        )
    except Exception as e:
        print(f"ПРЕДУПРЕЖДЕНИЕ: не удалось автоматически скачать веса: {e}")


if __name__ == "__main__":
    main()
