from __future__ import annotations

import argparse
import ast
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

MODEL_REPO = "opencv/person_detection_mediapipe"
MODEL_FILE = "person_detection_mediapipe_2023mar.onnx"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"

# SHA256 берём со страницы файла на Hugging Face ("Large File Pointer Details").
MODEL_SHA256 = "47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f"

MP_PERSONDET_PY_FILE = "mp_persondet.py"
MP_PERSONDET_PY_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MP_PERSONDET_PY_FILE}"
ANCHORS_FILE = "person_detection_mediapipe_2023mar.anchors.npy"


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
    with urllib.request.urlopen(req) as resp:  # nosec - URL is a fixed constant
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


def _extract_anchors_from_mp_persondet_py(text: str):
    anchor_def = "def _load_anchors"
    pos = text.find(anchor_def)
    if pos < 0:
        raise ValueError("Не удалось найти _load_anchors() в mp_persondet.py")
    pos = text.find("return np.array", pos)
    if pos < 0:
        raise ValueError("Не удалось найти 'return np.array' в mp_persondet.py")
    pos = text.find("[", pos)
    if pos < 0:
        raise ValueError("Не удалось найти список якорей '[' в mp_persondet.py")

    depth = 0
    end = None
    for i in range(pos, len(text)):
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError("Не удалось распарсить список якорей (нет закрывающей ']' )")

    list_text = text[pos:end]
    anchors = ast.literal_eval(list_text)
    return anchors


def ensure_anchors(*, model_path: Path, dst_dir: Path, force: bool) -> Path:
    anchors_path = dst_dir / ANCHORS_FILE
    if anchors_path.exists() and not force:
        print(f"OK: {anchors_path}")
        return anchors_path

    # Ленивый импорт, чтобы скрипт мог скачивать файлы даже если opencv-python ещё не установлен.
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Чтобы сгенерировать anchors нужен opencv-python + numpy: {e}") from e

    # Узнаём ожидаемое количество anchors из выходов ONNX модели.
    net = cv2.dnn.readNet(str(model_path))
    out_names = net.getUnconnectedOutLayersNames()
    blob = np.zeros((1, 3, 224, 224), dtype=np.float32)
    net.setInput(blob)
    out0, out1 = net.forward(out_names)
    expected_n = int(out0.shape[1])

    # Скачиваем mp_persondet.py и извлекаем anchors.
    tmp_py = dst_dir / MP_PERSONDET_PY_FILE
    print(f"Скачиваю исходник anchors из {MP_PERSONDET_PY_URL}")
    _download(MP_PERSONDET_PY_URL, tmp_py)
    text = tmp_py.read_text(encoding="utf-8", errors="replace")
    anchors = _extract_anchors_from_mp_persondet_py(text)
    arr = np.array(anchors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise SystemExit(f"Неожиданная форма anchors из mp_persondet.py: {arr.shape}")
    if int(arr.shape[0]) != expected_n:
        raise SystemExit(f"Несовпадение числа anchors: получено {arr.shape[0]}, а модель ожидает {expected_n}")

    anchors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(anchors_path), arr)
    print(f"Saved: {anchors_path} (n={arr.shape[0]})")
    return anchors_path


def ensure_model(*, dst_dir: Path, force: bool) -> Path:
    dst = dst_dir / MODEL_FILE
    if dst.exists() and not force:
        if _sha256(dst) == MODEL_SHA256:
            print(f"OK: {dst} (sha256 verified)")
            return dst
        print(f"ПРЕДУПРЕЖДЕНИЕ: {dst} уже существует, но sha256 не совпал; скачиваю заново.")

    print(f"Скачиваю модель из {MODEL_URL}")
    _download(MODEL_URL, dst)
    got = _sha256(dst)
    if got != MODEL_SHA256:
        raise SystemExit(f"Не совпал sha256 для {dst}\n  получено: {got}\n  ожидалось: {MODEL_SHA256}")
    print(f"Saved: {dst} (sha256 verified)")
    return dst


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Скачать OpenCV DNN модель(и), используемые в этом репозитории.\n"
            f"Сейчас: {MODEL_REPO}/{MODEL_FILE} (детектор человека, ONNX)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dst-dir",
        type=str,
        default="models/opencv_dnn",
        help="Куда сохранять скачанные файлы модели",
    )
    p.add_argument("--force", action="store_true", default=False, help="Скачать заново, даже если файл уже есть")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dst_dir = Path(str(args.dst_dir)).expanduser()
    model_path = ensure_model(dst_dir=dst_dir, force=bool(args.force))
    ensure_anchors(model_path=model_path, dst_dir=dst_dir, force=bool(args.force))


if __name__ == "__main__":
    main()
