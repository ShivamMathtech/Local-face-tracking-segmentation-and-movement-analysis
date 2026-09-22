"""Explicit model downloads; inference never contacts the network."""

from pathlib import Path
import hashlib
import urllib.request

MODELS = {
    "landmarker": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "face_landmarker.task",
    ),
    "detector": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "blaze_face_short_range.tflite",
    ),
    "yunet": (
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "face_detection_yunet_2023mar.onnx",
    ),
}


def download_model(name: str, folder: str = "models", expected_sha256: str | None = None):
    if name not in MODELS:
        raise ValueError(f"Choose one of: {', '.join(MODELS)}")
    url, filename = MODELS[name]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    temp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as out:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > 100 * 1024 * 1024:
                    raise ValueError("Model exceeds size limit")
                out.write(chunk)
        if temp.stat().st_size < 10000:
            raise ValueError("Downloaded response is too small to be a model")
        digest = hashlib.sha256(temp.read_bytes()).hexdigest()
        if expected_sha256 and digest.lower() != expected_sha256.lower():
            raise ValueError("Model SHA-256 mismatch")
        temp.replace(path)
        path.with_suffix(path.suffix + ".sha256").write_text(digest + "  " + filename + "\n")
        return path, digest
    finally:
        if temp.exists():
            temp.unlink()
