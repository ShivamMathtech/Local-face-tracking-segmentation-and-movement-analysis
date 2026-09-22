"""Source abstraction; BGR8 frames and monotonic source timestamps in seconds."""

from dataclasses import dataclass
from pathlib import Path
import time
import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    number: int


class FrameSource:
    def __init__(self, source: str | int, settings):
        self.source = source
        self.cap = None
        self.still = None
        self.number = 0
        self.started = time.monotonic()
        self.last_time = -1.0
        self.fps = settings.fps
        self.live = isinstance(source, int) or str(source).lower().startswith(
            ("rtsp://", "rtsps://", "http://", "https://")
        )
        self.demo = source == "demo"
        self.kind = "camera" if isinstance(source, int) else ("network stream" if self.live else "file")
        if self.demo:
            self.kind = "synthetic demo"
        elif not self.live and Path(str(source)).suffix.lower() in (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        ):
            data = np.fromfile(str(source), dtype=np.uint8)
            self.still = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if self.still is None:
                raise ValueError("Unable to decode image")
            self.kind = "image"
        else:
            if self.live and not isinstance(source, int):
                self.cap = cv2.VideoCapture(
                    source,
                    cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000],
                )
            else:
                self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                self.cap.release()
                raise OSError("Unable to open source. Check path, camera permissions or stream settings.")
            if isinstance(source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
                self.cap.set(cv2.CAP_PROP_FPS, settings.fps)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if 0 < fps <= 240:
                self.fps = fps

    def read(self) -> Frame | None:
        if self.demo:
            from .synthetic import demo_frame

            image = demo_frame(self.number)
            t = self.number / self.fps
        elif self.still is not None:
            if self.number:
                return None
            image = self.still.copy()
            t = 0.0
        else:
            ok, image = self.cap.read()
            if not ok:
                return None
            if self.live:
                t = time.monotonic() - self.started
            else:
                t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                # Some codecs provide no PTS: explicitly documented FPS fallback.
                if t <= self.last_time:
                    t = self.last_time + 1 / self.fps
        frame = Frame(image, float(t), self.number)
        self.number += 1
        self.last_time = float(t)
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
