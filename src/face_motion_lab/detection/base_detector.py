from typing import Protocol
import numpy as np
from face_motion_lab.types import FaceDetection


class Detector(Protocol):
    name: str

    def detect(self, frame: np.ndarray) -> list[FaceDetection]: ...
    def close(self) -> None: ...
