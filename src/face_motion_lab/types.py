"""Typed data shared by detectors, trackers, analytics and the UI."""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def clipped(self, width: int, height: int) -> "Box":
        x = min(max(self.x, 0), width)
        y = min(max(self.y, 0), height)
        return Box(
            x, y, max(0, min(self.x + self.width, width) - x), max(0, min(self.y + self.height, height) - y)
        )

    def ints(self) -> tuple[int, int, int, int]:
        return tuple(int(round(v)) for v in (self.x, self.y, self.width, self.height))

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height


@dataclass
class Landmarks:
    points: np.ndarray
    kind: str = "dense"

    # Sparse fallback features are actual detected corners, not anatomical landmarks.
    def normalized(self, width: int, height: int) -> np.ndarray:
        return self.points[:, :2] / np.array([width, height])


@dataclass
class FaceDetection:
    bbox: Box
    confidence: float | None
    landmarks: Landmarks | None = None
    appearance: np.ndarray | None = field(default=None, repr=False)
    face_id: int | None = None


@dataclass
class FaceTrack:
    face_id: int
    bbox: Box
    confidence: float | None
    status: str
    observed: bool
    lost_frames: int = 0
    association_score: float | None = None
    landmarks: Landmarks | None = None
