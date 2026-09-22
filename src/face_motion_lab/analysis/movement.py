"""Timestamp-based motion. Missing observations break derivatives, not accumulated distance."""

from collections import deque
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass
class MovementMetrics:
    timestamp: float
    frame: int
    face_id: int
    x: float
    y: float
    width: float
    height: float
    cx: float
    cy: float
    dx: float
    dy: float
    distance: float
    total_distance: float
    vx: float
    vy: float
    speed: float
    ax: float
    ay: float
    acceleration: float
    direction: str
    area: float
    aspect_ratio: float
    relative_distance: str
    segment: int


def direction(vx, vy, threshold=8.0):
    if math.hypot(vx, vy) < threshold:
        return "STATIONARY"
    labels = ["RIGHT", "DOWN-RIGHT", "DOWN", "DOWN-LEFT", "LEFT", "UP-LEFT", "UP", "UP-RIGHT"]
    return labels[int(math.floor((math.atan2(vy, vx) + math.pi / 8) / (math.pi / 4))) % 8]


class MovementAnalyzer:
    def __init__(self, settings):
        self.s = settings
        self.previous = None
        self.total_distance = 0.0
        self.segment = 0
        self.history = deque(maxlen=settings.history_limit)
        self.trail = deque(maxlen=settings.trajectory_length or None)
        self.samples = 0
        self.sums = {}
        self.mins = {}
        self.maxs = {}
        self.last_timestamp = None

    def gap(self):
        if self.previous is not None:
            self.segment += 1
        self.previous = None

    def update(self, box, timestamp, frame, face_id):
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return None
        raw = np.array(box.center)
        prev = self.previous
        center = (
            raw
            if prev is None
            else self.s.smoothing_factor * raw + (1 - self.s.smoothing_factor) * np.array([prev.cx, prev.cy])
        )
        dx = dy = distance = vx = vy = ax = ay = 0.0
        relative = "STABLE"
        if prev is not None:
            dt = timestamp - prev.timestamp
            dx, dy = (center - np.array([prev.cx, prev.cy])).tolist()
            distance = math.hypot(dx, dy)
            self.total_distance += distance
            a = self.s.smoothing_factor
            vx = a * dx / dt + (1 - a) * prev.vx
            vy = a * dy / dt + (1 - a) * prev.vy
            ax = (vx - prev.vx) / dt
            ay = (vy - prev.vy) / dt
            ratio = box.area / max(1, prev.area)
            relative = "CLOSER" if ratio > 1.035 else ("FARTHER" if ratio < 0.965 else "STABLE")
        m = MovementMetrics(
            timestamp,
            frame,
            face_id,
            box.x,
            box.y,
            box.width,
            box.height,
            float(center[0]),
            float(center[1]),
            dx,
            dy,
            distance,
            self.total_distance,
            vx,
            vy,
            math.hypot(vx, vy),
            ax,
            ay,
            math.hypot(ax, ay),
            direction(vx, vy, self.s.direction_threshold),
            box.area,
            box.width / max(box.height, 1),
            relative,
            self.segment,
        )
        self.previous = m
        self.last_timestamp = timestamp
        self.history.append(m)
        self.trail.append((timestamp, m.cx, m.cy, self.segment, m.speed))
        self.samples += 1
        for k in ("speed", "acceleration", "area", "cx", "cy"):
            v = getattr(m, k)
            self.sums[k] = self.sums.get(k, 0) + v
            self.mins[k] = min(self.mins.get(k, v), v)
            self.maxs[k] = max(self.maxs.get(k, v), v)
        return m

    def summary(self):
        result = {"observations": self.samples, "total_distance_px": self.total_distance}
        for k in self.sums:
            result[f"average_{k}"] = self.sums[k] / self.samples
            result[f"minimum_{k}"] = self.mins[k]
            result[f"maximum_{k}"] = self.maxs[k]
        return result

    def rows(self):
        return [asdict(m) for m in self.history]
