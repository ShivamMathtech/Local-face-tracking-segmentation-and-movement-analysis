"""Short-term association; IDs are session tracks, never biometric identities."""

from dataclasses import dataclass
import math
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from face_motion_lab.types import Box, FaceTrack
from face_motion_lab.utils.geometry import iou
from .kalman_tracker import KalmanBox
from .optical_flow import estimate_flow


@dataclass
class Track:
    face_id: int
    kf: KalmanBox
    bbox: Box
    appearance: np.ndarray
    landmarks: object = None
    confidence: float | None = None
    lost: int = 0
    observed: bool = True
    score: float | None = None
    expired: bool = False
    hits: int = 1


def appearance(frame, box):
    x, y, w, h = box.clipped(frame.shape[1], frame.shape[0]).ints()
    if w < 2 or h < 2:
        return np.zeros((16, 16), np.float32)
    hsv = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)


class FaceTracker:
    def __init__(self, settings):
        self.s = settings
        self.tracks = {}
        self.next_id = 1
        self.locked_id = None
        self.previous = None
        self.last_time = None
        self.reacquisitions = 0
        self.explicit_status = "UNLOCKED"

    def lock(self, face_id: int):
        t = self.tracks.get(face_id)
        if t is None or not t.observed or t.expired:
            raise ValueError("Select a currently detected face")
        self.locked_id = face_id
        self.explicit_status = "LOCKED"

    def unlock(self):
        self.locked_id = None
        self.explicit_status = "UNLOCKED"

    @property
    def status(self):
        if self.locked_id is None:
            return "UNLOCKED"
        t = self.tracks.get(self.locked_id)
        if t is None or t.expired:
            return "REACQUIRING"
        return "LOCKED" if t.observed else "TEMPORARILY_LOST"

    def update(self, frame, detections, timestamp: float):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dt = 1 / self.s.fps if self.last_time is None else max(0.001, timestamp - self.last_time)
        self.last_time = timestamp
        for d in detections:
            d.appearance = appearance(frame, d.bbox)
        active = [t for t in self.tracks.values() if not t.expired]
        predictions = []
        originals = {t.face_id: t.bbox for t in active}
        for t in active:
            old = t.bbox
            predicted = t.kf.predict(dt) if self.s.kalman else old
            if self.s.optical_flow and t.lost < 3:
                flow = estimate_flow(self.previous, gray, old)
                if flow is not None:
                    predicted = Box(old.x + flow[0], old.y + flow[1], predicted.width, predicted.height)
            predictions.append(predicted)
            t.bbox = predicted
            t.observed = False
        costs = np.full((len(active), len(detections)), 1e6)
        for i, t in enumerate(active):
            for j, d in enumerate(detections):
                predicted = predictions[i]
                scale = max(30, math.hypot(predicted.width, predicted.height))
                distance = math.dist(predicted.center, d.bbox.center) / scale
                ratio = d.bbox.area / max(1, predicted.area)
                hist = float(cv2.compareHist(t.appearance, d.appearance, cv2.HISTCMP_BHATTACHARYYA))
                # Conservative gates prevent easy swaps, but cannot guarantee identity.
                if distance > 1.25 or not 0.35 < ratio < 2.8 or hist > 0.72:
                    continue
                if t.face_id == self.locked_id and (distance > 0.8 or hist > 0.5):
                    continue
                cost = 0.40 * (1 - iou(predicted, d.bbox)) + 0.30 * min(distance, 1) + 0.30 * hist
                if t.landmarks is not None and d.landmarks is not None:
                    a, b = t.landmarks, d.landmarks
                    if (
                        a.kind == b.kind
                        and a.kind in ("dense", "five-point", "six-point")
                        and a.points.shape == b.points.shape
                    ):
                        old = originals[t.face_id]
                        aa = (a.points[:, :2] - np.array(old.center)) / max(old.width, 1)
                        bb = (b.points[:, :2] - np.array(d.bbox.center)) / max(d.bbox.width, 1)
                        shape = float(np.mean(np.linalg.norm(aa - bb, axis=1)))
                        cost = 0.9 * cost + 0.1 * min(shape, 1)
                costs[i, j] = cost
        # If a locked face has two plausible candidates, suspend association.
        for i, t in enumerate(active):
            if t.face_id == self.locked_id and len(detections) > 1:
                c = np.sort(costs[i])
                if c[0] < 1e5 and c[1] - c[0] < 0.10:
                    costs[i, :] = 1e6
        matched_t = set()
        matched_d = set()
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                t = active[i]
                d = detections[j]
                score = 1 - costs[i, j]
                if costs[i, j] >= 1e5 or score < self.s.tracking_confidence:
                    continue
                # A detection contested by another similar track is also ambiguous.
                other = np.delete(costs[:, j], i)
                if t.face_id == self.locked_id and other.size and np.min(other) - costs[i, j] < 0.08:
                    continue
                if t.lost and t.face_id == self.locked_id:
                    self.reacquisitions += 1
                t.kf.correct(d.bbox)
                t.bbox = t.kf.box if self.s.kalman else d.bbox
                t.appearance = 0.92 * t.appearance + 0.08 * d.appearance
                t.landmarks = d.landmarks
                t.confidence = d.confidence
                t.lost = 0
                t.observed = True
                t.score = float(score)
                t.hits += 1
                d.face_id = t.face_id
                matched_t.add(t.face_id)
                matched_d.add(j)
        for t in active:
            if t.face_id not in matched_t:
                t.lost += 1
                t.landmarks = None
                t.score = None
                if t.lost > self.s.max_lost_frames:
                    t.expired = True
        # An expired locked ID is retained as lost; new detections get new IDs.
        for j, d in enumerate(detections):
            if j not in matched_d:
                ident = self.next_id
                self.next_id += 1
                self.tracks[ident] = Track(
                    ident, KalmanBox(d.bbox), d.bbox, d.appearance, d.landmarks, d.confidence
                )
                d.face_id = ident
        for ident in list(self.tracks):
            if self.tracks[ident].expired and ident != self.locked_id:
                del self.tracks[ident]
        self.previous = gray
        return self.snapshot()

    def snapshot(self):
        return [
            FaceTrack(
                t.face_id,
                t.bbox,
                t.confidence,
                (
                    self.status
                    if t.face_id == self.locked_id
                    else (("DETECTED" if t.hits == 1 else "TRACKING") if t.observed else "TEMPORARILY_LOST")
                ),
                t.observed,
                t.lost,
                t.score,
                t.landmarks,
            )
            for t in self.tracks.values()
        ]
