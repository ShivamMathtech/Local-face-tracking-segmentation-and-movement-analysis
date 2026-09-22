"""UI-independent processing engine shared by desktop, CLI, tests and notebooks."""

from collections import deque
from dataclasses import asdict
import time
import numpy as np
import cv2
from .detection.detector_factory import create_detector
from .tracking.face_tracker import FaceTracker
from .landmarks.face_landmarks import LandmarkEngine
from .analysis.movement import MovementAnalyzer
from .analysis.coordinates import coordinates
from .analysis.head_pose import estimate_pose
from .analysis.calibration import load_calibration
from .segmentation.face_mask import face_mask, apply_segmentation
from .visualization.filters import transform, sharpness
from .visualization.heatmap import heatmap
from .visualization.overlays import draw_overlays


class Pipeline:
    def __init__(self, settings, detector=None):
        self.s = settings.validate()
        self.detector = detector or create_detector(settings)
        try:
            self.landmarker = LandmarkEngine(settings.landmark_model)
        except Exception:
            self.detector.close()
            raise
        self.calibration = load_calibration(settings.calibration_file) if settings.calibration_file else None
        self.tracker = FaceTracker(settings)
        self.analysis = MovementAnalyzer(settings)
        self.frame = None
        self.timestamp = 0.0
        self.number = 0
        self.metrics = None
        self.frames = 0
        self.lock_frames = 0
        self.tracked_frames = 0
        self.lost_frames = 0
        self.latencies = deque(maxlen=300)
        self.last_result = None
        self.first_time = None
        self.detection_count = 0
        self.measurements = deque(maxlen=settings.history_limit)

    def reset(self):
        self.tracker = FaceTracker(self.s)
        self.analysis = MovementAnalyzer(self.s)
        self.metrics = None
        self.lock_frames = 0
        self.tracked_frames = 0
        self.lost_frames = 0
        self.measurements.clear()

    def lock(self, face_id):
        self.tracker.lock(face_id)
        self.analysis = MovementAnalyzer(self.s)
        self.metrics = None
        self.lock_frames = 0
        self.tracked_frames = 0
        self.lost_frames = 0
        target = self.tracker.tracks[face_id]
        self.metrics = self.analysis.update(target.bbox, self.timestamp, self.number, face_id)
        self.lock_frames = 1
        self.tracked_frames = 1
        # Selection measures the displayed frame, including a still image.
        if self.metrics and self.measurements:
            row = self.measurements[-1]
            if row["frame"] == self.number and row["timestamp"] == self.timestamp:
                row.update(asdict(self.metrics))
                row.update(status="LOCKED", observed=True, face_id=face_id)

    def unlock(self):
        self.tracker.unlock()
        self.analysis.gap()
        self.metrics = None

    def process(self, frame, timestamp, number=0):
        started = time.perf_counter()
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Input must be uint8 BGR with 3 channels")
        if frame.shape[1] > self.s.processing_width:
            frame = cv2.resize(
                frame,
                (self.s.processing_width, round(frame.shape[0] * self.s.processing_width / frame.shape[1])),
            )
        if self.frame is not None and self.frame.shape != frame.shape:
            self.reset()
        self.frame = frame.copy()
        self.timestamp = timestamp
        self.number = number
        if self.first_time is None:
            self.first_time = timestamp
        detections = self.detector.detect(self.frame)
        self.detection_count = len(detections)
        if (
            self.s.landmarks
            or self.s.mask
            or self.s.head_pose
            or self.s.segmentation_mode != "Original"
            or self.s.view_mode == "Binary Mask"
        ):
            for d in detections:
                lm = self.landmarker.extract(self.frame, d.bbox)
                if lm is not None and (lm.kind == "dense" or d.landmarks is None):
                    d.landmarks = lm
        self.tracker.update(self.frame, detections, timestamp)
        self.frames += 1
        self.metrics = None
        target = self.tracker.tracks.get(self.tracker.locked_id)
        if self.tracker.locked_id is not None:
            self.lock_frames += 1
            if target is not None and target.observed:
                self.tracked_frames += 1
                self.metrics = self.analysis.update(target.bbox, timestamp, number, target.face_id)
            else:
                self.lost_frames += 1
                self.analysis.gap()
        result = self.render()
        latency = (time.perf_counter() - started) * 1000
        self.latencies.append(latency)
        result["latency_ms"] = latency
        result["processing_fps"] = 1000 / np.mean(self.latencies)
        row = {
            "timestamp": timestamp,
            "frame": number,
            "face_id": self.tracker.locked_id,
            "status": self.tracker.status,
            "observed": bool(target and target.observed),
            "detection_count": self.detection_count,
            "latency_ms": latency,
        }
        if self.metrics:
            row.update(asdict(self.metrics))
        self.measurements.append(row)
        result["row"] = row
        self.last_result = result
        return result

    def render(self):
        if self.frame is None:
            return None
        target = self.tracker.tracks.get(self.tracker.locked_id)
        active = target if target is not None and target.observed and not target.expired else None
        mask = np.zeros(self.frame.shape[:2], np.uint8)
        label = "No observed locked target"
        pose = blur = coord = None
        if active:
            mask, label = face_mask(self.frame, active.bbox, active.landmarks, self.s.mask_method)
            blur = sharpness(self.frame, active.bbox)
            coord = coordinates(
                *active.bbox.center, self.frame.shape[1], self.frame.shape[0], self.s.coordinate_origin
            )
            if self.s.head_pose:
                pose = estimate_pose(active.landmarks, self.frame.shape, self.calibration)
        density = heatmap(
            self.analysis.history,
            self.frame.shape,
            self.timestamp,
            self.s.heatmap_window,
            movement=self.s.view_mode == "Movement Intensity",
        )
        display = transform(self.frame, self.s, mask, active.bbox if active else None, density)
        display = apply_segmentation(display, mask, self.s)
        tracks = self.tracker.snapshot()
        display = draw_overlays(
            display,
            tracks,
            self.tracker.locked_id,
            self.s,
            self.analysis.trail,
            self.metrics,
            self.tracker.status,
            pose,
        )
        return {
            "image": display,
            "raw": self.frame.copy(),
            "mask": mask,
            "mask_label": label,
            "heatmap": density,
            "tracks": tracks,
            "locked_id": self.tracker.locked_id,
            "status": self.tracker.status,
            "metrics": asdict(self.metrics) if self.metrics else None,
            "coordinates": coord,
            "pose": pose,
            "blur": blur,
            "frame": self.number,
            "timestamp": self.timestamp,
            "detector": self.detector.name,
            "landmark_engine": self.landmarker.name,
            "summary": self.summary(),
            "graph_rows": self.analysis.rows()[-2000:],
            "detection_count": self.detection_count,
        }

    def summary(self):
        return {
            **self.analysis.summary(),
            "frames_processed": self.frames,
            "frames_tracked": self.tracked_frames,
            "frames_lost": self.lost_frames,
            "lock_frames": self.lock_frames,
            "continuity_percent": 100 * self.tracked_frames / self.lock_frames if self.lock_frames else None,
            "lost_percent": 100 * self.lost_frames / self.lock_frames if self.lock_frames else None,
            "reacquisitions": self.tracker.reacquisitions,
            "duration_seconds": max(0, self.timestamp - (self.first_time or 0)),
            "processing_fps": 1000 / np.mean(self.latencies) if self.latencies else 0,
        }

    def close(self):
        self.detector.close()
        self.landmarker.close()
