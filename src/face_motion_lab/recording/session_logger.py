"""Explicit recording; analytics stream to disk. No network uploads."""

from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict
import csv
import json
import cv2
from .exporter import FIELDS


class SessionRecorder:
    def __init__(self, root, settings, source_kind, shape, fps, detector):
        self.folder = Path(root) / datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%S_%fZ")
        self.folder.mkdir(parents=True, exist_ok=False)
        self.fps = float(fps)
        self.count = 0
        self.start_t = None
        self.previous = None
        self.closed = False
        self.shape = shape[:2]
        h, w = self.shape
        self.video = cv2.VideoWriter(
            str(self.folder / "processed_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h)
        )
        if not self.video.isOpened():
            self.video.release()
            self.folder.rmdir()
            raise OSError("MP4 encoder unavailable. Install an OpenCV build with FFmpeg MPEG-4 support.")
        self.csv = (self.folder / "tracking_data.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.csv, fieldnames=FIELDS, extrasaction="ignore")
        self.writer.writeheader()
        self.landmarks = (self.folder / "landmarks.jsonl").open("w", encoding="utf-8")
        self.meta = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "source_kind": source_kind,
            "resolution": [w, h],
            "fps": fps,
            "detector": detector,
            "tracker": "Kalman / gated appearance and geometry association",
            "settings": asdict(settings),
            "video_timing": "CFR resampling of source timestamps; long gaps limited to 10 seconds",
            "frames_written": 0,
            "large_timestamp_gaps": 0,
        }
        self._save()

    def _save(self):
        (self.folder / "session.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def write(self, result):
        if self.closed:
            raise RuntimeError("Recording is closed")
        image = result["image"]
        if image.shape[:2] != self.shape:
            raise ValueError("Recording resolution changed; start a new recording")
        t = result["timestamp"]
        if self.start_t is None:
            self.start_t = t
        desired = max(0, round((t - self.start_t) * self.fps))
        if desired - self.count > self.fps * 10:
            self.meta["large_timestamp_gaps"] += 1
            self.start_t = t - self.count / self.fps
            desired = self.count
        # Hold the previous frame across short processing gaps; analytics retain exact time.
        while self.count < desired:
            self.video.write(self.previous if self.previous is not None else image)
            self.count += 1
        if self.count == desired:
            self.video.write(image)
            self.count += 1
        self.previous = image.copy()
        self.writer.writerow(result["row"])
        faces = [
            {
                "face_id": track.face_id,
                "kind": track.landmarks.kind,
                "points": track.landmarks.points.tolist(),
            }
            for track in result["tracks"]
            if track.observed and track.landmarks is not None
        ]
        self.landmarks.write(json.dumps({"timestamp": t, "frame": result["frame"], "faces": faces}) + "\n")
        self.csv.flush()
        self.landmarks.flush()

    def close(self, summary=None):
        if self.closed:
            return
        self.video.release()
        self.csv.close()
        self.landmarks.close()
        self.closed = True
        self.meta.update(
            end_time=datetime.now(timezone.utc).isoformat(), frames_written=self.count, summary=summary or {}
        )
        self._save()
