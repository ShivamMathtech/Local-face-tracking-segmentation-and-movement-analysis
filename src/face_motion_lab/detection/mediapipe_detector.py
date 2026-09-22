"""MediaPipe Tasks API; no dependency on removed mp.solutions APIs."""

from pathlib import Path
import cv2
import numpy as np
from face_motion_lab.types import Box, FaceDetection, Landmarks


class MediaPipeDetector:
    name = "MediaPipe Face Detector"

    def __init__(self, path: str, confidence: float):
        if not Path(path).is_file():
            raise FileNotFoundError("Choose a MediaPipe face detector .tflite model")
        import mediapipe as mp

        self.mp = mp
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=confidence,
        )
        self.model = mp.tasks.vision.FaceDetector.create_from_options(options)

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.model.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        h, w = frame.shape[:2]
        detections = []
        for d in result.detections:
            b = d.bounding_box
            points = np.array([(k.x * w, k.y * h) for k in d.keypoints], dtype=float)
            detections.append(
                FaceDetection(
                    Box(b.origin_x, b.origin_y, b.width, b.height).clipped(w, h),
                    float(d.categories[0].score),
                    Landmarks(points, "six-point"),
                )
            )
        return detections

    def close(self):
        self.model.close()
