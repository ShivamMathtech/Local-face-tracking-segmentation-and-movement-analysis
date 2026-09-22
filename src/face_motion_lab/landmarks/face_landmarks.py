"""Dense model landmarks or honestly labelled image-feature fallback."""

from pathlib import Path
import cv2
import numpy as np
from face_motion_lab.types import Landmarks

OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]
FEATURES = [
    [33, 160, 158, 133, 153, 144, 33],
    [362, 385, 387, 263, 373, 380, 362],
    [61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91, 61],
    [70, 63, 105, 66, 107],
    [336, 296, 334, 293, 300],
    [168, 6, 197, 195, 5, 4, 1, 2],
    OVAL + [10],
]


class LandmarkEngine:
    def __init__(self, path: str = ""):
        self.model = None
        self.name = "Sparse image corners / ellipse mask fallback"
        if path:
            if not Path(path).is_file():
                raise FileNotFoundError("Face Landmarker model is missing")
            import mediapipe as mp

            self.mp = mp
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=path),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
            )
            self.model = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self.name = "MediaPipe dense landmarks"

    def extract(self, frame, box):
        height, width = frame.shape[:2]
        from face_motion_lab.types import Box

        roi = Box(
            box.x - 0.15 * box.width, box.y - 0.15 * box.height, box.width * 1.3, box.height * 1.3
        ).clipped(width, height)
        x, y, w, h = roi.ints()
        if w < 8 or h < 8:
            return None
        crop = frame[y : y + h, x : x + w]
        if self.model:
            result = self.model.detect(
                self.mp.Image(
                    image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                )
            )
            if result.face_landmarks:
                points = np.array([(p.x * w + x, p.y * h + y, p.z * w) for p in result.face_landmarks[0]])
                return Landmarks(points, "dense")
            return None
        # These are texture corners. Never label them eyes/nose/lips or use them for pose.
        bx, by, bw, bh = box.clipped(width, height).ints()
        gray = cv2.cvtColor(frame[by : by + bh, bx : bx + bw], cv2.COLOR_BGR2GRAY)
        pts = cv2.goodFeaturesToTrack(gray, 40, 0.03, 5)
        return Landmarks(pts.reshape(-1, 2) + [bx, by], "sparse corners") if pts is not None else None

    def close(self):
        if self.model:
            self.model.close()
