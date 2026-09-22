"""Deterministic geometric subjects for smoke tests; these are NOT real face detections."""

import cv2
import numpy as np
from face_motion_lab.types import Box, FaceDetection


def boxes_at(n: int) -> list[Box]:
    return [
        Box(125 + 90 * np.sin(n / 35), 130 + 45 * np.cos(n / 27), 105, 135),
        Box(465 + 65 * np.sin(n / 43), 155 + 50 * np.sin(n / 30), 95, 120),
    ]


def demo_frame(n: int) -> np.ndarray:
    frame = np.full((480, 800, 3), (30, 24, 18), np.uint8)
    for i, box in enumerate(boxes_at(n)):
        if i == 0 and 150 <= n % 360 < 180:
            continue
        x, y, w, h = box.ints()
        color = (90, 180, 220) if i == 0 else (195, 120, 85)
        cv2.ellipse(frame, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, color, -1)
        for ex in (x + w // 3, x + 2 * w // 3):
            cv2.circle(frame, (ex, y + h // 3), 5, (20, 20, 20), -1)
        cv2.ellipse(frame, (x + w // 2, y + 2 * h // 3), (18, 8), 0, 0, 180, (20, 20, 20), 2)
    cv2.putText(
        frame,
        "SYNTHETIC DEMO - scripted boxes, no detector benchmark",
        (20, 450),
        0,
        0.55,
        (170, 180, 190),
        1,
    )
    return frame


class SyntheticDetector:
    name = "Synthetic scripted boxes (demo only)"

    def __init__(self):
        self.number = 0

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        n = self.number
        self.number += 1
        return [
            FaceDetection(
                Box(
                    b.x * frame.shape[1] / 800,
                    b.y * frame.shape[0] / 480,
                    b.width * frame.shape[1] / 800,
                    b.height * frame.shape[0] / 480,
                ),
                None,
            )
            for i, b in enumerate(boxes_at(n))
            if not (i == 0 and 150 <= n % 360 < 180)
        ]

    def close(self) -> None:
        self.number = 0
