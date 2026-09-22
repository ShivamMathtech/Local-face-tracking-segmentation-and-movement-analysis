import cv2
import numpy as np
from face_motion_lab.types import Box


def estimate_flow(previous, current, box: Box) -> tuple[float, float] | None:
    if previous is None or previous.shape != current.shape:
        return None
    x, y, w, h = box.clipped(current.shape[1], current.shape[0]).ints()
    if w < 8 or h < 8:
        return None
    mask = np.zeros_like(previous)
    mask[y : y + h, x : x + w] = 255
    points = cv2.goodFeaturesToTrack(previous, 60, 0.02, 5, mask=mask)
    if points is None or len(points) < 5:
        return None
    moved, ok, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if moved is None:
        return None
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None)
    if back is None:
        return None
    valid = (ok.ravel() == 1) & (ok_back.ravel() == 1) & (np.linalg.norm(back - points, axis=2).ravel() < 1.5)
    if valid.sum() < 5:
        return None
    delta = (moved - points).reshape(-1, 2)[valid]
    center = np.median(delta, axis=0)
    if np.median(np.linalg.norm(delta - center, axis=1)) > 8:
        return None
    return float(center[0]), float(center[1])
