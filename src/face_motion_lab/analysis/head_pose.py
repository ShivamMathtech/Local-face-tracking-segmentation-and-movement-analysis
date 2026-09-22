"""Approximate six-point head pose; uncalibrated intrinsics are explicitly flagged."""

import cv2
import numpy as np

MODEL = np.array(
    [(0, 0, 0), (0, 330, -65), (-225, -170, -135), (225, -170, -135), (-150, 150, -125), (150, 150, -125)],
    np.float64,
)
INDICES = [1, 152, 33, 263, 61, 291]


def estimate_pose(landmarks, shape, calibration=None):
    if landmarks is None or landmarks.kind != "dense" or len(landmarks.points) < 468:
        return None
    h, w = shape[:2]
    matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float64)
    dist = np.zeros((5, 1))
    if calibration:
        matrix = np.asarray(calibration["camera_matrix"], dtype=float).copy()
        cw, ch = calibration["resolution"]
        matrix[0, :] *= w / cw
        matrix[1, :] *= h / ch
        dist = np.asarray(calibration["distortion_coefficients"], dtype=float)
    points = landmarks.points[INDICES, :2].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(MODEL, points, matrix, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rotation)[0]
    axes, _ = cv2.projectPoints(
        np.array([(0, 0, 0), (100, 0, 0), (0, 100, 0), (0, 0, 100)], float), rvec, tvec, matrix, dist
    )
    return {
        "pitch": float(angles[0]),
        "yaw": float(angles[1]),
        "roll": float(angles[2]),
        "calibrated": calibration is not None,
        "axes": axes.reshape(-1, 2).tolist(),
    }
