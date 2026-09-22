from pathlib import Path
import json
import cv2
import numpy as np


def load_calibration(path):
    data = json.loads(Path(path).read_text())
    k = np.asarray(data["camera_matrix"])
    d = np.asarray(data["distortion_coefficients"])
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid intrinsic matrix")
    if d.size not in (4, 5, 8, 12, 14) or not np.isfinite(d).all():
        raise ValueError("Invalid distortion coefficients")
    if len(data["resolution"]) != 2 or min(data["resolution"]) <= 0:
        raise ValueError("Invalid calibration resolution")
    return data


def calibrate(paths, columns=9, rows=6, square_size=25.0):
    if columns < 2 or rows < 2 or square_size <= 0:
        raise ValueError("Invalid chessboard geometry")
    obj = np.zeros((rows * columns, 3), np.float32)
    obj[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_size
    objects = []
    images = []
    size = None
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        current = (gray.shape[1], gray.shape[0])
        if size is not None and current != size:
            raise ValueError("Calibration images must have equal resolution")
        size = current
        ok, corners = cv2.findChessboardCorners(gray, (columns, rows))
        if ok:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            objects.append(obj.copy())
            images.append(corners)
    if len(images) < 6:
        raise ValueError("At least six successful chessboard views are required; capture 15+ diverse views")
    error, k, d, _, _ = cv2.calibrateCamera(objects, images, size, None, None)
    return {
        "camera_matrix": k.tolist(),
        "distortion_coefficients": d.tolist(),
        "resolution": list(size),
        "rms_reprojection_error_px": float(error),
        "views": len(images),
        "square_size": square_size,
    }
