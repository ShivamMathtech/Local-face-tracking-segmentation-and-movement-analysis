import cv2
import numpy as np


def heatmap(rows, shape, now, window=30.0, movement=False):
    h, w = shape[:2]
    field = np.zeros((max(1, h // 4), max(1, w // 4)), np.float32)
    recent = [m for m in rows if now - window <= m.timestamp <= now]
    previous = None
    for m in recent:
        # Dwell weighting uses actual observed intervals, never bridges a lost segment.
        dt = (
            0.0
            if previous is None or previous.segment != m.segment
            else min(0.25, max(0, m.timestamp - previous.timestamp))
        )
        x = int(np.clip(m.cx / 4, 0, field.shape[1] - 1))
        y = int(np.clip(m.cy / 4, 0, field.shape[0] - 1))
        field[y, x] += m.speed * dt if movement else dt
        previous = m
    field = cv2.GaussianBlur(field, (0, 0), 5)
    gray = cv2.normalize(field, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    rgb[gray == 0] = [25, 20, 18]
    return cv2.resize(rgb, (w, h))
