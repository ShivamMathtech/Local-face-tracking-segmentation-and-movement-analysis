from face_motion_lab.types import Box


def iou(a: Box, b: Box) -> float:
    w = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    h = max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    intersection = w * h
    return intersection / max(1e-9, a.area + b.area - intersection)
