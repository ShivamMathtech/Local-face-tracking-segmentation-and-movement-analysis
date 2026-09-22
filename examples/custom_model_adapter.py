"""Runnable ONNX adapter example for a model returning post-NMS Nx6 xyxy/score/class rows.
This is a defined contract, not a claim that every YOLO export has this output layout.
Usage: python examples/custom_model_adapter.py model.onnx image.jpg
"""

import sys
import cv2
import numpy as np
from face_motion_lab.detection.ml_adapter import ModelDetector
from face_motion_lab.types import Box, FaceDetection


def preprocess(frame):
    image = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255


def decode(outputs, shape):
    rows = np.asarray(outputs[0]).reshape(-1, 6)
    h, w = shape[:2]
    result = []
    for x1, y1, x2, y2, score, cls in rows:
        if score < 0.6 or int(cls) != 0:
            continue
        box = Box(x1 * w / 640, y1 * h / 640, (x2 - x1) * w / 640, (y2 - y1) * h / 640).clipped(w, h)
        result.append(FaceDetection(box, float(score)))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python examples/custom_model_adapter.py model.onnx image.jpg")
    detector = ModelDetector("onnx", sys.argv[1], preprocess, decode)
    frame = cv2.imread(sys.argv[2])
    if frame is None:
        raise SystemExit("Image could not be read")
    print(detector.detect(frame))
    detector.close()
