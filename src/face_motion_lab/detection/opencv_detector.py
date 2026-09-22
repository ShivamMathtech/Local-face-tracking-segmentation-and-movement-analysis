import cv2
from face_motion_lab.types import Box, FaceDetection, Landmarks


class HaarDetector:
    name = "OpenCV Haar (confidence unavailable)"

    def __init__(self):
        self.model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.model.empty():
            raise RuntimeError("OpenCV bundled Haar model is unavailable")

    def detect(self, frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        boxes = self.model.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(32, 32))
        return [FaceDetection(Box(*map(float, b)), None) for b in boxes]

    def close(self):
        self.model = None


class YuNetDetector:
    name = "OpenCV DNN YuNet"

    def __init__(self, path: str, confidence: float):
        from pathlib import Path

        if not Path(path).is_file():
            raise FileNotFoundError("Choose a YuNet ONNX model in Settings")
        self.model = cv2.FaceDetectorYN.create(path, "", (320, 320), confidence, 0.3, 5000)

    def detect(self, frame):
        self.model.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.model.detect(frame)
        if faces is None:
            return []
        return [
            FaceDetection(
                Box(*map(float, r[:4])).clipped(frame.shape[1], frame.shape[0]),
                float(r[14]),
                Landmarks(r[4:14].reshape(5, 2), "five-point"),
            )
            for r in faces
        ]

    def close(self):
        self.model = None
