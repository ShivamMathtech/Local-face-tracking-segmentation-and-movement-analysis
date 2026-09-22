from .opencv_detector import HaarDetector, YuNetDetector


def create_detector(settings):
    if settings.detector == "haar":
        return HaarDetector()
    if settings.detector == "yunet":
        return YuNetDetector(settings.detector_model, settings.detection_confidence)
    if settings.detector == "mediapipe":
        from .mediapipe_detector import MediaPipeDetector

        return MediaPipeDetector(settings.detector_model, settings.detection_confidence)
    raise ValueError(f"Unknown detector: {settings.detector}")
