import cv2
import numpy as np
from face_motion_lab.landmarks.face_landmarks import OVAL


def face_mask(frame, box, landmarks=None, method="refined"):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    dense = landmarks is not None and landmarks.kind == "dense" and len(landmarks.points) > 454
    if dense and method != "ellipse":
        pts = np.rint(landmarks.points[OVAL, :2]).astype(np.int32)
        if method == "hull":
            pts = cv2.convexHull(landmarks.points[:, :2].astype(np.int32))
        cv2.fillPoly(mask, [pts], 255)
        label = "landmark polygon" if method != "hull" else "landmark convex hull"
    else:
        cx, cy = box.center
        cv2.ellipse(
            mask,
            (round(cx), round(cy)),
            (max(1, round(box.width * 0.48)), max(1, round(box.height * 0.49))),
            0,
            0,
            360,
            255,
            -1,
        )
        label = "approximate ellipse (no dense model)" if not dense else "ellipse"
    if method == "skin":
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        mask = cv2.bitwise_and(mask, skin)
        label += " + heuristic skin threshold (lighting dependent)"
    if method in ("refined", "skin"):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Fill holes using external contours, then soften polygon edges.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, 255, -1)
        mask = cv2.GaussianBlur(mask, (7, 7), 1.5)
    return mask, label


def apply_segmentation(frame, mask, settings):
    mode = settings.segmentation_mode
    alpha = mask.astype(float) / 255
    if mode in ("Masked Face", "Background Removed"):
        background = np.zeros_like(frame) if mode == "Masked Face" else np.full_like(frame, 245)
        return (frame * alpha[:, :, None] + background * (1 - alpha[:, :, None])).astype(np.uint8)
    if mode == "Face Silhouette":
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if mode == "Face Highlighted":
        gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        frame = (frame * alpha[:, :, None] + gray * (1 - alpha[:, :, None]) * 0.35).astype(np.uint8)
    if settings.mask or mode == "Face Mask":
        if settings.mask_fill:
            a = alpha[:, :, None] * settings.mask_opacity
            frame = (frame * (1 - a) + np.array([95, 220, 75]) * a).astype(np.uint8)
        if settings.mask_boundary:
            contours, _ = cv2.findContours(
                (mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, contours, -1, (100, 235, 120), 1, cv2.LINE_AA)
    return frame
