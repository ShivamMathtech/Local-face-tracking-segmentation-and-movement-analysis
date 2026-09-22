import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError
from face_motion_lab.landmarks.face_landmarks import FEATURES, OVAL

GREEN = (115, 235, 85)
BLUE = (230, 190, 90)
LOST = (50, 165, 240)


def draw_landmarks(frame, landmarks, style):
    pts = np.rint(landmarks.points[:, :2]).astype(np.int32)
    dense = landmarks.kind == "dense"
    if style == "points" or not dense:
        for p in pts:
            cv2.circle(frame, tuple(p), 1, GREEN, -1, cv2.LINE_AA)
    elif style == "lines":
        for group in FEATURES:
            cv2.polylines(frame, [pts[group]], False, GREEN, 1, cv2.LINE_AA)
    elif style == "contour":
        cv2.polylines(frame, [pts[OVAL]], True, GREEN, 1, cv2.LINE_AA)
    elif style == "mesh":
        # A geometric Delaunay mesh, not the canonical MediaPipe topology.
        try:
            for tri in Delaunay(pts[:468]).simplices:
                cv2.polylines(frame, [pts[tri]], True, (85, 150, 110), 1, cv2.LINE_AA)
        except QhullError:
            cv2.polylines(frame, [pts[OVAL]], True, GREEN, 1)


def draw_overlays(frame, tracks, locked_id, settings, trail, metrics, status, pose=None):
    h, w = frame.shape[:2]
    if settings.grid:
        for x in range(0, w, 100):
            cv2.line(frame, (x, 0), (x, h), (65, 65, 65), 1)
        for y in range(0, h, 100):
            cv2.line(frame, (0, y), (w, y), (65, 65, 65), 1)
    if settings.trajectory and len(trail) > 1:
        prev = None
        for _, x, y, segment, _ in trail:
            current = (round(x), round(y))
            if prev is not None and segment == prev[1]:
                cv2.line(frame, prev[0], current, (190, 200, 55), 2, cv2.LINE_AA)
            prev = (current, segment)
    for t in tracks:
        if not t.observed and t.face_id != locked_id:
            continue
        color = GREEN if t.face_id == locked_id and t.observed else (BLUE if t.observed else LOST)
        x, y, bw, bh = t.bbox.ints()
        cx, cy = map(round, t.bbox.center)
        if settings.boxes:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2 if t.face_id == locked_id else 1)
            confidence = "n/a" if t.confidence is None else f"{t.confidence:.2f}"
            label = f"#{t.face_id:02d}  {'LOCKED' if t.face_id == locked_id else 'FACE'}  conf {confidence}"
            if not t.observed:
                label = f"#{t.face_id:02d}  PREDICTED / LOST"
            cv2.putText(frame, label, (max(0, x), max(18, y - 8)), 0, 0.48, color, 1, cv2.LINE_AA)
        if settings.coordinates:
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 18, 1)
        if settings.landmarks and t.observed and t.landmarks is not None:
            draw_landmarks(frame, t.landmarks, settings.landmark_style)
    if settings.movement_vector and metrics is not None:
        start = (round(metrics.cx - metrics.dx), round(metrics.cy - metrics.dy))
        end = (round(metrics.cx), round(metrics.cy))
        cv2.arrowedLine(frame, start, end, (70, 120, 255), 3, cv2.LINE_AA, tipLength=0.4)
    if pose and settings.head_pose:
        axis = np.rint(pose["axes"]).astype(int)
        for p, c in zip(axis[1:], [(70, 70, 240), (70, 240, 70), (240, 130, 70)]):
            cv2.line(frame, tuple(axis[0]), tuple(p), c, 2)
    banner = f"{status}   target: {locked_id if locked_id is not None else '--'}"
    if metrics:
        banner += f"   {metrics.direction}   {metrics.speed:.1f} px/s"
    cv2.rectangle(frame, (0, 0), (w, 30), (22, 22, 22), -1)
    cv2.putText(
        frame, banner, (12, 21), 0, 0.5, GREEN if status == "LOCKED" else (210, 210, 210), 1, cv2.LINE_AA
    )
    notice = {
        "Night Vision": "SIMULATED NIGHT VISION - RGB input",
        "Thermal Style": "THERMAL-STYLE / SIMULATED - no temperature data",
    }.get(settings.view_mode)
    if notice:
        cv2.rectangle(frame, (0, h - 30), (w, h), (22, 22, 22), -1)
        cv2.putText(frame, notice, (10, h - 10), 0, 0.5, (100, 220, 255), 1, cv2.LINE_AA)
    return frame
