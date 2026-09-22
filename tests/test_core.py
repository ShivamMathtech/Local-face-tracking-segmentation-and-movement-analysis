from dataclasses import replace
import csv
import json
import math
import numpy as np
import cv2
import pytest
from face_motion_lab.types import Box, FaceDetection, Landmarks
from face_motion_lab.config.settings import Settings
from face_motion_lab.utils.geometry import iou
from face_motion_lab.analysis.coordinates import coordinates
from face_motion_lab.analysis.movement import MovementAnalyzer, direction
from face_motion_lab.analysis.head_pose import estimate_pose, MODEL, INDICES
from face_motion_lab.tracking.kalman_tracker import KalmanBox
from face_motion_lab.tracking.face_tracker import FaceTracker
from face_motion_lab.tracking.optical_flow import estimate_flow
from face_motion_lab.segmentation.face_mask import face_mask
from face_motion_lab.visualization.filters import transform, MODES, sharpness
from face_motion_lab.input.synthetic import demo_frame, SyntheticDetector
from face_motion_lab.pipeline import Pipeline
from face_motion_lab.recording.exporter import write_csv, save_image, zip_session, export_analysis
from face_motion_lab.recording.session_logger import SessionRecorder


def test_box_geometry():
    box = Box(-10, -10, 30, 40)
    assert box.center == (5, 10)
    assert box.clipped(15, 15) == Box(0, 0, 15, 15)
    assert iou(Box(0, 0, 10, 10), Box(5, 0, 10, 10)) == pytest.approx(1 / 3)
    assert Box(0, 0, 10, 10).contains(5, 5)


@pytest.mark.parametrize(
    "origin,expected",
    [
        ("top-left", (75, 25, 0.75, 0.25)),
        ("center", (25, -25, 0.25, -0.25)),
        ("cartesian", (25, 25, 0.25, 0.25)),
    ],
)
def test_coordinates(origin, expected):
    r = coordinates(75, 25, 100, 100, origin)
    assert tuple(r[k] for k in ["x", "y", "x_norm", "y_norm"]) == expected


@pytest.mark.parametrize(
    "vx,vy,name",
    [
        (0, 0, "STATIONARY"),
        (10, 0, "RIGHT"),
        (-10, 0, "LEFT"),
        (0, 10, "DOWN"),
        (0, -10, "UP"),
        (10, 10, "DOWN-RIGHT"),
        (-10, -10, "UP-LEFT"),
        (-10, 10, "DOWN-LEFT"),
        (10, -10, "UP-RIGHT"),
    ],
)
def test_direction(vx, vy, name):
    assert direction(vx, vy, 3) == name


def test_timestamp_velocity_acceleration_and_gap():
    a = MovementAnalyzer(Settings(smoothing_factor=1))
    a.update(Box(0, 0, 10, 10), 0, 0, 1)
    one = a.update(Box(10, 0, 10, 10), 0.5, 1, 1)
    two = a.update(Box(40, 0, 10, 10), 1.5, 2, 1)
    assert one.vx == 20 and two.vx == 30 and two.ax == 10
    assert two.total_distance == 40 and two.direction == "RIGHT"
    assert a.update(Box(45, 0, 10, 10), 1.5, 2, 1) is None
    a.gap()
    three = a.update(Box(300, 0, 10, 10), 2.5, 3, 1)
    assert three.vx == 0 and three.total_distance == 40 and three.segment == 1


def test_bounded_trail():
    a = MovementAnalyzer(Settings(trajectory_length=10))
    for i in range(30):
        a.update(Box(i, 0, 10, 10), i, i, 1)
    assert len(a.trail) == 10
    assert a.summary()["maximum_speed"] > 0


def test_kalman_prediction_and_covariance():
    k = KalmanBox(Box(0, 0, 10, 10))
    for i in range(1, 30):
        k.predict(0.1)
        k.correct(Box(i * 2, 0, 10, 10))
    x = k.box.center[0]
    k.predict(0.1)
    assert k.box.center[0] > x
    assert np.all(np.linalg.eigvalsh(k.P) > 0)


def scene(boxes, colors=None):
    frame = np.zeros((160, 260, 3), np.uint8)
    colors = colors or [(40, 150, 220)] * len(boxes)
    for b, c in zip(boxes, colors):
        x, y, w, h = b.ints()
        frame[y : y + h, x : x + w] = c
    return frame


def test_lock_never_silently_transfers_after_expiry():
    s = Settings(max_lost_frames=2, optical_flow=False)
    t = FaceTracker(s)
    a = Box(20, 30, 40, 50)
    b = Box(170, 30, 40, 50)
    frame = scene([a, b], [(20, 30, 210), (210, 30, 20)])
    t.update(frame, [FaceDetection(a, 0.9), FaceDetection(b, 0.9)], 0)
    t.lock(1)
    for i in range(1, 5):
        t.update(scene([b], [(210, 30, 20)]), [FaceDetection(b, 0.9)], i / 30)
    assert t.locked_id == 1 and t.status == "REACQUIRING"
    tracks = t.update(frame, [FaceDetection(a, 0.9), FaceDetection(b, 0.9)], 0.2)
    assert t.locked_id == 1 and t.status == "REACQUIRING"
    new = next(r for r in tracks if r.observed and r.bbox.x < 100)
    assert new.face_id != 1
    t.lock(new.face_id)
    assert t.status == "LOCKED"


def test_short_gap_reacquisition_same_id():
    t = FaceTracker(Settings(optical_flow=False, max_lost_frames=5))
    b = Box(40, 30, 40, 50)
    frame = scene([b])
    t.update(frame, [FaceDetection(b, 0.9)], 0)
    t.lock(1)
    t.update(frame, [], 0.03)
    assert t.status == "TEMPORARILY_LOST"
    t.update(frame, [FaceDetection(b, 0.9)], 0.06)
    assert t.locked_id == 1 and t.status == "LOCKED" and t.reacquisitions == 1


def test_ambiguous_match_suspends_lock():
    t = FaceTracker(Settings(optical_flow=False))
    b = Box(80, 40, 40, 50)
    frame = scene([Box(60, 30, 100, 80)])
    t.update(frame, [FaceDetection(b, 0.9)], 0)
    t.lock(1)
    t.update(frame, [FaceDetection(Box(75, 40, 40, 50), 0.9), FaceDetection(Box(85, 40, 40, 50), 0.9)], 0.03)
    assert t.status == "TEMPORARILY_LOST"


def test_optical_flow_translation():
    rng = np.random.default_rng(4)
    gray = rng.integers(0, 255, (140, 140), dtype=np.uint8)
    moved = cv2.warpAffine(gray, np.float32([[1, 0, 3], [0, 1, 2]]), (140, 140))
    delta = estimate_flow(gray, moved, Box(25, 25, 70, 70))
    assert delta is not None and np.allclose(delta, [3, 2], atol=0.5)


@pytest.mark.parametrize("method", ["ellipse", "polygon", "hull", "refined", "skin"])
def test_mask_bounds(method):
    image = scene([Box(20, 30, 70, 90)])
    mask, label = face_mask(image, Box(20, 30, 70, 90), None, method)
    assert mask.dtype == np.uint8 and mask.shape == image.shape[:2]
    assert mask[0, 0] == 0 and "ellipse" in label


@pytest.mark.parametrize("mode", MODES)
def test_filter_does_not_mutate_source(mode):
    image = demo_frame(0)
    original = image.copy()
    mask = np.zeros(image.shape[:2], np.uint8)
    result = transform(image, Settings(view_mode=mode), mask, Box(10, 10, 30, 30), image.copy())
    assert np.array_equal(image, original)
    assert result.shape == image.shape and result.dtype == np.uint8


def test_sharpness_response():
    image = demo_frame(0)
    box = Box(0, 0, 800, 480)
    assert (
        sharpness(image, box)["sharpness"] > sharpness(cv2.GaussianBlur(image, (31, 31), 8), box)["sharpness"]
    )


def test_head_pose_projected_reference():
    k = np.array([[800, 0, 400], [0, 800, 240], [0, 0, 1]], float)
    xy, _ = cv2.projectPoints(MODEL, np.zeros(3), np.array([0.0, 0.0, 1500.0]), k, np.zeros(5))
    points = np.zeros((478, 3))
    points[INDICES, :2] = xy.reshape(-1, 2)
    pose = estimate_pose(Landmarks(points), (480, 800, 3))
    assert abs(pose["yaw"]) < 1 and abs(pose["pitch"]) < 1 and not pose["calibrated"]
    assert estimate_pose(Landmarks(points, "sparse corners"), (480, 800, 3)) is None


def test_settings_roundtrip_and_rejections(tmp_path):
    path = tmp_path / "settings.json"
    Settings().save(path)
    assert Settings.load(path) == Settings()
    for kwargs in [
        {"smoothing_factor": 0},
        {"mask_opacity": 2},
        {"fps": float("nan")},
        {"width": "abc"},
        {"mask": 1},
    ]:
        with pytest.raises(ValueError):
            Settings(**kwargs).validate()
    path.write_text('{"unknown": 3}')
    with pytest.raises(ValueError):
        Settings.load(path)


def test_detector_fallback_no_false_confidence():
    from face_motion_lab.detection.opencv_detector import HaarDetector

    detector = HaarDetector()
    result = detector.detect(np.zeros((240, 320, 3), np.uint8))
    assert result == []
    detector.close()


def test_end_to_end_export_and_record(tmp_path):
    p = Pipeline(Settings(), SyntheticDetector())
    recorder = None
    try:
        for i in range(15):
            result = p.process(demo_frame(i), i / 30, i)
            if i == 0:
                p.lock(1)
                recorder = SessionRecorder(
                    tmp_path, Settings(), "synthetic demo", result["image"].shape, 30, p.detector.name
                )
            recorder.write(result)
        recorder.close(p.summary())
        folder = export_analysis(tmp_path / "analysis", p, result)
        assert (folder / "graphs.png").is_file()
        rows = list(csv.DictReader((folder / "tracking_data.csv").open()))
        assert len(rows) == 15 and rows[-1]["face_id"] == "1"
        assert json.loads((folder / "statistics.json").read_text())["frames_tracked"] == 15
        cap = cv2.VideoCapture(str(recorder.folder / "processed_video.mp4"))
        ok, frame = cap.read()
        cap.release()
        assert ok and frame.shape[:2] == (480, 800)
        zip_session(recorder.folder, tmp_path / "session.zip")
        assert (tmp_path / "session.zip").is_file()
        with pytest.raises(ValueError):
            zip_session(recorder.folder, recorder.folder / "recursive.zip")
    finally:
        if recorder:
            recorder.close()
        p.close()


def test_source_video_timestamps_and_invalid(tmp_path):
    from face_motion_lab.input.frame_source import FrameSource

    path = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 120))
    for i in range(5):
        writer.write(np.full((120, 160, 3), i * 30, np.uint8))
    writer.release()
    source = FrameSource(str(path), Settings())
    frames = []
    while (frame := source.read()) is not None:
        frames.append(frame)
    source.close()
    assert len(frames) == 5
    assert all(b.timestamp > a.timestamp for a, b in zip(frames, frames[1:]))
    with pytest.raises(OSError):
        FrameSource(str(tmp_path / "missing.mp4"), Settings())


def test_still_image_lock_measurement_export(tmp_path):
    p = Pipeline(Settings(), SyntheticDetector())
    try:
        p.process(demo_frame(0), 0.0, 0)
        p.lock(1)
        rendered = p.render()
        assert rendered["metrics"]["speed"] == 0
        export_analysis(tmp_path, p, rendered)
        rows = list(csv.DictReader((tmp_path / "tracking_data.csv").open()))
        assert len(rows) == 1
        assert rows[0]["status"] == "LOCKED" and float(rows[0]["cx"]) > 0
    finally:
        p.close()
