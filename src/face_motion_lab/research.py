"""Headless algorithm comparisons on identical source frames."""

from dataclasses import replace
from pathlib import Path
import csv
import json
import time
from .pipeline import Pipeline
from .input.frame_source import FrameSource
from .input.synthetic import SyntheticDetector


def compare(source, settings, output, max_frames=300):
    reports = []
    for name, kalman, flow in [
        ("detector-only", False, False),
        ("detector+kalman", True, False),
        ("detector+flow", False, True),
        ("detector+kalman+flow", True, True),
    ]:
        s = replace(settings, kalman=kalman, optical_flow=flow)
        reader = FrameSource(source, s)
        pipeline = None
        elapsed = []
        detections = 0
        try:
            pipeline = Pipeline(s, SyntheticDetector() if source == "demo" else None)
            for _ in range(max_frames):
                f = reader.read()
                if f is None:
                    break
                start = time.perf_counter()
                result = pipeline.process(f.image, f.timestamp, f.number)
                elapsed.append(time.perf_counter() - start)
                detections += result["detection_count"]
                if pipeline.tracker.locked_id is None:
                    visible = [t for t in result["tracks"] if t.observed]
                    if visible:
                        pipeline.lock(max(visible, key=lambda t: t.bbox.area).face_id)
            summary = pipeline.summary()
            reports.append(
                {
                    "algorithm": name,
                    "detector": pipeline.detector.name,
                    "frames": len(elapsed),
                    "processing_fps": len(elapsed) / sum(elapsed) if elapsed else 0,
                    "mean_latency_ms": 1000 * sum(elapsed) / len(elapsed) if elapsed else 0,
                    "continuity_percent": summary["continuity_percent"],
                    "lost_frames": summary["frames_lost"],
                    "detection_count": detections,
                }
            )
        finally:
            reader.close()
            if pipeline:
                pipeline.close()
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "comparison.json").write_text(json.dumps(reports, indent=2))
    with (folder / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reports[0].keys())
        writer.writeheader()
        writer.writerows(reports)
    (folder / "report.md").write_text(
        "# Tracker experiment\n\n"
        "Same source replayed for each algorithm. First largest visible face is explicitly auto-selected for this benchmark. "
        "No ground-truth identities are available: continuity measures observed-frame availability, not identity accuracy. "
        "The demo uses scripted detections and is not a detector accuracy benchmark.\n\n"
        + "| Algorithm | FPS | Latency ms | Continuity % | Lost |\n|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {r['algorithm']} | {r['processing_fps']:.2f} | {r['mean_latency_ms']:.2f} | {r['continuity_percent']} | {r['lost_frames']} |"
            for r in reports
        )
    )
    return reports
