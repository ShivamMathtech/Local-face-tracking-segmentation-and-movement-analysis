from pathlib import Path
import csv
import json
import zipfile
import cv2
import numpy as np

FIELDS = [
    "timestamp",
    "frame",
    "face_id",
    "status",
    "observed",
    "detection_count",
    "latency_ms",
    "x",
    "y",
    "width",
    "height",
    "cx",
    "cy",
    "dx",
    "dy",
    "distance",
    "total_distance",
    "vx",
    "vy",
    "speed",
    "ax",
    "ay",
    "acceleration",
    "direction",
    "area",
    "aspect_ratio",
    "relative_distance",
    "segment",
]


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        raise ValueError("Choose PNG or JPEG")
    ok, data = cv2.imencode(ext, image)
    if not ok:
        raise OSError("Unable to encode image")
    data.tofile(str(path))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def graph_export(path, rows):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(14, 9), facecolor="#101a27")
    FigureCanvasAgg(fig)
    choices = [
        ("cx", "X position (px)"),
        ("cy", "Y position (px)"),
        ("vx", "Velocity X (px/s)"),
        ("vy", "Velocity Y (px/s)"),
        ("speed", "Speed (px/s)"),
        ("acceleration", "Acceleration (px/s²)"),
        ("area", "Face area (px²)"),
        ("width", "Width / height (px)"),
    ]
    observed = [r for r in rows if r.get("cx") is not None]
    for idx, (key, title) in enumerate(choices):
        ax = fig.add_subplot(4, 2, idx + 1)
        ax.set_facecolor("#101a27")
        groups = {}
        for r in observed:
            groups.setdefault((r.get("face_id"), r.get("segment", 0)), []).append(r)
        for group in groups.values():
            t = [r["timestamp"] for r in group]
            ax.plot(t, [r[key] for r in group], color="#5cdfb7", lw=1.5)
            if key == "width":
                ax.plot(t, [r["height"] for r in group], color="#78a9ff", lw=1.2)
        ax.set_title(title, color="white", fontsize=10)
        ax.tick_params(colors="#a7b6c9", labelsize=8)
        ax.grid(alpha=0.12)
        ax.set_xlabel("Source time (s)", color="#a7b6c9", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def export_analysis(folder, pipeline, result):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = list(pipeline.measurements)
    write_csv(folder / "tracking_data.csv", rows)
    (folder / "tracking_data.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    (folder / "statistics.json").write_text(
        json.dumps(pipeline.summary(), indent=2, allow_nan=False), encoding="utf-8"
    )
    save_image(folder / "current_frame.png", result["image"])
    save_image(folder / "face_mask.png", result["mask"])
    save_image(folder / "heatmap.png", result["heatmap"])
    canvas = np.full_like(result["image"], (28, 22, 18))
    from face_motion_lab.visualization.overlays import draw_overlays
    from dataclasses import replace

    settings = replace(
        pipeline.s,
        boxes=False,
        grid=True,
        coordinates=False,
        head_pose=False,
        movement_vector=False,
        trajectory=True,
    )
    draw_overlays(canvas, [], None, settings, pipeline.analysis.trail, None, "TRAJECTORY")
    save_image(folder / "trajectory.png", canvas)
    graph_export(folder / "graphs.png", rows)
    landmarks = []
    for t in result["tracks"]:
        if t.observed and t.landmarks is not None:
            landmarks.append(
                {
                    "face_id": t.face_id,
                    "kind": t.landmarks.kind,
                    "points": t.landmarks.points.tolist(),
                    "normalized_xy": t.landmarks.normalized(
                        result["raw"].shape[1], result["raw"].shape[0]
                    ).tolist(),
                }
            )
    (folder / "landmarks.json").write_text(json.dumps(landmarks, indent=2), encoding="utf-8")
    (folder / "export_info.json").write_text(
        json.dumps(
            {
                "retained_rows": len(rows),
                "history_limit": pipeline.s.history_limit,
                "note": "CSV/JSON contain the retained in-memory window; recording streams every processed row to disk.",
                "coordinate_units": "processed-frame pixels; CSV uses top-left origin",
                "mask_method": result["mask_label"],
            },
            indent=2,
        )
    )
    return folder


def zip_session(folder, destination):
    folder = Path(folder).resolve()
    destination = Path(destination).resolve()
    if folder == destination or folder in destination.parents:
        raise ValueError("Save the ZIP outside its source session folder")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and not p.is_symlink():
                z.write(p, p.relative_to(folder.parent))
