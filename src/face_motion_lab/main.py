"""Desktop launcher and reproducible headless research commands."""

from pathlib import Path
import argparse
import json
import sys
from .config.settings import Settings


def parser():
    p = argparse.ArgumentParser(description="Face Motion Lab: local face tracking and movement analysis")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, help="Camera device index")
    src.add_argument("--video", help="Video path")
    src.add_argument("--image", help="Image path")
    src.add_argument("--rtsp", help="RTSP/IP stream URL (prefer UI to avoid shell history)")
    src.add_argument(
        "--demo", action="store_true", help="Synthetic demo with scripted boxes, no model required"
    )
    p.add_argument("--config", type=Path, help="Settings YAML or JSON")
    p.add_argument("--detector", choices=["haar", "mediapipe", "yunet"])
    p.add_argument("--detector-model")
    p.add_argument("--landmark-model")
    p.add_argument("--headless", action="store_true", help="Process without Qt and export analytics")
    p.add_argument(
        "--auto-lock", action="store_true", help="Explicitly lock first largest visible face in headless run"
    )
    p.add_argument(
        "--record", action="store_true", help="Explicitly record headless processed video and analytics"
    )
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--output", default="data/output/run")
    p.add_argument("--compare", action="store_true", help="Compare four tracker settings on a video/demo")
    p.add_argument("--download-model", choices=["landmarker", "detector", "yunet"])
    p.add_argument("--model-dir", default="models")
    p.add_argument("--sha256")
    p.add_argument("--calibrate", help="Folder of chessboard images")
    p.add_argument("--board-columns", type=int, default=9)
    p.add_argument("--board-rows", type=int, default=6)
    p.add_argument("--square-size", type=float, default=25.0)
    p.add_argument("--debug", action="store_true")
    return p


def run_headless(source, settings, args):
    from .pipeline import Pipeline
    from .input.frame_source import FrameSource
    from .input.synthetic import SyntheticDetector
    from .recording.session_logger import SessionRecorder
    from .recording.exporter import export_analysis

    reader = FrameSource(source, settings)
    pipeline = None
    recorder = None
    result = None
    try:
        pipeline = Pipeline(settings, SyntheticDetector() if source == "demo" else None)
        for _ in range(args.max_frames):
            frame = reader.read()
            if frame is None:
                break
            result = pipeline.process(frame.image, frame.timestamp, frame.number)
            if args.auto_lock and pipeline.tracker.locked_id is None:
                visible = [t for t in result["tracks"] if t.observed]
                if visible:
                    pipeline.lock(max(visible, key=lambda t: t.bbox.area).face_id)
                    refreshed = pipeline.render()
                    refreshed["row"] = result["row"]
                    result = refreshed
            if args.record:
                if recorder is None:
                    recorder = SessionRecorder(
                        args.output,
                        settings,
                        reader.kind,
                        result["image"].shape,
                        reader.fps,
                        pipeline.detector.name,
                    )
                recorder.write(result)
        if result is None:
            raise RuntimeError("No frames were decoded")
        export_analysis(args.output, pipeline, result)
        print(json.dumps(pipeline.summary(), indent=2))
        return 0
    finally:
        if recorder:
            recorder.close(pipeline.summary() if pipeline else None)
        if pipeline:
            pipeline.close()
        reader.close()


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    from .utils.logging import setup_logging

    setup_logging(args.debug)
    try:
        if args.max_frames < 1:
            p.error("--max-frames must be positive")
        if args.download_model:
            from .models.model_manager import download_model

            path, digest = download_model(args.download_model, args.model_dir, args.sha256)
            print(f"Saved {path}\nSHA-256 {digest}")
            return 0
        if args.calibrate:
            from .analysis.calibration import calibrate

            paths = [
                p
                for p in Path(args.calibrate).iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")
            ]
            data = calibrate(paths, args.board_columns, args.board_rows, args.square_size)
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=True)
            (out / "calibration.json").write_text(json.dumps(data, indent=2))
            print(out / "calibration.json")
            return 0
        settings = Settings.load(args.config) if args.config else Settings.load()
        for name in ("detector", "detector_model", "landmark_model"):
            if getattr(args, name) is not None:
                setattr(settings, name, getattr(args, name))
        if args.camera is not None:
            settings.camera_index = args.camera
        settings.validate()
        source = (
            "demo"
            if args.demo
            else (args.camera if args.camera is not None else (args.video or args.image or args.rtsp))
        )
        if args.compare:
            if source is None or isinstance(source, int) or args.rtsp or args.image:
                p.error("--compare requires --video or --demo")
            from .research import compare

            print(json.dumps(compare(source, settings, args.output, args.max_frames), indent=2))
            return 0
        if args.headless:
            if source is None:
                p.error("--headless requires an input source")
            return run_headless(source, settings, args)
        from PySide6.QtWidgets import QApplication
        from .ui.main_window import MainWindow

        app = QApplication(sys.argv[:1])
        app.setApplicationName("Face Motion Lab")
        window = MainWindow(settings, source)
        window.show()
        return app.exec()
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        print(f"Face Motion Lab: {exc}", file=sys.stderr)
        return 2
