"""One processing owner, command queue, latest-result mailbox: no UI frame backlog."""

from dataclasses import replace
from queue import Queue, Empty
from threading import Event, Lock
import time
import logging
import psutil
from PySide6.QtCore import QThread, Signal
from face_motion_lab.pipeline import Pipeline
from face_motion_lab.input.frame_source import FrameSource
from face_motion_lab.input.synthetic import SyntheticDetector
from face_motion_lab.recording.session_logger import SessionRecorder
from face_motion_lab.recording.exporter import export_analysis, save_image


class Worker(QThread):
    error = Signal(str)
    notice = Signal(str)

    def __init__(self, source, settings, parent=None):
        super().__init__(parent)
        self.source_name = source
        self.settings = replace(settings)
        self.commands = Queue()
        self.stop_event = Event()
        self.mail_lock = Lock()
        self.latest = None

    def submit(self, action, value=None):
        self.commands.put((action, value))

    def stop(self):
        self.stop_event.set()

    def take(self):
        with self.mail_lock:
            value = self.latest
            self.latest = None
        return value

    def publish(self, result, recording, paused):
        if result is None:
            return
        result = dict(result)
        result["recording"] = str(recording.folder) if recording else None
        result["paused"] = paused
        result["memory_mb"] = self.process.memory_info().rss / 1048576
        result["cpu_percent"] = self.process.cpu_percent() / max(1, psutil.cpu_count())
        with self.mail_lock:
            self.latest = result

    def run(self):
        source = pipeline = recorder = None
        paused = False
        exhausted = False
        last = None
        failures = 0
        try:
            self.process = psutil.Process()
            self.process.cpu_percent()
            source = FrameSource(self.source_name, self.settings)
            pipeline = Pipeline(self.settings, SyntheticDetector() if self.source_name == "demo" else None)
            self.notice.emit(f"Source open: {source.kind} | {pipeline.detector.name}")
            while not self.stop_event.is_set():
                cycle = time.monotonic()
                dirty = False
                while True:
                    try:
                        action, value = self.commands.get_nowait()
                    except Empty:
                        break
                    try:
                        if action == "lock":
                            pipeline.lock(int(value))
                            dirty = True
                        elif action == "unlock":
                            pipeline.unlock()
                            dirty = True
                        elif action == "reset":
                            pipeline.reset()
                            if source.demo:
                                pipeline.detector.number = source.number
                            dirty = True
                        elif action == "pause":
                            if recorder and not paused:
                                recorder.close(pipeline.summary())
                                self.notice.emit("Recording stopped before pause")
                                recorder = None
                            paused = not paused
                            pipeline.analysis.gap()
                            dirty = True
                        elif action == "setting":
                            name, val = value
                            candidate = replace(self.settings, **{name: val}).validate()
                            setattr(self.settings, name, getattr(candidate, name))
                            if name == "trajectory_length":
                                from collections import deque

                                pipeline.analysis.trail = deque(pipeline.analysis.trail, maxlen=val or None)
                            dirty = True
                        elif action == "refresh":
                            if pipeline.frame is not None:
                                if source.demo:
                                    pipeline.detector.number = pipeline.number
                                last = pipeline.process(pipeline.frame, pipeline.timestamp, pipeline.number)
                                dirty = True
                        elif action == "record":
                            if pipeline.frame is None:
                                raise ValueError("Open a source before recording")
                            if source.kind == "image":
                                raise ValueError("Use Save Frame for a still image")
                            if paused or exhausted:
                                raise ValueError("Resume a running source before recording")
                            if recorder is None:
                                recorder = SessionRecorder(
                                    value,
                                    self.settings,
                                    source.kind,
                                    pipeline.frame.shape,
                                    source.fps,
                                    pipeline.detector.name,
                                )
                                self.notice.emit(f"Recording: {recorder.folder}")
                            dirty = True
                        elif action == "stop_recording":
                            if recorder:
                                folder = recorder.folder
                                recorder.close(pipeline.summary())
                                recorder = None
                                self.notice.emit(f"Session saved: {folder}")
                            dirty = True
                        elif action == "save_frame":
                            if last is None:
                                raise ValueError("No current frame")
                            save_image(value, last["image"])
                            self.notice.emit(f"Frame saved: {value}")
                        elif action == "export":
                            if pipeline.frame is None:
                                raise ValueError("No analysis to export")
                            export_analysis(value, pipeline, pipeline.render())
                            self.notice.emit(f"Analysis exported: {value}")
                    except Exception as exc:
                        self.error.emit(str(exc))
                if dirty:
                    rendered = pipeline.render()
                    if rendered is not None:
                        last = rendered
                        self.publish(last, recorder, paused or exhausted)
                if paused or exhausted:
                    self.stop_event.wait(0.025)
                    continue
                frame = source.read()
                if frame is None:
                    exhausted = True
                    if recorder:
                        folder = recorder.folder
                        recorder.close(pipeline.summary())
                        recorder = None
                        self.notice.emit(f"Recording saved: {folder}")
                    self.notice.emit(
                        "Image loaded"
                        if source.kind == "image"
                        else ("Stream ended / source unavailable" if source.live else "Playback complete")
                    )
                    if last:
                        self.publish(last, recorder, True)
                    continue
                try:
                    last = pipeline.process(frame.image, frame.timestamp, frame.number)
                    failures = 0
                    if recorder:
                        try:
                            recorder.write(last)
                        except Exception as exc:
                            recorder.close(pipeline.summary())
                            recorder = None
                            self.error.emit(f"Recording stopped: {exc}")
                    self.publish(last, recorder, False)
                except Exception as exc:
                    failures += 1
                    pipeline.analysis.gap()
                    self.error.emit(f"Frame processing failed: {exc}")
                    if failures >= 5:
                        exhausted = True
                        self.notice.emit("Processing paused after repeated errors. Reopen the source.")
                # Pace files/demo for playback; live sources supply acquisition timestamps.
                if not source.live:
                    self.stop_event.wait(max(0, 1 / source.fps - (time.monotonic() - cycle)))
        except Exception as exc:
            logging.getLogger(__name__).error("Source worker failed: %s", type(exc).__name__)
            self.error.emit(str(exc))
        finally:
            if recorder:
                recorder.close(pipeline.summary() if pipeline else None)
            if pipeline:
                pipeline.close()
            if source:
                source.close()
