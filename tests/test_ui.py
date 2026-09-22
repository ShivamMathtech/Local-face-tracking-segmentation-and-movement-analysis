"""Qt offscreen integration: worker starts, lock affects results, close joins thread."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import time
from PySide6.QtWidgets import QApplication
from face_motion_lab.config.settings import Settings
from face_motion_lab.ui.main_window import MainWindow


def wait_for(app, predicate, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("UI operation timed out")


def test_dashboard_demo_lock_and_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "save", lambda self, *args: None)
    app = QApplication.instance() or QApplication([])
    w = MainWindow(Settings())
    w.show()
    w.start_source("demo")
    try:
        wait_for(app, lambda: w.last is not None and w.faces.count() == 2)
        w.faces.setCurrentRow(0)
        w.lock_selected()
        wait_for(app, lambda: w.last["locked_id"] == 1 and w.last["metrics"] is not None)
        assert w.last["status"] == "LOCKED"
        w.set_option("view_mode", "Thermal Style")
        w.command("save_frame", str(tmp_path / "frame.png"))
        wait_for(app, lambda: (tmp_path / "frame.png").exists())
        w.stop_source()
        wait_for(app, lambda: not w.worker.isRunning())
    finally:
        w.stop_source()
        if w.worker:
            w.worker.wait(10000)
        w.close()
        app.processEvents()
