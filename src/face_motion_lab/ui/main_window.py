from pathlib import Path
from datetime import datetime
import time
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QInputDialog,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QTabWidget,
    QFormLayout,
    QCheckBox,
    QScrollArea,
    QSplitter,
    QMessageBox,
    QDoubleSpinBox,
)
from .camera_panel import CameraPanel
from .graph_panel import GraphPanel
from .settings_panel import SettingsDialog
from .sessions import SessionsDialog
from .worker import Worker
from face_motion_lab.visualization.filters import MODES

STYLE = """
QWidget {background:#0b1320;color:#d6e3f1;font-family:'Segoe UI','DejaVu Sans';font-size:12px;}
QMainWindow {background:#0b1320;} QLabel#brand {font-size:23px;font-weight:700;letter-spacing:2px;color:#f2f7fb;}
QLabel#sub {color:#8095af;font-size:11px;} QLabel#badge {color:#5cddb6;font-weight:700;}
QPushButton {background:#1a2b3e;border:1px solid #2b435b;border-radius:5px;padding:8px 12px;}
QPushButton:hover {background:#254159;border-color:#58ddb8;} QPushButton:disabled {color:#506277;}
QPushButton#primary {background:#218568;color:white;border:0;font-weight:600;}
QComboBox,QLineEdit,QSpinBox,QDoubleSpinBox,QListWidget {background:#101e2e;border:1px solid #263b51;border-radius:4px;padding:5px;}
QListWidget::item {padding:8px;} QListWidget::item:selected {background:#1c5b54;color:#b8ffdf;}
QTabBar::tab {padding:9px;background:#101e2e;color:#96abc1;} QTabBar::tab:selected {color:#5cddb6;border-bottom:2px solid #5cddb6;}
QTabWidget::pane {border:1px solid #213448;} QScrollArea {border:0;}
QSlider::groove:horizontal {height:4px;background:#294057;} QSlider::handle:horizontal {background:#5cddb6;width:12px;margin:-5px 0;border-radius:6px;}
QCheckBox {padding:3px;} QSplitter::handle {background:#25384a;height:4px;width:4px;}
QStatusBar {background:#101d2a;color:#9db2c5;} QToolTip {color:#e4f1fc;background:#234058;border:1px solid #5cddb6;}
"""


class MainWindow(QMainWindow):
    def __init__(self, settings, source=None):
        super().__init__()
        self.s = settings
        self.worker = None
        self.last = None
        self.last_graph = 0.0
        self.pending_source = None
        self.session_root = Path.cwd() / "data" / "sessions"
        self.setWindowTitle("Face Motion Lab · MathTech")
        self.resize(1510, 980)
        self.setMinimumSize(1100, 740)
        self.setStyleSheet(STYLE)
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 15, 20, 8)
        header = QHBoxLayout()
        title = QLabel("FACE MOTION LAB")
        title.setObjectName("brand")
        header.addWidget(title)
        sub = QLabel("COMPUTER VISION  /  RESEARCH WORKBENCH")
        sub.setObjectName("sub")
        header.addWidget(sub)
        header.addStretch()
        self.badge = QLabel("●  SOURCE OFFLINE")
        self.badge.setObjectName("badge")
        header.addWidget(self.badge)
        outer.addLayout(header)
        toolbar = QHBoxLayout()
        outer.addLayout(toolbar)
        self.button(toolbar, "Start camera", lambda: self.start_source(self.s.camera_index), primary=True)
        self.button(toolbar, "Open image", lambda: self.open_file(True))
        self.button(toolbar, "Open video", lambda: self.open_file(False))
        self.button(toolbar, "IP / RTSP", self.open_stream)
        self.button(toolbar, "Demo", lambda: self.start_source("demo"))
        self.button(toolbar, "Pause / Resume", lambda: self.command("pause"))
        self.button(toolbar, "Stop", self.stop_source)
        toolbar.addStretch()
        self.button(toolbar, "Device settings", self.settings_dialog)
        vertical = QSplitter(Qt.Orientation.Vertical)
        outer.addWidget(vertical, 1)
        top = QSplitter(Qt.Orientation.Horizontal)
        vertical.addWidget(top)
        video_container = QWidget()
        vl = QVBoxLayout(video_container)
        vl.setContentsMargins(0, 0, 0, 0)
        self.camera = CameraPanel()
        self.camera.clicked.connect(self.select_at)
        vl.addWidget(self.camera, 1)
        video_bar = QHBoxLayout()
        vl.addLayout(video_bar)
        self.view = QComboBox()
        self.view.addItems(MODES)
        self.view.setCurrentText(self.s.view_mode)
        self.view.currentTextChanged.connect(lambda value: self.set_option("view_mode", value))
        video_bar.addWidget(QLabel("VIEW"))
        video_bar.addWidget(self.view)
        self.button(video_bar, "Save frame", self.save_frame)
        self.button(video_bar, "Export analysis", self.export)
        self.button(video_bar, "● Record", self.record)
        self.button(video_bar, "■ End recording", lambda: self.command("stop_recording"))
        top.addWidget(video_container)
        side = QTabWidget()
        side.setMinimumWidth(320)
        top.addWidget(side)
        top.setSizes([1060, 370])
        tracking = QWidget()
        tl = QVBoxLayout(tracking)
        self.lock_status = QLabel("UNLOCKED\nSelect a detected face, then Lock.")
        self.lock_status.setWordWrap(True)
        self.lock_status.setObjectName("badge")
        tl.addWidget(self.lock_status)
        self.faces = QListWidget()
        self.faces.setMaximumHeight(145)
        tl.addWidget(self.faces)
        locks = QHBoxLayout()
        tl.addLayout(locks)
        self.button(locks, "Lock selected", self.lock_selected, primary=True)
        self.button(locks, "Unlock", lambda: self.command("unlock"))
        self.button(locks, "Reset", lambda: self.command("reset"))
        self.button(tl, "Detect faces / refresh", lambda: self.command("refresh"))
        self.metrics = QLabel("Open a source to begin.")
        self.metrics.setWordWrap(True)
        self.metrics.setTextFormat(Qt.TextFormat.RichText)
        tl.addWidget(self.metrics)
        tl.addStretch()
        note = QLabel(
            "Track IDs are temporary. After sustained loss, select and lock again. A match score is not identity accuracy."
        )
        note.setObjectName("sub")
        note.setWordWrap(True)
        tl.addWidget(note)
        side.addTab(self.scroll(tracking), "Target")
        effects = QWidget()
        form = QFormLayout(effects)
        for key, label in [
            ("boxes", "Bounding boxes"),
            ("coordinates", "Center / coordinates"),
            ("grid", "Coordinate grid"),
            ("landmarks", "Landmarks / feature points"),
            ("trajectory", "Trajectory"),
            ("movement_vector", "Movement vector"),
            ("mask", "Mask overlay"),
            ("mask_fill", "Mask fill"),
            ("mask_boundary", "Mask boundary"),
            ("head_pose", "Head pose (dense model)"),
        ]:
            check = QCheckBox()
            check.setChecked(getattr(self.s, key))
            check.toggled.connect(lambda val, k=key: self.set_option(k, val))
            form.addRow(label, check)
        for key, label, options in [
            ("landmark_style", "Landmark style", ["points", "lines", "mesh", "contour"]),
            (
                "segmentation_mode",
                "Segmentation",
                [
                    "Original",
                    "Face Mask",
                    "Masked Face",
                    "Background Removed",
                    "Face Highlighted",
                    "Face Silhouette",
                ],
            ),
            ("mask_method", "Mask method", ["ellipse", "polygon", "hull", "refined", "skin"]),
            ("coordinate_origin", "Coordinate origin", ["top-left", "center", "cartesian"]),
            ("edge_scope", "Edge scope", ["Full frame", "Face ROI", "Segmented face"]),
            ("thermal_colormap", "Thermal-style colormap", ["INFERNO", "JET", "HOT", "TURBO"]),
        ]:
            combo = QComboBox()
            combo.addItems(options)
            combo.setCurrentText(getattr(self.s, key))
            combo.currentTextChanged.connect(lambda val, k=key: self.set_option(k, val))
            form.addRow(label, combo)
        trail = QComboBox()
        trail.addItems(["10", "25", "50", "100", "500", "Unlimited"])
        trail.setCurrentText(str(self.s.trajectory_length) if self.s.trajectory_length else "Unlimited")
        trail.currentTextChanged.connect(
            lambda val: self.set_option("trajectory_length", 0 if val == "Unlimited" else int(val))
        )
        form.addRow("Trail length", trail)
        for key, label, lo, hi, step in [
            ("mask_opacity", "Mask opacity", 0.0, 1.0, 0.05),
            ("brightness", "Night brightness", -100.0, 100.0, 5.0),
            ("contrast", "Night contrast", 0.0, 5.0, 0.1),
            ("gain", "Night gain", 0.0, 5.0, 0.1),
            ("noise", "Night noise", 0.0, 50.0, 1.0),
            ("intensity", "Night intensity", 0.0, 3.0, 0.1),
            ("heatmap_window", "Heatmap seconds", 1.0, 3600.0, 5.0),
            ("graph_window", "Graph seconds", 1.0, 3600.0, 5.0),
        ]:
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setValue(getattr(self.s, key))
            spin.valueChanged.connect(lambda val, k=key: self.set_option(k, val))
            form.addRow(label, spin)
        side.addTab(self.scroll(effects), "Display")
        analytics = QWidget()
        al = QVBoxLayout(analytics)
        self.statistics = QLabel("No measurements yet.")
        self.statistics.setWordWrap(True)
        al.addWidget(self.statistics)
        al.addStretch()
        self.button(al, "Recorded sessions", self.sessions)
        side.addTab(analytics, "Session")
        self.graphs = GraphPanel()
        vertical.addWidget(self.graphs)
        vertical.setSizes([585, 300])
        bottom = QHBoxLayout()
        outer.addLayout(bottom)
        self.performance = QLabel("FPS —  |  LATENCY —  |  CPU —  |  MEMORY —")
        self.performance.setObjectName("sub")
        bottom.addWidget(self.performance)
        bottom.addStretch()
        credit = QLabel("Developed for Shivam Singh · MathTech  /  LOCAL PROCESSING")
        credit.setObjectName("sub")
        bottom.addWidget(credit)
        self.statusBar().showMessage("Ready. Camera and recording are off.")
        for shortcut, callback in [
            ("L", self.lock_selected),
            ("U", lambda: self.command("unlock")),
            ("Space", lambda: self.command("pause")),
        ]:
            action = QAction(self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            self.addAction(action)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(33)
        if source is not None:
            QTimer.singleShot(150, lambda: self.start_source(source))

    @staticmethod
    def scroll(widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def button(self, layout, title, callback, primary=False):
        button = QPushButton(title)
        button.clicked.connect(callback)
        if primary:
            button.setObjectName("primary")
        layout.addWidget(button)
        return button

    def command(self, action, value=None):
        if self.worker and self.worker.isRunning():
            self.worker.submit(action, value)
        else:
            self.statusBar().showMessage("Open a source first.")

    def set_option(self, key, value):
        setattr(self.s, key, value)
        self.command("setting", (key, value)) if self.worker else None

    def start_source(self, source):
        if self.worker and self.worker.isRunning():
            self.pending_source = source
            self.worker.stop()
            self.statusBar().showMessage("Closing current source…")
            return
        self.last = None
        self.faces.clear()
        self.graphs.update_rows([], self.s.graph_window)
        self.worker = Worker(source, self.s, self)
        self.worker.notice.connect(self.statusBar().showMessage)
        self.worker.error.connect(self.report_error)
        self.worker.finished.connect(self.worker_finished)
        self.badge.setText("●  OPENING SOURCE")
        self.worker.start()

    def worker_finished(self):
        self.badge.setText("●  SOURCE OFFLINE")
        self.statusBar().showMessage("Source stopped. Recordings finalized.")
        if self.pending_source is not None:
            source = self.pending_source
            self.pending_source = None
            QTimer.singleShot(0, lambda: self.start_source(source))

    def stop_source(self):
        self.pending_source = None
        if self.worker:
            self.worker.stop()

    def report_error(self, message):
        self.statusBar().showMessage("ERROR: " + message)
        self.badge.setText("●  CHECK SOURCE / SETTINGS")

    def open_file(self, image):
        kind = (
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.webp)"
            if image
            else "Videos (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*)"
        )
        path, _ = QFileDialog.getOpenFileName(self, "Open image" if image else "Open video", "", kind)
        if path:
            self.start_source(path)

    def open_stream(self):
        url, ok = QInputDialog.getText(
            self, "Network camera", "RTSP / HTTP stream URL (used locally; not saved):"
        )
        if ok and url.strip():
            self.start_source(url.strip())

    def settings_dialog(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Stop source first",
                "Stop the source before changing camera, detector or tracking settings. Display controls remain available while running.",
            )
            return
        dialog = SettingsDialog(self.s, self)
        if dialog.exec():
            self.s = dialog.result_settings

    def select_at(self, x, y):
        if self.last:
            candidates = [t for t in self.last["tracks"] if t.observed and t.bbox.contains(x, y)]
            if candidates:
                chosen = min(candidates, key=lambda t: t.bbox.area)
                for i in range(self.faces.count()):
                    if self.faces.item(i).data(Qt.ItemDataRole.UserRole) == chosen.face_id:
                        self.faces.setCurrentRow(i)

    def lock_selected(self):
        item = self.faces.currentItem()
        if item:
            self.command("lock", item.data(Qt.ItemDataRole.UserRole))
        else:
            self.statusBar().showMessage("Select a visible face in the video or list.")

    def save_frame(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save displayed frame", "face-motion-frame.png", "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            self.command("save_frame", path)

    def export(self):
        path = QFileDialog.getExistingDirectory(self, "Choose export parent folder")
        if path:
            self.command("export", str(Path(path) / datetime.now().strftime("analysis_%Y%m%d_%H%M%S")))

    def record(self):
        if self.worker:
            self.command("record", str(self.session_root))

    def sessions(self):
        if self.last and self.last.get("recording"):
            QMessageBox.information(self, "Recording active", "End recording before managing sessions.")
            return
        SessionsDialog(self.session_root, self).exec()

    def poll(self):
        if self.worker is None:
            return
        result = self.worker.take()
        if result is None:
            return
        self.last = result
        self.camera.set_frame(result["image"])
        self.badge.setText(
            "●  RECORDING"
            if result.get("recording")
            else ("●  SOURCE PAUSED / COMPLETE" if result.get("paused") else "●  SOURCE ONLINE")
        )
        ident = result["locked_id"]
        status = result["status"]
        self.lock_status.setText(
            f"{status}  ·  TARGET {ident if ident is not None else '—'}"
            + ("\nTarget lost. Select a visible face to reacquire." if status == "REACQUIRING" else "")
        )
        selected = (
            self.faces.currentItem().data(Qt.ItemDataRole.UserRole) if self.faces.currentItem() else None
        )
        visible = [t for t in result["tracks"] if t.observed]
        ids = [t.face_id for t in visible]
        current = [self.faces.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.faces.count())]
        if ids != current:
            self.faces.clear()
            for t in visible:
                item = QListWidgetItem(
                    f"Face #{t.face_id:02d}" + ("  · LOCKED" if t.face_id == ident else "")
                )
                item.setData(Qt.ItemDataRole.UserRole, t.face_id)
                self.faces.addItem(item)
                if selected == t.face_id:
                    self.faces.setCurrentItem(item)
        else:
            for i, t in enumerate(visible):
                self.faces.item(i).setText(
                    f"Face #{t.face_id:02d}" + ("  · LOCKED" if t.face_id == ident else "")
                )

        def fmt(v, unit="", precision=1):
            return "—" if v is None else f"{v:.{precision}f} {unit}"

        m = result["metrics"] or {}
        c = result["coordinates"] or {}
        blur = result["blur"] or {}
        pose = result["pose"] or {}
        track = next((t for t in visible if t.face_id == ident), None)
        rows = [
            ("Pixel center", f"{fmt(c.get('x'))}, {fmt(c.get('y'))}"),
            ("Normalized", f"{fmt(c.get('x_norm'), precision=3)}, {fmt(c.get('y_norm'), precision=3)}"),
            ("Velocity X / Y", f"{fmt(m.get('vx'))} / {fmt(m.get('vy'))} px/s"),
            ("Speed", fmt(m.get("speed"), "px/s")),
            ("Acceleration", fmt(m.get("acceleration"), "px/s²")),
            ("Direction", m.get("direction", "—")),
            ("Relative size", m.get("relative_distance", "—")),
            ("Distance travelled", fmt(m.get("total_distance"), "px")),
            ("Sharpness", fmt(blur.get("sharpness"))),
            ("Blur level", blur.get("blur_level", "—")),
            (
                "Yaw / Pitch / Roll",
                f"{fmt(pose.get('yaw'))} / {fmt(pose.get('pitch'))} / {fmt(pose.get('roll'))}",
            ),
            ("Match score", fmt(track.association_score if track else None, precision=2)),
        ]
        text = (
            "<table cellspacing='7'>"
            + "".join(f"<tr><td style='color:#8ca5bd'>{k}</td><td>{v}</td></tr>" for k, v in rows)
            + "</table>"
        )
        text += f"<p style='color:#8ca5bd'>{result['mask_label']}<br>{result['landmark_engine']}</p>"
        self.metrics.setText(text)
        stats = result["summary"]
        self.statistics.setText(
            "SESSION ANALYTICS\n\n"
            + "\n".join(
                [
                    f"Duration: {stats['duration_seconds']:.1f} s",
                    f"Processed frames: {stats['frames_processed']}",
                    f"Tracked / lost: {stats['frames_tracked']} / {stats['frames_lost']}",
                    f"Tracking continuity: {fmt(stats['continuity_percent'], '%')}",
                    f"Reacquisitions: {stats['reacquisitions']}",
                    f"Total distance: {stats['total_distance_px']:.1f} px",
                    f"Average speed: {fmt(stats.get('average_speed'), 'px/s')}",
                    f"Maximum speed: {fmt(stats.get('maximum_speed'), 'px/s')}",
                    f"Average acceleration: {fmt(stats.get('average_acceleration'), 'px/s²')}",
                    f"Area min / max: {fmt(stats.get('minimum_area'))} / {fmt(stats.get('maximum_area'))}",
                    f"Mean X / Y: {fmt(stats.get('average_cx'))} / {fmt(stats.get('average_cy'))}",
                    "\nContinuity is observation availability, not identity accuracy.",
                    "\nGraph history is bounded. Recording writes every processed row.",
                ]
            )
        )
        if time.monotonic() - self.last_graph > 0.15:
            self.graphs.update_rows(result["graph_rows"], self.s.graph_window)
            self.last_graph = time.monotonic()
        self.performance.setText(
            f"PROCESSING {result.get('processing_fps', stats['processing_fps']):.1f} FPS  |  LATENCY {result.get('latency_ms', 0):.1f} ms  |  CPU {result.get('cpu_percent', 0):.1f}%  |  MEMORY {result.get('memory_mb', 0):.0f} MB  |  {result['detector']}"
        )

    def closeEvent(self, event):
        self.pending_source = None
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(200):
                event.ignore()
                QTimer.singleShot(250, self.close)
                return
        try:
            self.s.save()
        except OSError as exc:
            self.statusBar().showMessage(str(exc))
        event.accept()
