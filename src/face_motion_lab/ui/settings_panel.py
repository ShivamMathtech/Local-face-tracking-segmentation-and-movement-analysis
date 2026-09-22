from dataclasses import asdict
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QDialogButtonBox,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QLineEdit,
    QScrollArea,
    QWidget,
    QCheckBox,
    QMessageBox,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
)
from face_motion_lab.config.settings import Settings


class SettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Device, models and tracking settings")
        self.resize(630, 720)
        self.original = settings
        self.result_settings = None
        self.widgets = {}
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        specs = [
            ("detector", "Detector", ["haar", "mediapipe", "yunet"]),
            ("detector_model", "Detector model file", "file"),
            ("landmark_model", "Dense landmark .task model", "file"),
            ("calibration_file", "Camera calibration JSON", "file"),
            ("camera_index", "Camera index", (0, 20)),
            ("width", "Camera width", (64, 7680)),
            ("height", "Camera height", (64, 4320)),
            ("fps", "Requested capture FPS", (1.0, 240.0)),
            ("processing_width", "Processing width", (64, 3840)),
            ("detection_confidence", "Detector confidence (not Haar)", (0.0, 1.0)),
            ("tracking_confidence", "Association score threshold", (0.0, 1.0)),
            ("max_lost_frames", "Maximum lost frames", (0, 1000)),
            ("smoothing_factor", "Smoothing: 1 = no smoothing", (0.01, 1.0)),
            ("direction_threshold", "Stationary threshold px/s", (0.0, 200.0)),
            ("graph_window", "Graph window seconds", (1.0, 3600.0)),
            ("heatmap_window", "Heatmap window seconds", (1.0, 3600.0)),
            ("history_limit", "Retained analysis rows", (100, 1000000)),
            ("kalman", "Kalman prediction", True),
            ("optical_flow", "Optical flow assist", True),
            ("head_pose", "Head pose (requires dense model)", True),
        ]
        for name, label, spec in specs:
            value = getattr(settings, name)
            if isinstance(spec, list):
                widget = QComboBox()
                widget.addItems(spec)
                widget.setCurrentText(value)
            elif spec == "file":
                widget = QLineEdit(value)
                row = QWidget()
                layout = QHBoxLayout(row)
                layout.setContentsMargins(0, 0, 0, 0)
                button = QPushButton("Browse")
                button.clicked.connect(lambda checked=False, w=widget: self.browse(w))
                layout.addWidget(widget)
                layout.addWidget(button)
                form.addRow(label, row)
                self.widgets[name] = widget
                continue
            elif spec is True:
                widget = QCheckBox()
                widget.setChecked(value)
            else:
                widget = QSpinBox() if type(spec[0]) is int else QDoubleSpinBox()
                widget.setRange(*spec)
                widget.setValue(value)
                if isinstance(widget, QDoubleSpinBox):
                    widget.setSingleStep(0.05)
            self.widgets[name] = widget
            form.addRow(label, widget)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def browse(self, widget):
        path, _ = QFileDialog.getOpenFileName(self, "Select model or calibration file")
        if path:
            widget.setText(path)

    def save(self):
        values = asdict(self.original)
        for key, w in self.widgets.items():
            if isinstance(w, QComboBox):
                values[key] = w.currentText()
            elif isinstance(w, QLineEdit):
                values[key] = w.text()
            elif isinstance(w, QCheckBox):
                values[key] = w.isChecked()
            else:
                values[key] = w.value()
        try:
            self.result_settings = Settings(**values).validate()
            self.result_settings.save()
            self.accept()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
