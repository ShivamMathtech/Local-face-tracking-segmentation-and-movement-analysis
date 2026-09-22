import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QGridLayout


class GraphPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        pg.setConfigOptions(antialias=False, background="#101b2a", foreground="#91a6bd")
        self.curves = {}
        self.plots = {}
        fields = [
            ("cx", "X position", "px"),
            ("cy", "Y position", "px"),
            ("vx", "Velocity X", "px/s"),
            ("vy", "Velocity Y", "px/s"),
            ("speed", "Speed", "px/s"),
            ("acceleration", "Acceleration", "px/s²"),
            ("area", "Face area", "px²"),
            ("width", "Width / height", "px"),
        ]
        for i, (key, title, unit) in enumerate(fields):
            plot = pg.PlotWidget()
            plot.setMinimumHeight(115)
            plot.setTitle(title, color="#c6d9e9", size="10pt")
            plot.showGrid(x=True, y=True, alpha=0.12)
            plot.setLabel("left", unit)
            plot.setLabel("bottom", "s")
            plot.setMenuEnabled(False)
            self.curves[key] = plot.plot(pen=pg.mkPen("#58ddb8", width=1.6), connect="finite")
            self.plots[key] = plot
            layout.addWidget(plot, i // 4, i % 4)
            if key == "width":
                self.height_curve = plot.plot(pen=pg.mkPen("#83aaff", width=1.5), connect="finite")

    def update_rows(self, rows, window):
        if not rows:
            for curve in self.curves.values():
                curve.setData([], [])
            self.height_curve.setData([], [])
            return
        now = rows[-1]["timestamp"]
        rows = [r for r in rows if r["timestamp"] >= now - window]
        expanded = []
        prev = None
        for row in rows:
            if prev is not None and row["segment"] != prev["segment"]:
                expanded.append(None)
            expanded.append(row)
            prev = row
        t = [np.nan if r is None else r["timestamp"] for r in expanded]
        for key, curve in self.curves.items():
            curve.setData(t, [np.nan if r is None else r[key] for r in expanded])
            self.plots[key].setXRange(max(0, now - window), max(window, now), padding=0)
        self.height_curve.setData(t, [np.nan if r is None else r["height"] for r in expanded])
