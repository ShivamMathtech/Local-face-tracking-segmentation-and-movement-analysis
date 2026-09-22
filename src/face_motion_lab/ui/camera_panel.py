from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QImage, QPainter, QColor
from PySide6.QtWidgets import QWidget


class CameraPanel(QWidget):
    clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.picture = None
        self.target = QRectF()
        self.setMinimumSize(400, 260)
        self.setMouseTracking(True)
        self.setAccessibleName("Video view; click a face to select")

    def set_frame(self, bgr):
        h, w = bgr.shape[:2]
        self.picture = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888).copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#080e16"))
        if self.picture is None:
            painter.setPen(QColor("#8eabc3"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "FACE MOTION LAB\n\nStart a camera, open media, or explore the synthetic demo.\n\nAll camera frames stay on this computer.",
            )
            return
        size = self.picture.size()
        size.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.target = QRectF(
            (self.width() - size.width()) / 2,
            (self.height() - size.height()) / 2,
            size.width(),
            size.height(),
        )
        painter.drawImage(self.target, self.picture)

    def mousePressEvent(self, event):
        if self.picture and self.target.contains(event.position()):
            pos = event.position() - self.target.topLeft()
            self.clicked.emit(
                pos.x() * self.picture.width() / self.target.width(),
                pos.y() * self.picture.height() / self.target.height(),
            )
