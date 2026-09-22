from pathlib import Path
import shutil
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QFileDialog,
    QMessageBox,
    QHBoxLayout,
    QLabel,
)
from face_motion_lab.recording.exporter import zip_session


class SessionsDialog(QDialog):
    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = Path(root)
        self.setWindowTitle("Recorded sessions")
        self.resize(650, 400)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(str(self.root)))
        self.items = QListWidget()
        layout.addWidget(self.items)
        bar = QHBoxLayout()
        layout.addLayout(bar)
        for title, action in [
            ("Choose folder", self.choose),
            ("Export selected ZIP", self.export),
            ("Delete selected session", self.delete),
        ]:
            b = QPushButton(title)
            b.clicked.connect(action)
            bar.addWidget(b)
        self.refresh()

    def refresh(self):
        self.items.clear()
        self.paths = sorted(
            [p for p in self.root.glob("session_*") if p.is_dir() and (p / "session.json").is_file()],
            reverse=True,
        )
        self.items.addItems([p.name for p in self.paths])

    def selected(self):
        i = self.items.currentRow()
        return self.paths[i] if 0 <= i < len(self.paths) else None

    def choose(self):
        folder = QFileDialog.getExistingDirectory(self, "Session root", str(self.root))
        if folder:
            self.root = Path(folder)
            self.refresh()

    def export(self):
        p = self.selected()
        if p:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export session", str(p.parent / (p.name + ".zip")), "ZIP (*.zip)"
            )
            if path:
                try:
                    zip_session(p, path)
                    QMessageBox.information(self, "Exported", path)
                except Exception as exc:
                    QMessageBox.warning(self, "Export failed", str(exc))

    def delete(self):
        p = self.selected()
        if (
            p
            and QMessageBox.question(self, "Delete session?", f"Permanently delete {p.name}?")
            == QMessageBox.StandardButton.Yes
        ):
            try:
                if p.is_symlink() or p.parent.resolve() != self.root.resolve():
                    raise ValueError("Invalid session path")
                shutil.rmtree(p)
                self.refresh()
            except Exception as exc:
                QMessageBox.warning(self, "Delete failed", str(exc))
