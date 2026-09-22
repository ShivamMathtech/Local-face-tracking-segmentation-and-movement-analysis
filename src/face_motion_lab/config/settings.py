"""Validated, version-independent JSON/YAML settings."""

from dataclasses import dataclass, asdict, fields
from pathlib import Path
import json
import math
import yaml

SETTINGS_PATH = Path.home() / ".face-motion-lab" / "settings.json"


@dataclass
class Settings:
    detector: str = "haar"
    detector_model: str = ""
    landmark_model: str = ""
    calibration_file: str = ""
    camera_index: int = 0
    width: int = 960
    height: int = 540
    fps: float = 30.0
    processing_width: int = 960
    detection_confidence: float = 0.6
    tracking_confidence: float = 0.5
    max_lost_frames: int = 18
    smoothing_factor: float = 0.55
    direction_threshold: float = 8.0
    trajectory_length: int = 100
    graph_window: float = 30.0
    heatmap_window: float = 30.0
    mask_opacity: float = 0.35
    landmarks: bool = True
    landmark_style: str = "points"
    mask: bool = False
    mask_method: str = "refined"
    mask_fill: bool = True
    mask_boundary: bool = True
    segmentation_mode: str = "Original"
    trajectory: bool = True
    coordinates: bool = True
    grid: bool = False
    boxes: bool = True
    movement_vector: bool = True
    head_pose: bool = False
    optical_flow: bool = True
    kalman: bool = True
    coordinate_origin: str = "top-left"
    view_mode: str = "RGB"
    edge_scope: str = "Full frame"
    brightness: float = 0.0
    contrast: float = 1.2
    gain: float = 1.2
    noise: float = 0.0
    intensity: float = 1.0
    thermal_colormap: str = "INFERNO"
    history_limit: int = 20000

    def validate(self) -> "Settings":
        defaults = Settings()
        for field in fields(self):
            value, sample = getattr(self, field.name), getattr(defaults, field.name)
            if isinstance(sample, bool) and type(value) is not bool:
                raise ValueError(f"{field.name} must be boolean")
            if type(sample) is int and (type(value) is not int):
                raise ValueError(f"{field.name} must be an integer")
            if isinstance(sample, float) and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f"{field.name} must be a finite number")
            if isinstance(sample, str) and not isinstance(value, str):
                raise ValueError(f"{field.name} must be text")
        for name in ("detection_confidence", "tracking_confidence", "mask_opacity"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if not 0 < self.smoothing_factor <= 1:
            raise ValueError("Smoothing must be in (0,1]")
        if not 64 <= self.width <= 7680 or not 64 <= self.height <= 4320:
            raise ValueError("Invalid resolution")
        if not 64 <= self.processing_width <= 3840:
            raise ValueError("Invalid processing width")
        if not 1 <= self.fps <= 240:
            raise ValueError("FPS must be between 1 and 240")
        if not 0 <= self.max_lost_frames <= 1000:
            raise ValueError("Invalid lost-frame limit")
        if self.trajectory_length < 0 or not 100 <= self.history_limit <= 1000000:
            raise ValueError("Invalid history length")
        if self.graph_window <= 0 or self.heatmap_window <= 0:
            raise ValueError("Time windows must be positive")
        if self.direction_threshold < 0:
            raise ValueError("Direction threshold must be nonnegative")
        if not 0 <= self.noise <= 50 or not 0 <= self.intensity <= 3:
            raise ValueError("Invalid image effect")
        if not 0 <= self.gain <= 5 or not 0 <= self.contrast <= 5 or not -255 <= self.brightness <= 255:
            raise ValueError("Invalid image adjustment")
        choices = {
            "detector": ("haar", "mediapipe", "yunet"),
            "coordinate_origin": ("top-left", "center", "cartesian"),
            "landmark_style": ("points", "lines", "mesh", "contour"),
            "mask_method": ("ellipse", "polygon", "hull", "refined", "skin"),
            "segmentation_mode": (
                "Original",
                "Face Mask",
                "Masked Face",
                "Background Removed",
                "Face Highlighted",
                "Face Silhouette",
            ),
            "edge_scope": ("Full frame", "Face ROI", "Segmented face"),
            "thermal_colormap": ("INFERNO", "JET", "HOT", "TURBO"),
            "view_mode": (
                "RGB",
                "Grayscale",
                "Canny",
                "Sobel",
                "Laplacian",
                "Gaussian Blur",
                "Median Blur",
                "Motion Blur",
                "Binary Mask",
                "Heatmap",
                "Density Map",
                "Movement Intensity",
                "Night Vision",
                "High Contrast",
                "Thermal Style",
            ),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"Invalid {name}")
        return self

    def save(self, path: Path = SETTINGS_PATH) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temp.replace(path)

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "Settings":
        path = Path(path)
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Settings must be a mapping")
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        return cls(**raw).validate()
