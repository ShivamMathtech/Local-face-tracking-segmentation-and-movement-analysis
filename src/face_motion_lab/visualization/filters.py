"""Display transforms only; never used as detector input."""

import cv2
import numpy as np

MODES = [
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
]


def sharpness(frame, box):
    x, y, w, h = box.clipped(frame.shape[1], frame.shape[0]).ints()
    if w < 3 or h < 3:
        return None
    gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "sharpness": value,
        "blur_score": 1 / (1 + value / 100),
        "blur_level": "HIGH" if value < 50 else ("MEDIUM" if value < 150 else "LOW"),
    }


def transform(frame, settings, mask=None, box=None, heatmap=None):
    out = frame.copy()
    mode = settings.view_mode
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode == "Grayscale":
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == "Gaussian Blur":
        out = cv2.GaussianBlur(frame, (17, 17), 0)
    elif mode == "Median Blur":
        out = cv2.medianBlur(frame, 7)
    elif mode == "Motion Blur":
        kernel = np.zeros((15, 15), np.float32)
        kernel[7, :] = 1 / 15
        out = cv2.filter2D(frame, -1, kernel)
    elif mode in ("Canny", "Sobel", "Laplacian"):
        if mode == "Canny":
            edges = cv2.Canny(gray, 70, 150)
        elif mode == "Sobel":
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            edges = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
        else:
            edges = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_32F))
        out = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        if settings.edge_scope != "Full frame":
            scope = np.zeros(gray.shape, np.uint8)
            if settings.edge_scope == "Segmented face" and mask is not None:
                scope = mask
            elif box is not None:
                x, y, w, h = box.clipped(gray.shape[1], gray.shape[0]).ints()
                scope[y : y + h, x : x + w] = 255
            out = np.where((scope > 127)[:, :, None], out, frame)
    elif mode == "Binary Mask":
        out = cv2.cvtColor(mask if mask is not None else np.zeros_like(gray), cv2.COLOR_GRAY2BGR)
    elif mode == "High Contrast":
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif mode == "Night Vision":
        lum = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        lum = cv2.GaussianBlur(lum, (3, 3), 0).astype(float)
        lum = lum * settings.contrast * settings.gain + settings.brightness
        if settings.noise:
            lum += np.random.default_rng().normal(0, settings.noise, lum.shape)
        lum = np.clip(lum * settings.intensity, 0, 255).astype(np.uint8)
        out = np.zeros_like(frame)
        out[:, :, 1] = lum
        out[:, :, 0] = lum // 8
        out[:, :, 2] = lum // 10
    elif mode == "Thermal Style":
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        out = cv2.applyColorMap(norm, getattr(cv2, "COLORMAP_" + settings.thermal_colormap))
    elif mode in ("Heatmap", "Density Map", "Movement Intensity") and heatmap is not None:
        out = cv2.addWeighted(frame, 0.45, heatmap, 0.7, 0) if mode == "Heatmap" else heatmap.copy()
    return out
