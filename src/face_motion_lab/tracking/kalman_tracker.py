"""Constant-velocity Kalman filter: [cx, cy, vx, vy, width, height]."""

import numpy as np
from face_motion_lab.types import Box


class KalmanBox:
    def __init__(self, box: Box):
        self.x = np.array([*box.center, 0.0, 0.0, box.width, box.height])
        self.P = np.diag([20.0, 20.0, 300.0, 300.0, 20.0, 20.0])
        self.H = np.zeros((4, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 4] = self.H[3, 5] = 1
        self.R = np.diag([9.0, 9.0, 16.0, 16.0])

    @property
    def box(self):
        cx, cy, _, _, w, h = self.x
        return Box(float(cx - w / 2), float(cy - h / 2), float(max(8, w)), float(max(8, h)))

    def predict(self, dt: float):
        dt = max(1e-4, min(dt, 1.0))
        a = np.eye(6)
        a[0, 2] = a[1, 3] = dt
        self.x = a @ self.x
        self.P = a @ self.P @ a.T + np.diag([2 * dt, 2 * dt, 30 * dt, 30 * dt, 2 * dt, 2 * dt])
        return self.box

    def correct(self, box: Box):
        z = np.array([*box.center, box.width, box.height])
        innovation = z - self.H @ self.x
        s = self.H @ self.P @ self.H.T + self.R
        k = np.linalg.solve(s, (self.P @ self.H.T).T).T
        self.x += k @ innovation
        # Joseph form preserves numerical symmetry and positive semidefiniteness.
        ikh = np.eye(6) - k @ self.H
        self.P = ikh @ self.P @ ikh.T + k @ self.R @ k.T
        return self.box
