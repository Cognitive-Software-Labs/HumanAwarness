"""Minimal constant-velocity Kalman filter for a single bounding box.

Vendored instead of depending on a library: the model is small (8-dim state,
~50 lines) and is the same standard formulation used by SORT/DeepSORT/
ByteTrack-family trackers, so there is little benefit to an external
dependency for it.

State: [cx, cy, w, h, vcx, vcy, vw, vh] (box center, size, and their
per-frame velocities). Measurement: [cx, cy, w, h].
"""

import numpy as np

_STATE_DIM = 8
_MEAS_DIM = 4


def _bbox_to_measurement(bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return np.array([cx, cy, w, h], dtype=np.float64)


def _state_to_bbox(state: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, w, h = state[0], state[1], state[2], state[3]
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class KalmanBoxTracker:
    """Tracks one box's position/size over time with a constant-velocity model."""

    def __init__(self, bbox_xyxy: tuple[float, float, float, float]):
        self.F = np.eye(_STATE_DIM)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # position/size += velocity * dt (dt=1 frame)

        self.H = np.zeros((_MEAS_DIM, _STATE_DIM))
        for i in range(4):
            self.H[i, i] = 1.0

        # Moderate process noise on position/size, higher on velocity (we don't
        # know how fast a person accelerates); moderate measurement noise.
        self.Q = np.eye(_STATE_DIM) * 1.0
        self.Q[4:, 4:] *= 2.0
        self.R = np.eye(_MEAS_DIM) * 10.0
        self.P = np.eye(_STATE_DIM) * 100.0  # high initial uncertainty, esp. velocity

        self.x = np.zeros(_STATE_DIM)
        self.x[:4] = _bbox_to_measurement(bbox_xyxy)

    def predict(self) -> tuple[float, float, float, float]:
        """Advance the model by one frame and return the predicted bbox."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return _state_to_bbox(self.x)

    def update(self, bbox_xyxy: tuple[float, float, float, float]) -> None:
        """Correct the model with an observed bbox."""
        z = _bbox_to_measurement(bbox_xyxy)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(_STATE_DIM) - K @ self.H) @ self.P

    def reset(self, bbox_xyxy: tuple[float, float, float, float]) -> None:
        """Snap the filter to a new position (e.g. after a ReID re-acquisition,
        where the stale motion model shouldn't be trusted)."""
        self.x = np.zeros(_STATE_DIM)
        self.x[:4] = _bbox_to_measurement(bbox_xyxy)
        self.P = np.eye(_STATE_DIM) * 100.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return _state_to_bbox(self.x)
