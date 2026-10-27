"""Extracts gaze yaw/pitch (degrees, camera-frame) from a face crop via
MobileGaze (mobileone_s0, Gaze360-trained) - see human_gaze/mobileone_gaze.py
for the vendored architecture.

Weights: downloaded from the upstream repo's GitHub release
(github.com/yakhyo/gaze-estimation), cached locally on first use, same
pattern as 6DRepNet360's weights in human_head_pose. Code license is MIT,
but the weights are trained exclusively on Gaze360, whose own license
restricts "models trained on the dataset" to research/non-commercial use
(confirmed by reading gaze360's LICENSE.md directly, not assumed) - the
same tradeoff the design doc already accepts for this model choice
(docs/context/gaze.md), not something decided here.
"""

import os
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from human_gaze.mobileone_gaze import mobileone_s0_gaze

_WEIGHTS_URL = "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0.pt"

_NUM_BINS = 90
_BIN_WIDTH_DEG = 4.0
_ANGLE_OFFSET_DEG = 180.0  # bins span -180..+176 deg (90 * 4 = 360, centered by this offset)
_RESIZE_SHORT_SIDE = 448  # matches the upstream repo's `transforms.Resize(448)` - see docs/context/gaze.md
# for why this aspect-preserving resize is used instead of the ONNX export
# path's distorting square resize.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _default_cache_path() -> Path:
    cache_dir = Path.home() / ".cache" / "human_gaze"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "mobileone_s0_gaze360.pt"


def _resize_short_side(rgb: np.ndarray, target: int) -> np.ndarray:
    """Matches torchvision's `transforms.Resize(target)` (a single int):
    scales the shorter side to `target`, preserving aspect ratio, no crop -
    deliberately NOT a square resize (see module docstring)."""
    h, w = rgb.shape[:2]
    scale = target / min(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    return cv2.resize(rgb, (new_w, new_h))


def gaze_yaw_pitch_to_vector(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Yaw/pitch (degrees) -> unit 3D gaze vector, camera optical frame
    (+x right, +y down, +z forward) - same sign convention as the upstream
    repo's own `gaze_to_3d` (utils/helpers.py), reproduced here rather than
    re-derived to avoid silently picking a different (wrong) convention.

    Unused by the current flat-angle pipeline - kept for the phase-2
    geometric decision rule (docs/context/gaze.md), which needs an actual
    3D gaze vector rather than a yaw angle to test against a distance-based
    visual cone."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    return np.array(
        [-np.cos(pitch) * np.sin(yaw), -np.sin(pitch), -np.cos(pitch) * np.cos(yaw)],
        dtype=np.float64,
    )


class GazeEstimator:
    def __init__(self, weights_path: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        path = Path(weights_path) if weights_path else _default_cache_path()

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Same download-to-temp-then-rename pattern as 6DRepNet360's
            # weights - an interrupted download must not leave a truncated
            # file that `not path.exists()` would treat as already cached.
            tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
            os.close(tmp_fd)
            tmp_path = Path(tmp_name)
            try:
                urllib.request.urlretrieve(_WEIGHTS_URL, str(tmp_path))
                tmp_path.replace(path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise

        self.model = mobileone_s0_gaze(num_bins=_NUM_BINS)
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        self._bin_idx = torch.arange(_NUM_BINS, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def estimate(self, bgr_face_crops: list[np.ndarray]) -> list[tuple[float, float] | None]:
        """Return one (yaw_deg, pitch_deg) per face crop (None for empty
        crops). Crops may differ in aspect ratio (no center-crop is
        applied, see `_resize_short_side`), so each is run through the
        model individually rather than batched into one fixed-size tensor."""
        results: list[tuple[float, float] | None] = [None] * len(bgr_face_crops)
        for i, crop in enumerate(bgr_face_crops):
            if crop is None or crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = _resize_short_side(rgb, _RESIZE_SHORT_SIDE).astype(np.float32) / 255.0
            normalized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

            yaw_logits, pitch_logits = self.model(tensor)
            yaw_deg = self._decode(yaw_logits)
            pitch_deg = self._decode(pitch_logits)
            results[i] = (yaw_deg, pitch_deg)
        return results

    def _decode(self, logits: torch.Tensor) -> float:
        """Expected value over the softmax bin distribution -> degrees.
        Exact formula from the upstream repo (inference.py): `sum(softmax_i
        * i) * binwidth - angle_offset`."""
        probs = F.softmax(logits, dim=1)
        value = (probs * self._bin_idx).sum(dim=1) * _BIN_WIDTH_DEG - _ANGLE_OFFSET_DEG
        return float(value.item())
