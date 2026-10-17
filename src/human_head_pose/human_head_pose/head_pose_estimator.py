"""Extracts head pose (pitch/yaw/roll, degrees) from head crops via 6DRepNet360.

Weights (the "Full-Rotation" checkpoint, fine-tuned on 300W-LP + Panoptic -
the widest-range one the upstream repo publishes, matching why 360 was picked
over the original 6DRepNet in the first place: see docs/context/head_pose.md)
are a plain HTTPS download, cached locally on first use - no gdown/Google
Drive dance needed here, unlike OSNet's weights in human_tracking.
"""

import os
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch

from human_head_pose.sixdrepnet360 import SixDRepNet360, compute_euler_angles_from_rotation_matrices

# The model's own forward-direction vector in its own axes, used to recover
# yaw/pitch straight from the rotation matrix - see _face_direction_pitch_yaw.
_FORWARD_AXIS = torch.tensor([0.0, 0.0, 1.0])

_WEIGHTS_URL = (
    "https://cloud.ovgu.de/s/TewGC9TDLGgKkmS/download/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
)

_INPUT_SIZE = 224  # after the same resize-256/center-crop-224 the upstream repo trains and tests with.
_RESIZE_SHORT_SIDE = 256
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _default_cache_path() -> Path:
    cache_dir = Path.home() / ".cache" / "human_head_pose"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "6drepnet360_full_rotation.pth"


def _face_direction_pitch_yaw(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Yaw and pitch of the face-normal direction (`R @ [0,0,1]`), in
    radians - NOT `sixdrepnet360`'s own 3-angle (pitch, yaw, roll) Euler
    decomposition of the same matrix.

    A full Euler decomposition can represent the same physical "facing
    directly away" orientation more than one way - confirmed on real
    back-of-head crops (see docs/context/head_pose.md): decomposed yaw read
    near 0 (implying "facing the camera") on three different people whose
    backs were unambiguously turned to it, while pitch and roll had instead
    swung to near +-180 to compensate - the rotation didn't go away, it
    just aliased onto the other two angles. That's a known degeneracy of
    recovering 3 sequential Euler angles from a rotation matrix once one of
    them is cos()=0 (i.e. here, whenever the person doesn't happen to
    confirm the model's implicit assumption that yaw carries the primary
    rotation): several different (pitch, yaw, roll) triples describe the
    identical rotation matrix, and which one the decomposition formula
    picks isn't the one a reader expects.

    Converting the face-normal *vector* straight to azimuth/elevation
    sidesteps that: it only recovers the 2 degrees of freedom that actually
    determine which way the face points (azimuth = yaw-like, elevation =
    pitch-like) and deliberately drops the 3rd (roll/twist around the face's
    own view axis, which doesn't change where it's pointing) - with only 2
    angles to recover from a single 3D direction, there's no third angle
    left for the rotation to alias onto.
    """
    forward = _FORWARD_AXIS.to(rotation_matrices.device)
    face_dir = torch.matmul(rotation_matrices, forward)
    yaw = torch.atan2(face_dir[:, 0], face_dir[:, 2])
    pitch = torch.asin(torch.clamp(-face_dir[:, 1], -1.0, 1.0))
    return torch.stack((pitch, yaw), dim=1)


def _resize_center_crop(rgb: np.ndarray) -> np.ndarray:
    """Resize the shorter side to 256, then center-crop 224x224 - exactly the
    upstream repo's `transforms.Resize(256), transforms.CenterCrop(224)`."""
    h, w = rgb.shape[:2]
    scale = _RESIZE_SHORT_SIDE / min(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(rgb, (new_w, new_h))

    y_off = max(0, (new_h - _INPUT_SIZE) // 2)
    x_off = max(0, (new_w - _INPUT_SIZE) // 2)
    cropped = resized[y_off : y_off + _INPUT_SIZE, x_off : x_off + _INPUT_SIZE]
    if cropped.shape[0] != _INPUT_SIZE or cropped.shape[1] != _INPUT_SIZE:
        # The crop came from near a resized edge too small to fill 224 (a very
        # thin/extreme-aspect head box) - pad with the edge instead of failing.
        cropped = cv2.copyMakeBorder(
            cropped,
            0,
            _INPUT_SIZE - cropped.shape[0],
            0,
            _INPUT_SIZE - cropped.shape[1],
            cv2.BORDER_REPLICATE,
        )
    return cropped


class HeadPoseEstimator:
    def __init__(self, weights_path: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        path = Path(weights_path) if weights_path else _default_cache_path()

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            # Download to a temp file and only rename it into place on
            # success - otherwise an interruption partway through (network
            # drop, Ctrl-C) leaves a truncated file at `path`, which the
            # `not path.exists()` check above would then treat as already
            # cached forever, failing opaquely on every future load.
            tmp_fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
            os.close(tmp_fd)
            tmp_path = Path(tmp_name)
            try:
                urllib.request.urlretrieve(_WEIGHTS_URL, str(tmp_path))
                tmp_path.replace(path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise

        self.model = SixDRepNet360()
        state_dict = torch.load(str(path), map_location="cpu")
        self.model.load_state_dict(state_dict.get("model_state_dict", state_dict))
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def estimate(self, bgr_crops: list[np.ndarray]) -> list[tuple[float, float, float] | None]:
        """Return one (pitch_deg, yaw_deg, roll_deg) per crop (None for empty crops)."""
        valid_idx = [i for i, c in enumerate(bgr_crops) if c is not None and c.size > 0]
        if not valid_idx:
            return [None] * len(bgr_crops)

        batch = torch.empty(len(valid_idx), 3, _INPUT_SIZE, _INPUT_SIZE, dtype=torch.float32)
        for batch_i, crop_i in enumerate(valid_idx):
            rgb = cv2.cvtColor(bgr_crops[crop_i], cv2.COLOR_BGR2RGB)
            cropped = _resize_center_crop(rgb).astype(np.float32) / 255.0
            normalized = (cropped - _IMAGENET_MEAN) / _IMAGENET_STD
            batch[batch_i] = torch.from_numpy(normalized).permute(2, 0, 1)

        batch = batch.to(self.device)
        rotation_matrices = self.model(batch)

        # pitch/yaw: from the face-normal vector directly (see
        # _face_direction_pitch_yaw) - robust at any orientation, including
        # facing directly away. roll: still from sixdrepnet360's own Euler
        # decomposition - unaffected by the ambiguity above (it's the yaw/
        # pitch pair that aliases into one another and into roll, not roll
        # in isolation) and not used by gaze_robot_deg, only for display.
        pitch_yaw_deg = (_face_direction_pitch_yaw(rotation_matrices) * 180.0 / np.pi).cpu().numpy()
        euler_deg = compute_euler_angles_from_rotation_matrices(rotation_matrices) * 180.0 / np.pi
        roll_deg = euler_deg[:, 2].cpu().numpy()

        result: list[tuple[float, float, float] | None] = [None] * len(bgr_crops)
        for batch_i, crop_i in enumerate(valid_idx):
            pitch, yaw = pitch_yaw_deg[batch_i]
            roll = roll_deg[batch_i]
            result[crop_i] = (float(pitch), float(yaw), float(roll))
        return result
