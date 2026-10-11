"""Extracts OSNet appearance embeddings from person crops, for ReID matching.

Weights (osnet_x1_0 trained on MSMT17 - chosen for generalization to an
unseen camera/environment, see docs/context/human_tracking.md) are cached
locally and downloaded via gdown on first use only; gdown itself is only
imported at that point, so a machine with the weights already cached never
needs it installed.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from human_tracking.osnet import osnet_x1_0

# google drive file id for osnet_x1_0 trained on MSMT17 (combineall=True),
# from the deep-person-reid MODEL_ZOO.md "cross-domain" table - chosen over
# the same-domain checkpoint because it's trained on more data and is the
# one the upstream repo positions for deployment/generalization use.
_MSMT17_GDRIVE_ID = "1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x"

_INPUT_HW = (256, 128)  # OSNet's expected (height, width)


def _letterbox_resize(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize into `target_hw` preserving aspect ratio, padding the rest with
    black instead of stretching to fill it.

    A plain cv2.resize to OSNet's fixed 256x128 distorts a partial-body crop's
    proportions (e.g. a half-body view gets squished into a full-body-shaped
    box) in a way OSNet never saw in training - ReID benchmarks are almost
    all full-body pedestrian crops. Letterboxing keeps the person's actual
    proportions intact; the model just sees some black padding instead.
    """
    target_h, target_w = target_hw
    src_h, src_w = rgb.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = cv2.resize(rgb, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=rgb.dtype)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def _default_cache_path() -> Path:
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "human_tracking"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "osnet_x1_0_msmt17.pth"


class ReIDEmbedder:
    def __init__(self, weights_path: str | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        path = Path(weights_path) if weights_path else _default_cache_path()

        if not path.exists():
            import gdown

            path.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(f"https://drive.google.com/uc?id={_MSMT17_GDRIVE_ID}", str(path), quiet=False)

        self.model = osnet_x1_0(num_classes=1000, pretrained=False, loss="softmax")
        state_dict = torch.load(str(path), map_location="cpu")
        model_dict = self.model.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(matched)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def embed(self, bgr_crops: list[np.ndarray]) -> list[np.ndarray | None]:
        """Return one L2-normalized 512-dim embedding per crop (None for empty crops)."""
        valid_idx = [i for i, c in enumerate(bgr_crops) if c is not None and c.size > 0]
        if not valid_idx:
            return [None] * len(bgr_crops)

        batch = torch.empty(len(valid_idx), 3, *_INPUT_HW, dtype=torch.float32)
        for batch_i, crop_i in enumerate(valid_idx):
            rgb = cv2.cvtColor(bgr_crops[crop_i], cv2.COLOR_BGR2RGB)
            resized = _letterbox_resize(rgb, _INPUT_HW)
            batch[batch_i] = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        batch = batch.to(self.device)
        embeddings = F.normalize(self.model(batch), dim=1).cpu().numpy()

        result: list[np.ndarray | None] = [None] * len(bgr_crops)
        for batch_i, crop_i in enumerate(valid_idx):
            result[crop_i] = embeddings[batch_i]
        return result
