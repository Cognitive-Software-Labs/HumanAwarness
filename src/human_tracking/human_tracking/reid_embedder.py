"""Extracts OSNet appearance embeddings from person crops, for ReID matching.

Two architectures are supported (`model`), both trained on MSMT17 - chosen
for generalization to an unseen camera/environment, see
docs/context/human_tracking.md:
- `osnet_x1_0` (default): plain OSNet.
- `osnet_ain_x1_0`: OSNet with adaptive instance normalisation, the
  upstream model zoo's domain-generalisation variant.

Weights are cached locally and downloaded via gdown on first use only, from
the official deep-person-reid model zoo; gdown itself is only imported at
that point, so a machine with the weights already cached never needs it
installed. A fresh download is checked against a pinned sha256.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from human_tracking.osnet import osnet_x1_0
from human_tracking.osnet_ain import osnet_ain_x1_0


@dataclass(frozen=True)
class _ReIDModelSpec:
    constructor: Callable[..., torch.nn.Module]
    gdrive_id: str
    cache_filename: str
    sha256: str


# Google Drive file ids from the deep-person-reid MODEL_ZOO.md "MSMT17
# (combineall=True) -> Market1501 & DukeMTMC-reID" cross-domain table - the
# checkpoints the upstream repo positions for deployment/generalization use.
_MODELS = {
    "osnet_x1_0": _ReIDModelSpec(
        constructor=osnet_x1_0,
        gdrive_id="1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x",
        cache_filename="osnet_x1_0_msmt17.pth",
        sha256="48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727",
    ),
    "osnet_ain_x1_0": _ReIDModelSpec(
        constructor=osnet_ain_x1_0,
        gdrive_id="1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal",
        cache_filename="osnet_ain_x1_0_msmt17.pth",
        sha256="8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f",
    ),
}
REID_MODELS = tuple(_MODELS)

_INPUT_HW = (256, 128)  # OSNet's expected (height, width)

# torchreid's training-time input normalization (ImageNet statistics).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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


def _default_cache_path(filename: str) -> Path:
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "human_tracking"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified(spec: _ReIDModelSpec, path: Path) -> None:
    import gdown

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".download")
    gdown.download(id=spec.gdrive_id, output=str(partial), quiet=False)
    actual = _sha256(partial)
    if actual != spec.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ReID checkpoint sha256 mismatch: expected {spec.sha256}, got {actual}")
    partial.rename(path)


def load_reid_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    """Load a torchreid checkpoint into `model`, failing loudly on mismatch.

    Strips the `module.` prefix that DataParallel-trained checkpoints carry
    (the official OSNet-AIN MSMT17 file has it on every key) and drops the
    training-set classifier head, which is never used for embeddings. Every
    other model parameter must be present with the right shape - a partial
    load would silently leave random weights in the backbone.
    """
    state_dict = state_dict.get("state_dict", state_dict)
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model_dict = model.state_dict()
    backbone_keys = [k for k in model_dict if not k.startswith("classifier.")]
    bad = [k for k in backbone_keys if k not in state_dict or state_dict[k].shape != model_dict[k].shape]
    if bad:
        raise ValueError(f"ReID checkpoint does not match the architecture ({len(bad)} keys, e.g. {bad[:3]})")
    model_dict.update({k: state_dict[k] for k in backbone_keys})
    model.load_state_dict(model_dict)


class ReIDEmbedder:
    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cpu",
        model: str = "osnet_x1_0",
        normalize: bool = True,
    ):
        if model not in _MODELS:
            raise ValueError(f"Unknown ReID model '{model}', expected one of {REID_MODELS}")
        spec = _MODELS[model]
        self.model_name = model
        self.normalize = normalize
        self.device = torch.device(device)
        path = Path(weights_path) if weights_path else _default_cache_path(spec.cache_filename)

        if not path.exists():
            _download_verified(spec, path)

        self.model = spec.constructor(num_classes=1000, pretrained=False, loss="softmax")
        load_reid_state_dict(self.model, torch.load(str(path), map_location="cpu", weights_only=True))
        self.model.eval()
        self.model.to(self.device)
        self._mean = _IMAGENET_MEAN.to(self.device)
        self._std = _IMAGENET_STD.to(self.device)

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
        if self.normalize:
            batch = (batch - self._mean) / self._std
        embeddings = F.normalize(self.model(batch), dim=1).cpu().numpy()

        result: list[np.ndarray | None] = [None] * len(bgr_crops)
        for batch_i, crop_i in enumerate(valid_idx):
            result[crop_i] = embeddings[batch_i]
        return result
