"""Where the YOLO-pose weights live, so the node finds them without a network.

A bare name like `yolo26n-pose.pt` is otherwise resolved by Ultralytics
relative to the current working directory, and a miss there triggers a
download from GitHub - so the same launch command works or fails offline
depending on which folder it was started from. Resolving bare names against
`~/.cache/human_perception/` (the same per-package cache the head pose, gaze
and ReID weights use) makes the lookup independent of the cwd.
"""

import os
from pathlib import Path


def model_cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "human_perception"


def resolve_model_path(model_path: str) -> str:
    """Return `model_path` if it points at an existing file or carries a
    directory; for a bare file name not found in the cwd, return the cached
    copy if there is one, else the name unchanged (Ultralytics then downloads
    it, which needs a network)."""
    path = Path(model_path).expanduser()
    if path.exists() or path.parent != Path("."):
        return str(path)
    cached = model_cache_dir() / path.name
    return str(cached) if cached.exists() else model_path
