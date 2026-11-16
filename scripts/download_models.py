#!/usr/bin/env python3
"""Fetch every model the pipeline loads into ~/.cache, so it can then run
with no network.

Run once on a machine that is online (or copy ~/.cache/human_* over from one
that already ran it). Each stage's own loader does the fetching - the same
code path, URL and checksum a node would use on first launch - so this just
triggers them all up front. Weights already in the cache are not downloaded
again; YOLO weights found in the repo root are copied rather than downloaded.

Usage:
    source .venv/bin/activate
    python3 scripts/download_models.py            # what the default config uses
    python3 scripts/download_models.py --extras   # + yolo26s-pose, OSNet-AIN
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "human_tracking"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_head_pose"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_gaze"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_perception"))

from human_gaze.gaze_estimator import GazeEstimator
from human_gaze.mediapipe_gate import MediaPipeGate
from human_head_pose.head_pose_estimator import HeadPoseEstimator
from human_perception.model_path import model_cache_dir
from human_tracking.reid_embedder import ReIDEmbedder


def fetch_yolo(name: str) -> None:
    from ultralytics.utils.downloads import attempt_download_asset

    target = model_cache_dir() / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    local = REPO_ROOT / name
    if local.exists():
        shutil.copy2(local, target)
    else:
        attempt_download_asset(str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extras", action="store_true", help="also fetch the optional yolo26s-pose and OSNet-AIN weights")
    args = parser.parse_args()

    yolo_names = ["yolo26n-pose.pt"] + (["yolo26s-pose.pt"] if args.extras else [])
    reid_models = ["osnet_x1_0"] + (["osnet_ain_x1_0"] if args.extras else [])

    for name in yolo_names:
        print(f"[detection] {name}")
        fetch_yolo(name)
    for model in reid_models:
        print(f"[tracking]  {model}")
        ReIDEmbedder(model=model)
    print("[head pose] 6DRepNet360")
    HeadPoseEstimator()
    print("[gaze]      MobileGaze")
    GazeEstimator()
    print("[gaze]      MediaPipe Face Landmarker")
    MediaPipeGate()

    cache_root = model_cache_dir().parent
    print("\nCached weights:")
    total = 0
    for d in ("human_perception", "human_tracking", "human_head_pose", "human_gaze"):
        for f in sorted((cache_root / d).glob("*")):
            if f.is_file():
                total += f.stat().st_size
                print(f"  {f.stat().st_size / 1e6:7.1f} MB  {f}")
    print(f"  {total / 1e6:7.1f} MB  total")


if __name__ == "__main__":
    main()
