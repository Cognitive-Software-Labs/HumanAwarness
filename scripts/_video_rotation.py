"""Shared helper so every video test script sees footage upright, regardless
of container rotation metadata.

Phone-recorded video (WhatsApp in particular) is commonly stored as
landscape pixels plus a 90/-90/180 degree "rotate on playback" tag in the
container, rather than actually rotating the stored pixels. Neither
OpenCV's VideoCapture/VideoWriter nor Ultralytics' own video loader - both
used throughout this repo's test scripts - read that tag, so a portrait
phone video gets silently processed sideways: confirmed to make detection,
tracking, and head pose output meaningless (e.g. head yaw read as ~0 degrees
on a person facing fully away, because the crop fed to the model was
actually rotated ~90 degrees from what it was trained on). This was
previously worked around by hand for the whatsapp2/whatsapp3 test clips
(see the `*_rotated` files in test_output/) but never fixed in code, so it
silently resurfaced on the next new clip - see docs/context/human_tracking.md.
"""

import hashlib
import subprocess
from pathlib import Path

_CACHE_DIR = Path.home() / ".cache" / "human_awareness_scripts" / "rotated_videos"


def _container_rotation_degrees(video_path: Path) -> int:
    """Reads the container's display-matrix rotation tag via ffprobe, 0 if
    absent/unreadable (e.g. ffprobe not installed - fails open, not closed,
    since this is a best-effort correction, not a correctness guarantee)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream_side_data=rotation",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                return int(float(line))
            except ValueError:
                continue
    return 0


def ensure_upright(video_path: Path) -> Path:
    """Returns a path to an upright version of `video_path`: the original
    path unchanged if it has no rotation tag, otherwise a cached copy with
    the rotation baked into the pixel data (ffmpeg's filtergraph path
    applies the display-matrix rotation automatically once any -vf is
    given) and the tag cleared, so every downstream cv2/Ultralytics reader
    sees the already-correct frames.

    Cached under ~/.cache (keyed by resolved path + mtime) so repeated runs
    on the same clip don't re-encode every time.
    """
    video_path = Path(video_path)
    rotation = _container_rotation_degrees(video_path)
    if rotation % 360 == 0:
        return video_path

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{video_path.resolve()}:{video_path.stat().st_mtime_ns}".encode()).hexdigest()[:16]
    cached = _CACHE_DIR / f"{video_path.stem}_{key}_upright.mp4"
    if cached.exists():
        return cached

    print(
        f"[_video_rotation] '{video_path.name}' carries a {rotation}-degree rotation tag that "
        f"OpenCV/Ultralytics ignore - baking it into the pixels once (cached at {cached})..."
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", "format=yuv420p",
            "-c:v", "libx264", "-crf", "18",
            "-metadata:s:v:0", "rotate=0",
            str(cached),
        ],
        check=True, capture_output=True,
    )
    return cached
