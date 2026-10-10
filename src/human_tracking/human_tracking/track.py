"""A single tracked person and its lifecycle state.

Lifecycle (see docs/context/human_tracking.md):

    NEW --(3 consecutive confirmed frames)--> ACTIVE
    ACTIVE --(missed detection)--> LOST_KALMAN
    LOST_KALMAN --(re-matched by motion/IoU)--> ACTIVE
    LOST_KALMAN --(timeout, e.g. 2-3s)--> LOST_GALLERY
    LOST_GALLERY --(re-matched by ReID)--> ACTIVE
    LOST_GALLERY --(timeout, e.g. 30-60s)--> FORGOTTEN (track is dropped)

NEW and FORGOTTEN are never published - a track_id only "exists" externally
once it's a confirmed person (ACTIVE or still-recoverable LOST_*), matching
the idea that awareness state is tracked per confirmed identity, not per
raw/unconfirmed detection blip.
"""

from dataclasses import dataclass

import numpy as np

from human_tracking.kalman_box_tracker import KalmanBoxTracker

NEW = "NEW"
ACTIVE = "ACTIVE"
LOST_KALMAN = "LOST_KALMAN"
LOST_GALLERY = "LOST_GALLERY"
FORGOTTEN = "FORGOTTEN"


@dataclass
class GalleryEntry:
    embedding: np.ndarray
    quality: float


def is_plausible_person_crop(
    keypoints: list[tuple[float, float, float]],
    bbox_xyxy: tuple[float, float, float, float],
    min_keypoints: int = 4,
    min_keypoint_conf: float = 0.3,
    min_spread_frac: float = 0.3,
) -> bool:
    """Reject crops that don't actually show enough of a person's body to
    trust a ReID embedding from them.

    Two earlier versions of this check used bbox *shape* (aspect ratio,
    then also height-as-fraction-of-frame) as a proxy for "is this a real,
    mostly-complete person." Both got replaced after real testing
    (docs/context/human_tracking.md) found cases a shape-only check can't
    tell apart: a box clipped by the frame edge during a legitimate exit
    or re-entry looks geometrically identical - narrow, tall, touching the
    boundary - whether it's a real partial view of a person or a genuinely
    degenerate detection (e.g. a stray arm/bag fragment). One specific
    width/height ratio threshold was simultaneously too strict for a real
    close-up framing shot and too lenient for a real degenerate crop; no
    single shape rule resolves both.

    This version uses the actual pose estimate instead of inferring body
    shape from the box: require a minimum number of keypoints the detector
    is actually confident about, spread across a reasonable fraction of
    the box's height (not all clustered in one small region, e.g. just a
    face). A real person clipped at the frame edge (losing width, not
    height) still has keypoints spanning most of their visible height, so
    this correctly allows that case through while still rejecting crops
    with too few confidently-located keypoints to mean anything.

    Args:
        keypoints: per-keypoint (x, y, confidence) tuples, in the order
            produced by the pose model (any length/order - only confidence
            and y are used here).
    """
    x1, y1, x2, y2 = bbox_xyxy
    bbox_height = y2 - y1
    if bbox_height <= 0:
        return False

    visible_ys = [kp_y for _kp_x, kp_y, conf in keypoints if conf >= min_keypoint_conf]
    if len(visible_ys) < min_keypoints:
        return False

    spread = max(visible_ys) - min(visible_ys)
    return spread >= min_spread_frac * bbox_height


def crop_quality(confidence: float, bbox_xyxy: tuple[float, float, float, float], frame_area: float) -> float:
    """Heuristic quality score for a detection crop: higher confidence and a
    larger (closer/clearer) crop both make for a more reliable gallery sample."""
    x1, y1, x2, y2 = bbox_xyxy
    area_frac = max(0.0, (x2 - x1) * (y2 - y1)) / max(frame_area, 1.0)
    return float(confidence) * min(area_frac * 4.0, 1.0)  # saturate once reasonably large


class Track:
    _next_id = 1

    def __init__(self, bbox_xyxy: tuple[float, float, float, float], gallery_size: int = 8):
        self.state = NEW
        self.track_id: int | None = None  # assigned only on NEW -> ACTIVE promotion
        self.kalman = KalmanBoxTracker(bbox_xyxy)
        self.gallery_size = gallery_size
        self.gallery: list[GalleryEntry] = []

        self.consecutive_hits = 0
        self.seconds_since_last_seen = 0.0

        self.last_bbox = bbox_xyxy
        self.last_confidence = 0.0
        self.last_keypoints = []

    def _assign_id(self) -> int:
        new_id = Track._next_id
        Track._next_id += 1
        return new_id

    def update_gallery(self, embedding: np.ndarray | None, quality: float) -> None:
        if embedding is None:
            return
        if len(self.gallery) < self.gallery_size:
            self.gallery.append(GalleryEntry(embedding, quality))
            return
        worst_idx = min(range(len(self.gallery)), key=lambda i: self.gallery[i].quality)
        if quality > self.gallery[worst_idx].quality:
            self.gallery[worst_idx] = GalleryEntry(embedding, quality)

    def best_similarity(self, embedding: np.ndarray) -> float:
        """Max cosine similarity between `embedding` and anything in this track's gallery."""
        if not self.gallery:
            return -1.0
        sims = [float(np.dot(embedding, entry.embedding)) for entry in self.gallery]
        return max(sims)

    def mark_matched(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        confidence: float,
        keypoints,
        embedding: np.ndarray | None,
        quality: float,
        confirm_frames: int,
        snap_kalman: bool = False,
    ) -> None:
        """Record a successful match (direct motion/IoU match or ReID re-acquisition)."""
        if snap_kalman:
            self.kalman.reset(bbox_xyxy)
        else:
            self.kalman.update(bbox_xyxy)

        self.last_bbox = bbox_xyxy
        self.last_confidence = confidence
        self.last_keypoints = keypoints
        self.seconds_since_last_seen = 0.0
        self.update_gallery(embedding, quality)

        if self.state == NEW:
            self.consecutive_hits += 1
            if self.consecutive_hits >= confirm_frames:
                self.state = ACTIVE
                self.track_id = self._assign_id()
        else:
            self.state = ACTIVE
            self.consecutive_hits += 1

    def mark_missed(self, dt_seconds: float, kalman_timeout_s: float, gallery_timeout_s: float) -> None:
        """Advance this track's state given it was NOT matched to any detection this frame."""
        if self.state == NEW:
            self.state = FORGOTTEN  # tentative tracks don't get a grace period
            return

        self.seconds_since_last_seen += dt_seconds

        if self.state == ACTIVE:
            self.state = LOST_KALMAN
        elif self.state == LOST_KALMAN:
            if self.seconds_since_last_seen >= kalman_timeout_s:
                self.state = LOST_GALLERY
        elif self.state == LOST_GALLERY:
            if self.seconds_since_last_seen >= gallery_timeout_s:
                self.state = FORGOTTEN

    @property
    def is_alive(self) -> bool:
        return self.state != FORGOTTEN

    @property
    def is_published(self) -> bool:
        """Only confirmed identities (never NEW/FORGOTTEN) are reported externally."""
        return self.state in (ACTIVE, LOST_KALMAN, LOST_GALLERY) and self.track_id is not None
