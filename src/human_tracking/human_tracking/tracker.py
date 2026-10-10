"""Per-frame association and lifecycle advancement for all tracks.

Two matching strategies, applied in order:
1. Motion (IoU between each track's Kalman-predicted bbox and each
   detection) - cheap, used for NEW/ACTIVE/LOST_KALMAN tracks.
2. Appearance (ReID embedding similarity against each track's gallery) -
   for detections the motion step left unmatched, against LOST_GALLERY
   tracks (motion prediction is no longer trusted by then) *and* against
   LOST_KALMAN tracks whose motion match just failed this frame.

   The LOST_KALMAN fallback matters in practice (docs/context/human_tracking.md):
   real pre-occlusion detections are often noisy (a shrinking/jittering box
   right as someone is about to be hidden), and a constant-velocity Kalman
   filter can extrapolate that noise into a wildly wrong predicted position
   over even a couple of seconds - IoU matching then fails even though the
   real detection reappeared somewhere reasonable. Appearance doesn't care
   where the motion model thinks the person should be.

Stage 2 has two extra guards beyond a plain similarity threshold, added
after real testing showed that a bare threshold produces false-positive
merges once several lost tracks are candidates at once (e.g. a crowded
scene):
- a spatial gate: a track can only match a detection within a plausible
  walking distance of where it was last seen, scaled by both the track's
  own apparent size and how long it's been lost.
- a similarity margin: a candidate match is only accepted if it clearly
  beats whatever else that track or detection could instead match to -
  an ambiguous near-tie is left unmatched rather than guessed at.
"""

from dataclasses import dataclass

import numpy as np

from human_tracking.track import ACTIVE, LOST_GALLERY, LOST_KALMAN, NEW, Track, crop_quality, is_plausible_person_crop


@dataclass
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: list[tuple[float, float, float]]  # per-keypoint (x, y, confidence)
    embedding: np.ndarray | None


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_height(bbox: tuple[float, float, float, float]) -> float:
    return max(1.0, bbox[3] - bbox[1])


class Tracker:
    def __init__(
        self,
        gallery_size: int = 8,
        iou_threshold: float = 0.3,
        reid_threshold: float = 0.65,
        reid_margin: float = 0.08,
        max_speed_bbox_heights_per_sec: float = 3.0,
        confirm_frames: int = 3,
        kalman_timeout_s: float = 2.5,
        gallery_timeout_s: float = 45.0,
    ):
        self.tracks: list[Track] = []
        self.gallery_size = gallery_size
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold
        self.reid_margin = reid_margin
        self.max_speed_bbox_heights_per_sec = max_speed_bbox_heights_per_sec
        self.confirm_frames = confirm_frames
        self.kalman_timeout_s = kalman_timeout_s
        self.gallery_timeout_s = gallery_timeout_s

    def update(self, detections: list[Detection], frame_area: float, dt_seconds: float) -> list[Track]:
        motion_tracks = [t for t in self.tracks if t.state in (NEW, ACTIVE, LOST_KALMAN)]
        gallery_tracks = [t for t in self.tracks if t.state == LOST_GALLERY]

        # Gate embeddings once up front: a crop without enough confident,
        # spread-out keypoints (see is_plausible_person_crop) never
        # contributes to a gallery and never acts as a ReID match
        # candidate, whether incoming or stored.
        gated_embeddings: list[np.ndarray | None] = [
            d.embedding if (d.embedding is not None and is_plausible_person_crop(d.keypoints, d.bbox_xyxy)) else None
            for d in detections
        ]

        for t in motion_tracks:
            t.kalman.predict()

        # --- stage 1: motion (IoU) matching ---
        pairs = []
        for ti, t in enumerate(motion_tracks):
            for di, d in enumerate(detections):
                iou = compute_iou(t.kalman.bbox, d.bbox_xyxy)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)

        matched_track_idx: set[int] = set()
        matched_det_idx: set[int] = set()
        for _iou, ti, di in pairs:
            if ti in matched_track_idx or di in matched_det_idx:
                continue
            matched_track_idx.add(ti)
            matched_det_idx.add(di)
            d = detections[di]
            quality = crop_quality(d.confidence, d.bbox_xyxy, frame_area)
            motion_tracks[ti].mark_matched(
                d.bbox_xyxy, d.confidence, d.keypoints, gated_embeddings[di], quality, self.confirm_frames
            )

        unmatched_detections = [(di, d) for di, d in enumerate(detections) if di not in matched_det_idx]

        # --- stage 2: ReID matching ---
        # Pool = every LOST_GALLERY track, plus any LOST_KALMAN track that
        # just failed its motion match this frame (see module docstring).
        failed_lost_kalman = [
            t for ti, t in enumerate(motion_tracks) if ti not in matched_track_idx and t.state == LOST_KALMAN
        ]
        reid_pool = gallery_tracks + failed_lost_kalman

        reid_pairs = []
        for ti, t in enumerate(reid_pool):
            track_center = _bbox_center(t.last_bbox)
            track_scale = _bbox_height(t.last_bbox)
            max_dist = self.max_speed_bbox_heights_per_sec * track_scale * max(t.seconds_since_last_seen, dt_seconds)
            for ui, (di, d) in enumerate(unmatched_detections):
                emb = gated_embeddings[di]
                if emb is None:
                    continue
                det_center = _bbox_center(d.bbox_xyxy)
                dist = ((track_center[0] - det_center[0]) ** 2 + (track_center[1] - det_center[1]) ** 2) ** 0.5
                if dist > max_dist:
                    continue  # spatial gate: too far to plausibly be the same person
                sim = t.best_similarity(emb)
                if sim >= self.reid_threshold:
                    reid_pairs.append((sim, ti, ui))
        reid_pairs.sort(key=lambda p: p[0], reverse=True)

        matched_pool_idx: set[int] = set()
        matched_udet_idx: set[int] = set()
        # A track/detection is "resolved" once touched by its best remaining
        # candidate, whether that candidate is accepted or rejected for being
        # ambiguous - either way it must not fall through and match a worse
        # candidate afterward.
        resolved_tracks: set[int] = set()
        resolved_dets: set[int] = set()
        for pi, (sim, ti, ui) in enumerate(reid_pairs):
            if ti in resolved_tracks or ui in resolved_dets:
                continue
            # similarity margin: only accept if this candidate clearly beats whatever
            # else remains available for the same track or the same detection - an
            # ambiguous near-tie is left unmatched rather than guessed at this frame.
            runner_up = max(
                (
                    s2
                    for s2, t2, u2 in reid_pairs[pi + 1 :]
                    if (t2 == ti or u2 == ui) and t2 not in resolved_tracks and u2 not in resolved_dets
                ),
                default=None,
            )
            resolved_tracks.add(ti)
            resolved_dets.add(ui)
            if runner_up is not None and (sim - runner_up) < self.reid_margin:
                continue
            matched_pool_idx.add(ti)
            matched_udet_idx.add(ui)
            di, d = unmatched_detections[ui]
            quality = crop_quality(d.confidence, d.bbox_xyxy, frame_area)
            reid_pool[ti].mark_matched(
                d.bbox_xyxy, d.confidence, d.keypoints, gated_embeddings[di], quality, self.confirm_frames, snap_kalman=True
            )

        # --- advance whatever still wasn't matched by either stage ---
        matched_pool_tracks = {reid_pool[ti] for ti in matched_pool_idx}
        for ti, t in enumerate(motion_tracks):
            if ti not in matched_track_idx and t not in matched_pool_tracks:
                t.mark_missed(dt_seconds, self.kalman_timeout_s, self.gallery_timeout_s)
        for t in gallery_tracks:
            if t not in matched_pool_tracks:
                t.mark_missed(dt_seconds, self.kalman_timeout_s, self.gallery_timeout_s)

        # --- spawn new tentative tracks for whatever's left unmatched ---
        for ui, (di, d) in enumerate(unmatched_detections):
            if ui in matched_udet_idx:
                continue
            new_track = Track(d.bbox_xyxy, gallery_size=self.gallery_size)
            quality = crop_quality(d.confidence, d.bbox_xyxy, frame_area)
            new_track.mark_matched(
                d.bbox_xyxy, d.confidence, d.keypoints, gated_embeddings[di], quality, self.confirm_frames
            )
            self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if t.is_alive]
        return [t for t in self.tracks if t.is_published]
