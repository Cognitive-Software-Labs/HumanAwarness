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

A third stage, reconciliation, runs right after stage 1: a just-created
(or just-reconfirmed) track is still allowed to lose its detection to an
older lost identity via ReID, for a short grace window after its own
creation. This exists because a person's first reappearance frame(s) can
have too little information for ReID to recognize them immediately (e.g.
too few confident keypoints on a small/edge-clipped crop) even though the
very same person becomes clearly recognizable a few frames later - by
which point, without this stage, a brand-new track would already have
claimed every subsequent detection via motion/IoU, and the original lost
identity would never get another chance (docs/context/human_tracking.md
has the real case this fixes). When reconciliation accepts a match, the
young track's identity is discarded and the older, reclaimed identity
takes over its position - so a track_id a caller already saw published
can be superseded by an older one within the grace window.
"""

from dataclasses import dataclass

import numpy as np

from human_tracking.track import ACTIVE, LOST_GALLERY, LOST_KALMAN, NEW, Track, crop_quality, is_plausible_person_crop


def _find_reid_matches(
    lost_pool: list[Track],
    candidates: list[tuple[int, tuple[float, float], np.ndarray | None, np.ndarray | None]],
    max_speed_bbox_heights_per_sec: float,
    reid_threshold: float,
    reid_margin: float,
    dt_seconds: float,
    excluded_pool_idx: frozenset[int] = frozenset(),
    density_threshold_step: float = 0.0,
    density_margin_step: float = 0.0,
) -> dict[int, int]:
    """Greedily match `lost_pool` tracks against `candidates` by appearance
    similarity, gated by a spatial plausibility check (see module docstring)
    and resolved with a similarity margin against the runner-up on either
    side. Each candidate is `(key, center_xy, embedding, torso_embedding)`.
    Returns `{lost_pool_index: candidate_key}` for accepted matches.

    Density gating: a track/candidate pair's required threshold and margin
    both scale up with how many *other* spatially-plausible pairings exist
    for either side (`density_threshold_step`/`density_margin_step` per
    extra contender) - with exactly one plausible pairing on both sides
    (the common single/few-person case), this has no effect at all.
    Real-footage testing found this does NOT fully address false merges in
    a dense crowd (docs/context/human_tracking.md): the false merges found
    there mostly had only one real above-threshold contender at the moment
    of the match, not several competing - the failure mode was a single
    confident-looking but wrong match, which no amount of competition-aware
    gating catches. Kept as a real, if partial, mitigation regardless.
    """
    # Pass 1: every spatially-plausible pairing, regardless of similarity -
    # this defines "how many bodies could plausibly be this person" (local
    # density) independent of what the appearance model happens to say.
    spatial_pairs = []  # (ti, key, sim)
    for ti, t in enumerate(lost_pool):
        if ti in excluded_pool_idx:
            continue
        track_center = _bbox_center(t.last_bbox)
        track_scale = _bbox_height(t.last_bbox)
        max_dist = max_speed_bbox_heights_per_sec * track_scale * max(t.seconds_since_last_seen, dt_seconds)
        for key, center, emb, temb in candidates:
            if emb is None:
                continue
            dist = ((track_center[0] - center[0]) ** 2 + (track_center[1] - center[1]) ** 2) ** 0.5
            if dist > max_dist:
                continue  # spatial gate: too far to plausibly be the same person
            spatial_pairs.append((ti, key, t.best_similarity(emb, temb)))

    contenders_per_track: dict[int, int] = {}
    contenders_per_key: dict[int, int] = {}
    for ti, key, _sim in spatial_pairs:
        contenders_per_track[ti] = contenders_per_track.get(ti, 0) + 1
        contenders_per_key[key] = contenders_per_key.get(key, 0) + 1

    # Pass 2: apply the (possibly density-raised) threshold.
    pairs = []  # (sim, ti, key, margin_for_this_pair)
    for ti, key, sim in spatial_pairs:
        extra_contenders = max(contenders_per_track[ti], contenders_per_key[key]) - 1
        eff_threshold = reid_threshold + density_threshold_step * extra_contenders
        eff_margin = reid_margin + density_margin_step * extra_contenders
        if sim >= eff_threshold:
            pairs.append((sim, ti, key, eff_margin))
    pairs.sort(key=lambda p: p[0], reverse=True)

    matches: dict[int, int] = {}
    resolved_tracks: set[int] = set()
    resolved_keys: set[int] = set()
    for pi, (sim, ti, key, eff_margin) in enumerate(pairs):
        if ti in resolved_tracks or key in resolved_keys:
            continue
        # similarity margin: only accept if this candidate clearly beats whatever
        # else remains available for the same track or the same candidate - an
        # ambiguous near-tie is left unmatched rather than guessed at this frame.
        runner_up = max(
            (
                s2
                for s2, t2, k2, _m2 in pairs[pi + 1 :]
                if (t2 == ti or k2 == key) and t2 not in resolved_tracks and k2 not in resolved_keys
            ),
            default=None,
        )
        resolved_tracks.add(ti)
        resolved_keys.add(key)
        if runner_up is not None and (sim - runner_up) < eff_margin:
            continue
        matches[ti] = key
    return matches


@dataclass
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints: list[tuple[float, float, float]]  # per-keypoint (x, y, confidence)
    embedding: np.ndarray | None
    torso_embedding: np.ndarray | None = None  # see track.torso_crop_bbox


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
        reconcile_window_s: float = 1.5,
        density_threshold_step: float = 0.05,
        density_margin_step: float = 0.05,
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
        self.reconcile_window_s = reconcile_window_s
        self.density_threshold_step = density_threshold_step
        self.density_margin_step = density_margin_step
        # {abandoned_track_id: surviving_track_id} for every reconciliation
        # merge so far - a caller that already published `abandoned_track_id`
        # knows it was retroactively superseded, not actually a separate person.
        self.merged_ids: dict[int, int] = {}

    def update(self, detections: list[Detection], frame_area: float, dt_seconds: float) -> list[Track]:
        for t in self.tracks:
            t.age_seconds += dt_seconds

        motion_tracks = [t for t in self.tracks if t.state in (NEW, ACTIVE, LOST_KALMAN)]
        gallery_tracks = [t for t in self.tracks if t.state == LOST_GALLERY]

        # Gate embeddings once up front: a crop without enough confident,
        # spread-out keypoints (see is_plausible_person_crop) never
        # contributes to a gallery and never acts as a ReID match
        # candidate, whether incoming or stored. Both the full-crop and the
        # torso-only embedding are gated by the same plausibility check - if
        # the crop as a whole isn't trustworthy, neither reading from it is.
        plausible = [is_plausible_person_crop(d.keypoints, d.bbox_xyxy) for d in detections]
        gated_embeddings: list[np.ndarray | None] = [
            d.embedding if (d.embedding is not None and p) else None for d, p in zip(detections, plausible)
        ]
        gated_torso_embeddings: list[np.ndarray | None] = [
            d.torso_embedding if (d.torso_embedding is not None and p) else None
            for d, p in zip(detections, plausible)
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
        track_to_det: dict[int, int] = {}
        for _iou, ti, di in pairs:
            if ti in matched_track_idx or di in matched_det_idx:
                continue
            matched_track_idx.add(ti)
            matched_det_idx.add(di)
            track_to_det[ti] = di
            d = detections[di]
            quality = crop_quality(d.confidence, d.bbox_xyxy, frame_area)
            motion_tracks[ti].mark_matched(
                d.bbox_xyxy,
                d.confidence,
                d.keypoints,
                gated_embeddings[di],
                quality,
                self.confirm_frames,
                torso_embedding=gated_torso_embeddings[di],
            )

        unmatched_detections = [(di, d) for di, d in enumerate(detections) if di not in matched_det_idx]

        # Pool shared by both remaining stages: every LOST_GALLERY track,
        # plus any LOST_KALMAN track that just failed its motion match this
        # frame (see module docstring).
        failed_lost_kalman = [
            t for ti, t in enumerate(motion_tracks) if ti not in matched_track_idx and t.state == LOST_KALMAN
        ]
        reid_pool = gallery_tracks + failed_lost_kalman

        # --- stage 1.5: reconciliation ---
        # Give every just-(re)matched track that's still within its grace
        # window a chance to lose its detection to an older lost identity
        # instead - see module docstring for why this matters in practice.
        reconcile_candidates = [
            (ti, _bbox_center(motion_tracks[ti].last_bbox), gated_embeddings[di], gated_torso_embeddings[di])
            for ti, di in track_to_det.items()
            if motion_tracks[ti].age_seconds <= self.reconcile_window_s
        ]
        reconcile_matches = _find_reid_matches(
            reid_pool,
            reconcile_candidates,
            self.max_speed_bbox_heights_per_sec,
            self.reid_threshold,
            self.reid_margin,
            dt_seconds,
            density_threshold_step=self.density_threshold_step,
            density_margin_step=self.density_margin_step,
        )
        reconciled_young_tracks: set[Track] = set()
        for lost_ti, young_ti in reconcile_matches.items():
            young_track = motion_tracks[young_ti]
            di = track_to_det[young_ti]
            d = detections[di]
            quality = crop_quality(d.confidence, d.bbox_xyxy, frame_area)
            reclaimed_track = reid_pool[lost_ti]
            reclaimed_track.mark_matched(
                d.bbox_xyxy,
                d.confidence,
                d.keypoints,
                gated_embeddings[di],
                quality,
                self.confirm_frames,
                torso_embedding=gated_torso_embeddings[di],
                snap_kalman=True,
            )
            if young_track.track_id is not None:
                self.merged_ids[young_track.track_id] = reclaimed_track.track_id
            reconciled_young_tracks.add(young_track)

        # --- stage 2: ReID matching for whatever's left truly unmatched ---
        det_candidates = [
            (ui, _bbox_center(d.bbox_xyxy), gated_embeddings[di], gated_torso_embeddings[di])
            for ui, (di, d) in enumerate(unmatched_detections)
        ]
        reid_matches = _find_reid_matches(
            reid_pool,
            det_candidates,
            self.max_speed_bbox_heights_per_sec,
            self.reid_threshold,
            self.reid_margin,
            dt_seconds,
            excluded_pool_idx=frozenset(reconcile_matches.keys()),
            density_threshold_step=self.density_threshold_step,
            density_margin_step=self.density_margin_step,
        )
        for ti, ui in reid_matches.items():
            di, d = unmatched_detections[ui]
            quality = crop_quality(d.confidence, d.bbox_xyxy, frame_area)
            reid_pool[ti].mark_matched(
                d.bbox_xyxy,
                d.confidence,
                d.keypoints,
                gated_embeddings[di],
                quality,
                self.confirm_frames,
                torso_embedding=gated_torso_embeddings[di],
                snap_kalman=True,
            )

        # --- advance whatever still wasn't matched by any stage ---
        matched_pool_tracks = {reid_pool[ti] for ti in reid_matches} | {reid_pool[ti] for ti in reconcile_matches}
        for ti, t in enumerate(motion_tracks):
            if ti not in matched_track_idx and t not in matched_pool_tracks:
                t.mark_missed(dt_seconds, self.kalman_timeout_s, self.gallery_timeout_s)
        for t in gallery_tracks:
            if t not in matched_pool_tracks:
                t.mark_missed(dt_seconds, self.kalman_timeout_s, self.gallery_timeout_s)

        # --- spawn new tentative tracks for whatever's left unmatched ---
        matched_udet_idx = set(reid_matches.values())
        for ui, (di, d) in enumerate(unmatched_detections):
            if ui in matched_udet_idx:
                continue
            new_track = Track(d.bbox_xyxy, gallery_size=self.gallery_size)
            quality = crop_quality(d.confidence, d.bbox_xyxy, frame_area)
            new_track.mark_matched(
                d.bbox_xyxy,
                d.confidence,
                d.keypoints,
                gated_embeddings[di],
                quality,
                self.confirm_frames,
                torso_embedding=gated_torso_embeddings[di],
            )
            self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if t.is_alive and t not in reconciled_young_tracks]
        return [t for t in self.tracks if t.is_published]
