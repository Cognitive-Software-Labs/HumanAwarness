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

A directional variant of the spatial gate (reject a candidate behind
where the track's Kalman velocity was heading when lost) was tried as a
hard veto on a single candidate and reverted - see
docs/context/human_tracking.md. It fixed a confirmed real false merge
(two different people, one exiting left, another appearing from the
right) but broke an already-validated single-person clip where the same
person simply turned around and walked back the way they came - a real,
common reappearance pattern that looks directionally identical (cosine
~-0.9 to -1.0 against the old heading either way) to the false-merge case
it was meant to catch; no pure heading-based signal separated the two as
an absolute judgment on one candidate. It came back in a narrower form,
in the reconciliation stage below: heading (and keypoint-derived
orientation) as a *ranking* tie-breaker between two already
appearance-qualified candidates competing for the same reappearance -
never as a veto on a lone candidate - since the real false-merge case
that found it had exactly that shape (two different people's lost tracks
both spatially and appearance plausible for the same detection at once).

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

Both appearance-matching stages (2 and the reconciliation stage above) fuse
OSNet similarity with a keypoint-derived orientation signal (see
human_tracking.orientation): a track/candidate pair with confidently-known,
different orientations (e.g. a gallery sample that was FRONTAL vs. a
reappearance that's BACK) gets a small similarity bonus before the
threshold/margin check, since OSNet's own score is known to understate a
genuine same-person match across a real viewpoint flip. This is additive
and only fires with direct per-pair evidence, not a blanket threshold
change - unlike the earlier bare-threshold-lowering experiment
(docs/context/human_tracking.md), it shouldn't by itself make unrelated
same-orientation candidates (the actual crowd-scene false-merge failure
mode) any easier to match.

Two optional "bystander" constraints (ported from the Ridgeback leader-
follower's non-leader memory, see docs/context/human_tracking.md) use the one
piece of identity evidence that needs no appearance model at all: two tracks
matched to separate detections in the same frame are two different people.
- `cooccurrence_veto`: a young track that was ever seen at the same time as a
  lost identity can never be reconciled into it (hard constraint, stage 1.5).
- `bystander_negative_check`: a lost identity may only be revived on a
  detection (stage 1.5 or 2) if that detection looks more like the lost
  identity than like any *other* track visible elsewhere in the same frame -
  those are certainly not this detection, so resembling one of them at least
  as much means the appearance evidence isn't discriminating.
"""

from dataclasses import dataclass

import numpy as np

from human_tracking.orientation import UNKNOWN as ORIENTATION_UNKNOWN
from human_tracking.orientation import estimate_orientation
from human_tracking.track import ACTIVE, LOST_GALLERY, LOST_KALMAN, NEW, Track, crop_quality, is_plausible_person_crop


_MIN_CONFIDENT_HEADING_SPEED_FRAC = 0.002  # bbox-heights/frame below which a track's last heading is too noisy to trust


def _find_reid_matches(
    lost_pool: list[Track],
    candidates: list[tuple[int, tuple[float, float], np.ndarray | None, np.ndarray | None, str]],
    max_speed_bbox_heights_per_sec: float,
    reid_threshold: float,
    reid_margin: float,
    dt_seconds: float,
    excluded_pool_idx: frozenset[int] = frozenset(),
    density_threshold_step: float = 0.0,
    density_margin_step: float = 0.0,
    orientation_relief: float = 0.0,
    torso_rescue_weight: float = 0.4,
    heading_rank_weight: float = 0.0,
    orientation_rank_weight: float = 0.0,
    blocked_pairs: frozenset[tuple[int, int]] = frozenset(),
    negative_sims: dict[int, float] | None = None,
    negative_margin: float = 0.0,
    veto_log: list[tuple[str, int, int, float, float]] | None = None,
) -> dict[int, int]:
    """Greedily match `lost_pool` tracks against `candidates` by appearance
    similarity, gated by a spatial plausibility check (see module docstring)
    and resolved with a similarity margin against the runner-up on either
    side. Each candidate is
    `(key, center_xy, embedding, torso_embedding, orientation)`.
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

    `heading_rank_weight`/`orientation_rank_weight`: two more signals -
    where a lost track's Kalman velocity says it was headed, and its
    keypoint-derived body orientation - but applied *only* to ranking among
    candidates that already qualify on appearance alone, never to whether a
    candidate qualifies in the first place (`qualify_sim` below is always
    appearance-only). A hard reject on heading was tried first and reverted
    (docs/context/human_tracking.md): a real person simply turning around
    and walking back produces the same "candidate is behind my last
    heading" signature as two different people arriving from opposite
    directions, so it can't be used as an absolute veto on a single
    candidate. It's a real signal, though, for deciding *between* two
    already-qualified candidates competing for the same detection - which
    is the case that was actually found in real footage: two different
    people's lost tracks both spatially and appearance-plausible for the
    same reappearance, correctly resolved by preferring whichever one the
    candidate is actually arriving from the direction of.

    `blocked_pairs`: `(lost_pool_index, candidate_key)` pairs known to be two
    different people (co-visible at some point) - dropped before the density
    count, since a provably-different track isn't a plausible contender.
    `negative_sims`: per candidate key, its best similarity to tracks known
    NOT to be it (visible elsewhere this frame); a pair only qualifies if it
    beats that by `negative_margin`. Both are logged to `veto_log` as
    `(kind, lost_pool_index, candidate_key, qualify_sim, negative_sim)`
    when they reject a pair that would otherwise have passed the threshold.
    """
    # Pass 1: every spatially-plausible pairing, regardless of similarity -
    # this defines "how many bodies could plausibly be this person" (local
    # density) independent of what the appearance model happens to say.
    spatial_pairs = []  # (ti, key, qualify_sim, rank_sim)
    for ti, t in enumerate(lost_pool):
        if ti in excluded_pool_idx:
            continue
        if not t.gallery:
            # Can never produce a match (best_similarity returns -1.0
            # unconditionally for an empty gallery) - excluded before the
            # spatial gate, not just the similarity threshold, so it also
            # doesn't inflate the density/contenders count below. A real
            # case found this mattering: an unrelated empty-gallery track
            # that was merely spatially nearby was being counted as a
            # "contender," raising the effective threshold enough that a
            # genuine reappearance's own original track never requalified
            # for its own person - see docs/context/human_tracking.md.
            continue
        track_center = _bbox_center(t.last_bbox)
        track_scale = _bbox_height(t.last_bbox)
        max_dist = max_speed_bbox_heights_per_sec * track_scale * max(t.seconds_since_last_seen, dt_seconds)
        # Kalman state is [cx, cy, w, h, vcx, vcy, vw, vh] - velocity stays
        # frozen at whatever it was at the last real detection through pure
        # predict() calls (no measurement to update it with), so this is the
        # track's heading at the moment it was lost, not a stale/drifted value.
        vcx, vcy = t.kalman.x[4], t.kalman.x[5]
        heading_speed = (vcx**2 + vcy**2) ** 0.5
        has_confident_heading = heading_speed >= _MIN_CONFIDENT_HEADING_SPEED_FRAC * track_scale
        for key, center, emb, temb, orientation in candidates:
            if emb is None:
                continue
            dx, dy = center[0] - track_center[0], center[1] - track_center[1]
            dist = (dx**2 + dy**2) ** 0.5
            if dist > max_dist:
                continue  # spatial gate: too far to plausibly be the same person
            qualify_sim = t.best_similarity(
                emb,
                temb,
                query_orientation=orientation,
                orientation_relief=orientation_relief,
                torso_rescue_weight=torso_rescue_weight,
            )
            if (ti, key) in blocked_pairs:
                if veto_log is not None and qualify_sim >= reid_threshold:
                    veto_log.append(("cooccurrence", ti, key, qualify_sim, float("nan")))
                continue
            rank_sim = qualify_sim
            if orientation_rank_weight:
                # Reuses best_similarity's own orientation-aware entry
                # selection rather than re-deriving which gallery entry
                # "wins" - the delta this produces vs. the relief-free
                # qualify_sim is exactly the orientation contribution.
                oriented_sim = t.best_similarity(
                    emb,
                    temb,
                    query_orientation=orientation,
                    orientation_relief=orientation_rank_weight,
                    torso_rescue_weight=torso_rescue_weight,
                )
                rank_sim += oriented_sim - qualify_sim
            if heading_rank_weight and has_confident_heading and dist > 0:
                cos_heading = (dx * vcx + dy * vcy) / (dist * heading_speed)
                rank_sim += heading_rank_weight * cos_heading
            spatial_pairs.append((ti, key, qualify_sim, rank_sim))

    contenders_per_track: dict[int, int] = {}
    contenders_per_key: dict[int, int] = {}
    for ti, key, _qualify_sim, _rank_sim in spatial_pairs:
        contenders_per_track[ti] = contenders_per_track.get(ti, 0) + 1
        contenders_per_key[key] = contenders_per_key.get(key, 0) + 1

    # Pass 2: apply the (possibly density-raised) threshold to qualify_sim
    # (appearance alone decides eligibility - heading/orientation only
    # influence which already-eligible candidate wins, below).
    pairs = []  # (rank_sim, ti, key, margin_for_this_pair)
    for ti, key, qualify_sim, rank_sim in spatial_pairs:
        extra_contenders = max(contenders_per_track[ti], contenders_per_key[key]) - 1
        eff_threshold = reid_threshold + density_threshold_step * extra_contenders
        eff_margin = reid_margin + density_margin_step * extra_contenders
        if qualify_sim < eff_threshold:
            continue
        if negative_sims is not None and key in negative_sims:
            neg = negative_sims[key]
            if qualify_sim - neg < negative_margin:
                if veto_log is not None:
                    veto_log.append(("bystander", ti, key, qualify_sim, neg))
                continue
        pairs.append((rank_sim, ti, key, eff_margin))
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


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection_area(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    inter = _bbox_intersection_area(box_a, box_b)
    union = _bbox_area(box_a) + _bbox_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def suppress_contained_detections(
    detections: list[Detection], containment_threshold: float = 0.9
) -> list[Detection]:
    """Drop a detection whose bbox is almost entirely covered by a larger
    detection's bbox - a duplicate/redundant detection of the same person,
    not a second person.

    In a tightly-seated group, the detector sometimes emits both a
    full-body box for someone *and* a second, smaller box for just part of
    them (e.g. upper-body-only) that the detector's own NMS doesn't merge -
    plain IoU-based NMS compares intersection-over-*union*, which stays low
    whenever the two boxes differ a lot in size, even if the smaller one is
    almost entirely inside the larger one. This instead compares
    intersection-over-the-*smaller*-box's-area (unaffected by the size
    difference), which correctly flags that case. Confirmed against real
    footage: two boxes for the same person with only ~49% IoU (comfortably
    below any normal NMS threshold) at 97% containment - this function
    drops the second one, standard NMS wouldn't have.

    Deliberately conservative (default 0.9): two people sitting close enough
    for their boxes to partially overlap is normal and must not be
    suppressed - only a near-total containment is confidently a duplicate,
    not a second real person.
    """
    order = sorted(range(len(detections)), key=lambda i: _bbox_area(detections[i].bbox_xyxy), reverse=True)
    keep = [True] * len(detections)
    for a in range(len(order)):
        i = order[a]
        if not keep[i]:
            continue
        for b in range(a + 1, len(order)):
            j = order[b]
            if not keep[j]:
                continue
            smaller_area = _bbox_area(detections[j].bbox_xyxy)
            if smaller_area <= 0:
                continue
            inter = _bbox_intersection_area(detections[i].bbox_xyxy, detections[j].bbox_xyxy)
            if inter / smaller_area >= containment_threshold:
                keep[j] = False
    return [d for d, k in zip(detections, keep) if k]


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
        orientation_relief: float = 0.12,
        containment_suppression_threshold: float = 0.9,
        torso_rescue_weight: float = 0.4,
        reconcile_heading_rank_weight: float = 0.15,
        reconcile_orientation_rank_weight: float = 0.12,
        cooccurrence_veto: bool = True,
        cooccurrence_max_overlap: float = 0.5,
        bystander_negative_check: bool = False,
        bystander_margin: float = 0.0,
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
        self.orientation_relief = orientation_relief
        self.containment_suppression_threshold = containment_suppression_threshold
        self.torso_rescue_weight = torso_rescue_weight
        self.reconcile_heading_rank_weight = reconcile_heading_rank_weight
        self.reconcile_orientation_rank_weight = reconcile_orientation_rank_weight
        self.cooccurrence_veto = cooccurrence_veto
        self.cooccurrence_max_overlap = cooccurrence_max_overlap
        self.bystander_negative_check = bystander_negative_check
        self.bystander_margin = bystander_margin
        # Diagnostics: one dict per pair a bystander constraint rejected that
        # would otherwise have passed the ReID threshold.
        self.veto_events: list[dict] = []
        # {abandoned_track_id: surviving_track_id} for every reconciliation
        # merge so far - a caller that already published `abandoned_track_id`
        # knows it was retroactively superseded, not actually a separate person.
        self.merged_ids: dict[int, int] = {}

    def update(self, detections: list[Detection], frame_area: float, dt_seconds: float) -> list[Track]:
        detections = suppress_contained_detections(detections, self.containment_suppression_threshold)

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
        # Same gating: an orientation read off a crop too degraded to trust
        # for ReID at all isn't trustworthy either - UNKNOWN grants no relief.
        orientations: list[str] = [
            estimate_orientation(d.keypoints) if p else ORIENTATION_UNKNOWN for d, p in zip(detections, plausible)
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
                orientation=orientations[di],
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
            (
                ti,
                _bbox_center(motion_tracks[ti].last_bbox),
                gated_embeddings[di],
                gated_torso_embeddings[di],
                orientations[di],
            )
            for ti, di in track_to_det.items()
            if motion_tracks[ti].age_seconds <= self.reconcile_window_s
        ]
        reconcile_blocked: frozenset[tuple[int, int]] = frozenset()
        if self.cooccurrence_veto:
            reconcile_blocked = frozenset(
                (pi, young_ti)
                for young_ti, *_rest in reconcile_candidates
                for pi, lost in enumerate(reid_pool)
                if lost.uid in motion_tracks[young_ti].co_visible
            )
        reconcile_negatives = None
        if self.bystander_negative_check:
            reconcile_negatives = {
                young_ti: self._negative_similarity(
                    emb, temb, [motion_tracks[ti] for ti in track_to_det if ti != young_ti]
                )
                for young_ti, _center, emb, temb, _orientation in reconcile_candidates
                if emb is not None
            }
        reconcile_veto_log: list = []
        reconcile_matches = _find_reid_matches(
            reid_pool,
            reconcile_candidates,
            self.max_speed_bbox_heights_per_sec,
            self.reid_threshold,
            self.reid_margin,
            dt_seconds,
            density_threshold_step=self.density_threshold_step,
            density_margin_step=self.density_margin_step,
            # orientation_relief (the threshold-affecting kind) stays 0 here,
            # unlike stage 2 below. The relief exists to recognize a person
            # *reappearing* after an absence at a different orientation than
            # their gallery sample (see human_tracking/orientation.py) - but
            # a reconciliation candidate isn't absent, it's a track with a
            # live, currently motion-matched detection this very frame.
            # Granting the same "maybe they turned around while gone" relief
            # as a free pass through the threshold here has no such
            # justification, and real-footage testing found it was the
            # deciding factor in false merges between two different,
            # simultaneously-visible people of different orientations (e.g.
            # one facing the robot, one with their back to it - an ordinary
            # multi-person framing, not a rare edge case) - see
            # docs/context/human_tracking.md. Orientation and heading are
            # still used here, just only as ranking tie-breakers between
            # already-qualified candidates - see
            # reconcile_orientation_rank_weight/reconcile_heading_rank_weight
            # below and _find_reid_matches' docstring.
            orientation_relief=0.0,
            torso_rescue_weight=self.torso_rescue_weight,
            heading_rank_weight=self.reconcile_heading_rank_weight,
            orientation_rank_weight=self.reconcile_orientation_rank_weight,
            blocked_pairs=reconcile_blocked,
            negative_sims=reconcile_negatives,
            negative_margin=self.bystander_margin,
            veto_log=reconcile_veto_log,
        )
        for kind, pi, young_ti, sim, neg in reconcile_veto_log:
            self.veto_events.append(
                {
                    "kind": kind,
                    "stage": "reconcile",
                    "lost_id": reid_pool[pi].track_id,
                    "young_id": motion_tracks[young_ti].track_id,
                    "sim": round(sim, 3),
                    "neg": None if neg != neg else round(neg, 3),
                }
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
                orientation=orientations[di],
                snap_kalman=True,
            )
            if young_track.track_id is not None:
                self.merged_ids[young_track.track_id] = reclaimed_track.track_id
            # Same person, so everyone seen alongside the young track is a
            # different person from the reclaimed identity too.
            reclaimed_track.co_visible |= young_track.co_visible
            for other in self.tracks:
                if young_track.uid in other.co_visible:
                    other.co_visible.add(reclaimed_track.uid)
            reconciled_young_tracks.add(young_track)

        # --- stage 2: ReID matching for whatever's left truly unmatched ---
        det_candidates = [
            (ui, _bbox_center(d.bbox_xyxy), gated_embeddings[di], gated_torso_embeddings[di], orientations[di])
            for ui, (di, d) in enumerate(unmatched_detections)
        ]
        stage2_negatives = None
        if self.bystander_negative_check:
            reclaimed_by = {young_ti: reid_pool[pi] for pi, young_ti in reconcile_matches.items()}
            visible_now = [reclaimed_by.get(ti, motion_tracks[ti]) for ti in track_to_det]
            stage2_negatives = {
                ui: self._negative_similarity(emb, temb, visible_now)
                for ui, _center, emb, temb, _orientation in det_candidates
                if emb is not None
            }
        stage2_veto_log: list = []
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
            orientation_relief=self.orientation_relief,
            torso_rescue_weight=self.torso_rescue_weight,
            negative_sims=stage2_negatives,
            negative_margin=self.bystander_margin,
            veto_log=stage2_veto_log,
        )
        for kind, pi, _ui, sim, neg in stage2_veto_log:
            self.veto_events.append(
                {
                    "kind": kind,
                    "stage": "reid",
                    "lost_id": reid_pool[pi].track_id,
                    "young_id": None,
                    "sim": round(sim, 3),
                    "neg": round(neg, 3),
                }
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
                orientation=orientations[di],
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
                orientation=orientations[di],
            )
            self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if t.is_alive and t not in reconciled_young_tracks]
        self._record_cooccurrence([t for t in self.tracks if t.seconds_since_last_seen == 0.0])
        return [t for t in self.tracks if t.is_published]

    def _record_cooccurrence(self, seen_this_frame: list[Track]) -> None:
        """Mark every pair of tracks matched to separate detections this frame
        as known-different people. Pairs whose boxes overlap by more than
        `cooccurrence_max_overlap` (intersection over the smaller box) are
        skipped: a fragment/duplicate box of one person that escaped
        containment suppression, or two people mid-crossing, is not clean
        evidence - and they'll co-occur in plenty of clean frames anyway if
        they really are two people."""
        for i, a in enumerate(seen_this_frame):
            for b in seen_this_frame[i + 1 :]:
                if b.uid in a.co_visible:
                    continue
                smaller = min(_bbox_area(a.last_bbox), _bbox_area(b.last_bbox))
                if smaller <= 0:
                    continue
                if _bbox_intersection_area(a.last_bbox, b.last_bbox) / smaller > self.cooccurrence_max_overlap:
                    continue
                a.co_visible.add(b.uid)
                b.co_visible.add(a.uid)

    def _negative_similarity(
        self, emb: np.ndarray | None, temb: np.ndarray | None, others: list[Track]
    ) -> float:
        """Best similarity of a query to tracks known not to be it (visible on
        another detection this same frame) - same scoring as the stage that
        uses it, minus orientation relief (no absence to excuse here)."""
        if emb is None:
            return -1.0
        return max(
            (t.best_similarity(emb, temb, torso_rescue_weight=self.torso_rescue_weight) for t in others if t.gallery),
            default=-1.0,
        )
