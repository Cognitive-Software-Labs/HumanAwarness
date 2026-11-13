# Human tracking / ReID (persistent identity across occlusion)

## What it is

Takes raw per-frame person detections (from `human_perception`) and turns
them into *persistent identities* that survive brief occlusion or the
person temporarily leaving the camera's view — so that "awareness" can be
tracked as a property of a person over time, not reset every time
detection drops out for a frame or two.

## Status

Wired into a real ROS2 node (`human_tracking_node`), buildable with
`colcon` (`package.xml`/`setup.py` added), and verified end-to-end: a
synthetic Image + `PersonDetectionArray` pair published in-process to the
node correctly promoted a track to `ACTIVE` after exactly `confirm_frames`
(3) frames and republished it with the right `track_id`/`state`/bbox/
keypoints. Core logic is still also validated via the standalone test
script (`scripts/test_human_tracking_on_video.py`), which remains the
primary tool for testing against real video.

## Lifecycle

```
NEW --(3 consecutive confirmed frames)--> ACTIVE
ACTIVE --(missed detection)--> LOST_KALMAN
LOST_KALMAN --(re-matched by motion/IoU, or by ReID as a fallback)--> ACTIVE
LOST_KALMAN --(timeout, default 2.5s)--> LOST_GALLERY
LOST_GALLERY --(re-matched by ReID)--> ACTIVE
LOST_GALLERY --(timeout, default 45s)--> FORGOTTEN (track dropped)
```

`NEW` and `FORGOTTEN` are never published — a `track_id` only "exists"
externally once confirmed (`ACTIVE` or a still-recoverable `LOST_*`),
matching the idea that awareness state belongs to a confirmed identity,
not a raw unconfirmed detection blip.

On top of this, any `NEW` or `ACTIVE` track can still lose its current
detection to an older `LOST_KALMAN`/`LOST_GALLERY` identity via ReID, for
`reconcile_window_s` (default 1.5s) after its own creation — see
"Reconciliation" below. When that happens the young track is discarded
entirely and the reclaimed identity takes its place; a `track_id` a
caller already saw published can therefore be superseded by an older one
within that window.

## Design decisions

- **Containment-based duplicate-detection suppression** (`suppress_contained_
  detections`, run first thing in `Tracker.update`): drops a raw detection
  whose bbox is almost entirely covered by a larger detection's bbox -
  confirmed via real multi-person footage (see "Multi-person validation"
  below) to be a duplicate/fragment detection of the same person, not a
  second one. In a tightly-seated group, the detector can emit both a
  full-body box for someone *and* a second, smaller box for just part of
  them (e.g. upper-body-only) that its own NMS doesn't merge - plain NMS
  compares intersection-*over-union*, which stays low whenever the two
  boxes differ a lot in size even though the smaller one is almost entirely
  inside the larger one. This compares intersection-over-the-*smaller*-
  box's-area instead, which isn't thrown off by the size difference.
  Deliberately conservative (default `containment_suppression_threshold`
  0.9): two people sitting close enough for their real, separate boxes to
  partially overlap is normal and must never be suppressed - only
  near-total containment is confidently a duplicate.
- **Motion model**: a minimal vendored constant-velocity Kalman filter
  (`kalman_box_tracker.py`) on `[cx, cy, w, h]` + their velocities. Not a
  library (`filterpy` had the same thin/stale-metadata red flags as
  `torchreid` below) — this is ~50 lines of standard, well-understood math
  every SORT-family tracker reimplements the same way.
- **ReID model**: OSNet (`osnet_x1_0`), vendored directly from
  [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)
  (MIT-licensed, see `osnet.py`'s header) rather than installed via the
  `torchreid` PyPI package - that package's `__init__` eagerly imports a
  full training framework (dataset loaders, `tensorboard`, etc.) just to
  get the model definition, surfacing multiple undeclared missing
  dependencies. Weights: `osnet_x1_0` trained on **MSMT17** (the largest,
  most diverse ReID benchmark - chosen over higher-scoring same-dataset
  checkpoints because it generalizes better to an unseen camera, which is
  what actually matters here), downloaded via `gdown` from the official
  model zoo on first use, cached at `~/.cache/human_tracking/`.
- **Two-stage lost-track matching** (`tracker.py`): motion/IoU first
  (cheap), falling back to ReID both for `LOST_GALLERY` tracks *and* for
  `LOST_KALMAN` tracks whose motion match just failed this frame - added
  after finding that a real pre-occlusion detection is often noisy (a
  shrinking/jittering box right as someone starts being hidden), which a
  constant-velocity Kalman filter can extrapolate into a wildly wrong
  predicted position in as little as ~2 seconds, failing the IoU match
  even though the person reappeared somewhere reasonable.
- **ReID match guards**, beyond a plain similarity threshold: a spatial
  gate (a lost track can only match a detection within a plausible
  walking distance, scaled by its own size and how long it's been lost)
  and a similarity margin (a candidate must clearly beat whatever else
  that track/detection could instead match to; an ambiguous near-tie is
  left unmatched rather than guessed at). Added after real crowd-scene
  testing showed a bare threshold produces false-positive merges once
  several lost tracks are simultaneous candidates.
- **Crop plausibility gate** (`is_plausible_person_crop`): before a
  detection's embedding is allowed to contribute to a gallery or act as a
  ReID candidate, require a minimum number of confident pose keypoints
  spread across a reasonable fraction of the box's height. **This went
  through two prior versions based on bbox *shape* (aspect ratio, then
  also height-as-fraction-of-frame) - both were replaced** after real
  testing found cases a shape-only check can't tell apart: a box clipped
  by the frame edge during a legitimate exit/re-entry looks geometrically
  identical (narrow, tall, boundary-touching) to a genuinely degenerate
  detection (e.g. a stray arm/bag fragment). No single shape threshold
  resolved both; using the actual keypoint data instead of inferring body
  shape from the box does.
- **Gallery**: up to 8 embeddings per track, kept by *quality* (confidence
  × crop size, see `crop_quality`) rather than first-seen-first-kept, so a
  track's stored appearance stays representative even if its first few
  samples were poor.
- **Letterbox-preserving resize** (`reid_embedder.py`): every crop is
  resized into OSNet's fixed 256x128 input by scaling to fit and padding
  with black, not by stretching to fill it. A plain stretch distorts a
  partial-body crop's proportions (e.g. a half-body view gets squished
  into a full-body-shaped box) in a way OSNet never saw in training -
  ReID benchmarks are almost all full-body crops. Measured effect on a
  real partial-view reappearance: cosine similarity 0.490 -> 0.508 - real
  but modest, since it fixes distortion, not missing information.
- **Torso-only region matching** (`torso_crop_bbox`, `Track.best_similarity`):
  alongside the full-crop embedding, a second embedding is computed from
  just the shoulders-to-hips region (derived from pose keypoints) when
  confident shoulder+hip keypoints exist. A track's gallery stores both;
  matching compares both and takes the max. This is a pragmatic
  approximation of part-based/visibility-aware ReID (not a retrained part
  model) aimed at the case where a full-body gallery sample and a
  partial/edge-clipped reappearance aren't comparable as whole-crop
  embeddings, but both still show *some* torso. It doesn't help when the
  pose model itself can't place shoulder/hip keypoints on the reappearance
  crop at all (confirmed on a real case - see Reconciliation below), since
  there's then no torso region to extract on that side either.
- **Keypoint-based orientation fusion** (`orientation.py`, used in
  `Track.best_similarity`): a real reappearance after 1-2s away often comes
  back at a different body orientation than it left in (e.g. walked off
  facing the camera, walked back in with their back to it) - OSNet was
  never trained to make its embedding viewpoint-*invariant*, so a genuine
  same-person match across that flip can legitimately score well below
  `reid_threshold`, which looks identical to "different person" to a bare
  threshold. Considered (and rejected) **6DRepNet-style head pose** for this
  first: it needs a detected face to run at all, so it has nothing to say
  exactly when the face isn't visible - the back-of-head case that's
  actually failing. Used instead: a classifier over the same COCO-17
  keypoints already available (`estimate_orientation` - FRONTAL if the face
  is actually visible, BACK if only the back-of-head/ear keypoints are,
  PROFILE for one eye/ear, UNKNOWN otherwise), no new model/dependency, and
  it still produces an answer in the no-face case. When a query and a
  specific gallery entry have confidently-known, *different* orientations,
  that one comparison gets a similarity bonus (`orientation_relief`, scaled
  down for a partial PROFILE mismatch vs. a full FRONTAL/BACK flip) before
  the threshold/margin check - additive, per-pair, and zero whenever either
  side is UNKNOWN or they match, specifically so it doesn't repeat the
  earlier bare-threshold-lowering mistake (see `reid_threshold` section
  below) of making unrelated same-orientation candidates easier to match
  too. Synthetic validation (`Tracker` exercised directly, no real footage
  available - see "Not done yet"): a controlled cross-orientation
  reappearance at a similarity the plain threshold rejects is correctly
  reclaimed with the relief enabled and correctly still lost without it;
  the same setup with a second, spatially-plausible same-orientation
  distractor track correctly still picks the right identity, not the
  distractor, confirming the relief doesn't erode the margin-based
  disambiguation that's keeping crowd scenes from collapsing.
  **That synthetic validation had a real blind spot, found on real footage:
  it only checked a same-orientation distractor, never a *different*-
  orientation one.** A 3-person handheld clip (see "Multi-person
  validation" below) produced a confirmed false merge between two
  different real people (a man, FRONTAL, and a woman with her back
  turned, BACK) where `orientation_relief` was the deciding factor -
  instrumented directly: raw similarity 0.570 (correctly below
  `reid_threshold`), +0.12 relief for the FRONTAL/BACK mismatch pushed it
  to 0.690, accepted. The relief has no way to tell "same person, turned
  around while gone" apart from "two different people who happen to face
  different ways" - and in any scene with more than one person, the
  latter is at least as likely as the former, since people don't all
  face the camera at once. **Fix**: `orientation_relief` is no longer
  applied during the reconciliation stage (1.5) specifically - only in
  stage 2 (`Tracker.update`). Reconciliation's "young" candidate always
  has a live, currently motion-matched detection this very frame; it was
  never absent, so the relief's own justification (a genuine reappearance
  after being gone) doesn't apply there the way it does in stage 2, where
  the track being matched against really has been gone. Re-validated: the
  specific man/woman identity *swap* (id churning back and forth between
  two different real people) is gone from the 3-person clip after this
  fix, and all three previously-validated single-person clips plus the
  2-person clip (see "Multi-person validation") still net to the correct
  identity count - no regression.
- **Reconciliation stage** (`Tracker.update`'s "stage 1.5"): a real
  phone-recorded exit/re-entry test showed that a person's first
  reappearance frame can have keypoints too degraded for ReID to
  recognize them (shoulder/hip confidence 0.02-0.12 - torso matching
  above can't help either), while the *same* track's keypoints become
  confident again just ~7 frames (~0.2s) later as they walk further into
  frame. Without this stage, that's already too late: a brand-new track
  claims the detection on the bad frame and then wins every subsequent
  motion/IoU match, so the original lost identity never gets a second
  chance. The fix: for `reconcile_window_s` (default 1.5s) after a
  track's creation, it can still lose its current detection to an older
  `LOST_KALMAN`/`LOST_GALLERY` identity via the same ReID spatial-gate +
  margin logic used elsewhere - if that happens, the young track (and any
  `track_id` it may have already had assigned) is discarded and the
  reclaimed identity takes its place. Verified on the real clip: 2
  confirmed ids -> 1 reconciled -> 1 net identity, with zero regressions
  on the two previously-correct clips and zero false reconciliations in a
  5-person clip. **Not re-tested against the crowded-scene stress case**
  (that footage was deleted earlier) - a second ReID-matching opportunity
  per track is, in principle, a second chance for a wrong match too, even
  though none appeared in the tests run so far.

## Parameters (`Tracker.__init__`)

| param | default | meaning |
|---|---|---|
| `confirm_frames` | 3 | consecutive hits needed for `NEW` -> `ACTIVE` |
| `kalman_timeout_s` | 2.5 | `LOST_KALMAN` -> `LOST_GALLERY` |
| `gallery_timeout_s` | 45.0 | `LOST_GALLERY` -> `FORGOTTEN` |
| `iou_threshold` | 0.3 | motion-match acceptance |
| `reid_threshold` | **0.65** | ReID match acceptance (see below) |
| `reid_margin` | 0.08 | required similarity gap over the runner-up candidate |
| `max_speed_bbox_heights_per_sec` | 3.0 | spatial gate speed limit, in bbox-heights/sec |
| `gallery_size` | 8 | embeddings kept per track |
| `reconcile_window_s` | 1.5 | how long a new/young track can still lose its detection to an older identity via ReID |
| `density_threshold_step` | 0.05 | extra `reid_threshold` required per additional simultaneous spatial contender |
| `density_margin_step` | 0.05 | extra `reid_margin` required per additional simultaneous spatial contender |
| `orientation_relief` | 0.12 | similarity bonus for a ReID comparison where query and gallery sample have confidently-known, different keypoint-derived orientations (scaled down for a partial PROFILE mismatch - see "Design decisions") |
| `containment_suppression_threshold` | 0.9 | drop a raw detection this fraction or more contained within a larger one - see "Design decisions" |
| `torso_rescue_weight` | 0.4 | how much a higher per-entry torso-only similarity can lift a lower full-crop similarity (0 = full-crop only, 1 = torso can fully override) - see "Multi-person validation" |
| `reconcile_heading_rank_weight` | 0.15 | during reconciliation only: how much a lost track's Kalman-velocity heading can shift ranking (not eligibility) between already-qualified competing candidates - see "Multi-person validation" |
| `reconcile_orientation_rank_weight` | 0.12 | during reconciliation only: same, for keypoint-derived body orientation consistency |

## Multi-person validation (real footage, not synthetic)

First real multi-person test: a 4-person seated meeting clip (free-licensed
stock footage), people close enough together that their bodies/bboxes
overlap significantly. Investigated two distinct symptoms reported from
watching the tracked output: people sometimes not detected despite being
clearly visible, and a person's id occasionally changing even though they
never left frame.

- **Root-caused by instrumenting the tracker directly** (frame-by-frame
  state transitions, raw per-frame detection counts/boxes), not guessed:
  - *"Not detected"*: mostly plain detector recall in a tightly-seated,
    overlapping-bodies scene - only 173/848 frames (20%) had all 4 people
    detected at all; most frames had 2-3. This mostly self-heals via the
    existing `LOST_KALMAN` grace period (same id preserved across a brief
    gap) - a detector-capacity/occlusion limitation (`yolo26n-pose` is the
    smallest variant), not a tracking logic bug. The existing `det_model`
    parameter (not touched here) is the designed lever if this needs
    improving - a larger variant trades speed for exactly this kind of
    recall.
  - *"Random new id"*: a genuine bug, confirmed by tracing one instance -
    a track confirmed at frame 2, matched perfectly every single frame
    with zero gaps, got silently reassigned to a different track's id at
    frame 22, 20 frames (0.67s) into a perfectly clean life. Traced to
    *duplicate raw detections*: the detector emitted a second, redundant
    box (97% contained within a larger one, for the same physical person)
    that the tracker's IoU/ReID matching treated as a real second
    candidate, destabilizing which track "owned" that person.
- **First fix attempt failed, and was reverted** - gating reconciliation
  eligibility on `consecutive_hits` (requiring a young track to still be
  *freshly* confirmed, not just young by wall-clock `age_seconds`, see
  stage 1.5's docstring) seemed principled, but regression-tested *worse*:
  net distinct identities on the same clip went from 4 (correct) to 8
  (badly fragmented). This proved the original reconciliation mechanism was
  mostly doing legitimate, necessary cleanup work in this clip, not causing
  false merges - the actual problem was upstream, in the raw detections
  feeding it, not in reconciliation's own eligibility logic. Reverted
  immediately on this evidence rather than kept "because it seemed
  reasoned" - a reasoned hypothesis still has to survive contact with real
  footage, same standard as every other change in this file.
- **Actual fix**: `suppress_contained_detections` (see "Design decisions").
  Re-validated on the same clip: **net 4 distinct identities (matching the
  4 real people exactly), zero reconciliation events needed at all, zero id
  churn** - every person kept one stable id for the entire clip. Re-ran the
  two single-person clips (`whatsapp2`/`whatsapp3`) used throughout this
  file's earlier validation to confirm no regression there - identical
  results to before (nothing to suppress when there's no overlapping boxes
  to begin with).
- **Not yet tested**: a real crowd (10+ simultaneous people) - this clip is
  4 people at a table, a meaningfully different regime from the
  Shibuya-crossing-style stress case the `reid_threshold` section below
  describes (and that footage no longer exists to re-test against). The
  crowd-scene false-merge risk documented there remains open.
- **A second, separate issue surfaced by the same clip, display-side, not a
  tracking bug**: `scripts/test_tracking_and_head_pose_on_video.py` only
  ever drew `ACTIVE` tracks, so two of the four people (the two with the
  least reliable raw detections - see above) appeared to "not be detected"
  on-screen far more often than they were actually lost internally.
  Measured directly: one person was tracked (published, some state) 100% of
  the video but drawn only 37% of the time; the other, 90% tracked but
  drawn only 5% of the time. Fixed in that script (not in `Tracker` itself
  - this was purely about what the demo chooses to draw): it now also
  draws `LOST_KALMAN` tracks, using `t.kalman.bbox` (the live, continuously
  re-predicted position - already updated every frame regardless of
  whether a detection matched, that's the whole point of the Kalman step)
  rather than the frozen `last_bbox`, labeled distinctly ("(predicted)", a
  different color, no head-pose attempt since there's no fresh crop that
  frame). `LOST_GALLERY` stays undrawn - by then the motion estimate is too
  stale to responsibly show. Re-measured: the two people went from 37%/5%
  visible to 91%/48% - the remaining gap for the second person is genuine
  extended occlusion (`LOST_GALLERY`), which a motion-prediction display
  can't responsibly paper over; that part would need better raw detection
  (see `det_model` above), not a display change.
- **The actual detection-level fix, since the display fix alone wasn't
  enough** (correctly pushed back on - it only reveals what the tracker
  already knew, it doesn't get the detector to see these two people more
  often). First had to resolve an apparent contradiction: the two "hard"
  people looked completely unobstructed to the eye, not blocked by the
  other two or by furniture. Checked the raw frames directly rather than
  assume - from this particular side-on camera angle, each "hard" person
  sits almost directly behind their neighbor along the same sightline, so
  their face stays visible but their torso/hips (what a pose model actually
  needs to separate two people) are mostly hidden behind the nearer
  person's body - real occlusion, just not the kind a viewer's eye
  registers as "blocked," since a person mentally fills in bodies their
  eye can partially see. Confirmed in the raw detector output: on a typical
  frame, instead of 4 separate boxes there were 2, each spanning *both*
  people in a pair as one merged blob.
  Benchmarked the actual levers (not assumed) on this clip - model size,
  confidence threshold, and detector input resolution (`imgsz`, newly
  exposed - it didn't exist as a parameter anywhere before this), alone and
  combined, measuring the two hard people's `ACTIVE` percentage and net
  identity count/stability each time:

  | config | hard person A | hard person B | net ids | reconciliations |
  |---|---|---|---|---|
  | nano, conf 0.5, imgsz 640 (baseline) | 37% | 5% | 4 | 0 |
  | small, conf 0.5, imgsz 640 | 85% | 12% | 4 | 1 |
  | nano, conf 0.25, imgsz 640 | 90% | 15% | 4 | 1 (+1 spurious id, self-cleaned) |
  | nano, conf 0.5, imgsz 960 | 94% | 15% | 4 | 2 |
  | small, conf 0.5, imgsz 960 | 100% | 23% | 4 | 0 |
  | **small, conf 0.5, imgsz 1280** | **100%** | **35%** | **4** | **0** |
  | small, conf 0.5, imgsz 1536 | 13% | 15% | 4 | 0 |

  Takeaways, all found empirically rather than assumed:
  - **Lowering `conf` is the wrong lever** - it recovers some real
    detections but also introduces spurious single-frame ghost ids (caught
    and self-cleaned by reconciliation here, but not a dependable outcome -
    see the crowd-scene false-merge risk documented above).
  - **Bigger input resolution is not monotonically better** - 1536 was
    measured *worse* than 1280 on this exact clip (13%/15% vs 100%/35%),
    most likely an anchor/stride-scale mismatch past some point, not
    "more detail always helps." Don't assume a higher number is safer
    without re-measuring.
  - **`small` model + `imgsz=1280` was the clear winner** on every axis
    measured - best detection recall for both hard people, zero
    reconciliation events needed (the most stable result of any
    configuration tested, including the baseline), and effectively no
    speed cost on this dev GPU (106-114 fps across every variant tested,
    vs. the baseline's 102 fps).
  - **Not yet re-verified on target robot hardware.** All of this was
    measured on a development GPU; a larger model and higher input
    resolution cost more compute and memory, and this project's own
    earlier GPU-vs-CPU benchmarking (see OSNet above) already shows real
    hardware-dependent swings - an embedded/Jetson-class device may not
    sustain the same headroom. Re-measure fps on actual deployment
    hardware before adopting these values there.
  - **Deliberately left as opt-in parameters, not a changed shipped
    default** - `model_path`/`det_model` and the newly-added `imgsz`
    keep their original defaults (`yolo26n-pose.pt`, `640`) in
    `human_perception`'s node/config and both test scripts. Whether to
    adopt `yolo26s-pose.pt` + `imgsz=1280` as the new default is a
    deployment-hardware tradeoff only whoever picks the target hardware
    can make - not something to flip silently.

- **3-person handheld selfie-style clip - confirmed the crowd-scene
  false-merge risk documented below isn't actually crowd-exclusive.**
  Earlier validation only checked the false-merge risk against a dense
  10+-person crowd and concluded the 0.65 `reid_threshold` tradeoff "is
  still justified" for "a single-or-few-person robot-awareness scenario
  (the actual target use case)" (see `reid_threshold` section below) -
  this clip is exactly that target use case (3 people, close range) and
  still produced a clean, confirmed identity swap between two different
  real people (a man and a woman) within under a second of them crossing
  paths. Root-caused with direct instrumentation (raw similarity scores,
  not visual guessing) to two separate contributing causes:
  1. `orientation_relief` being wrongly applicable during reconciliation -
     fixed, see the orientation-fusion entry above.
  2. Raw OSNet similarity alone (no relief at all, both sides FRONTAL)
     scored 0.653 for two different real people - just over
     `reid_threshold` - during a ~2-frame (0.07s) occlusion as all three
     people were close together. This is the same embedding-reliability
     limit the crowd-scene testing already found, just newly confirmed to
     also occur with only 2-3 people, not only in a dense crowd - **not
     fixed here**, since the existing `reid_threshold`/margin tradeoff was
     already deliberately tuned against real conflicting evidence (see
     below) and silently retuning it again without the same rigor would
     repeat the mistake the docs already warn against.
  After fix 1, the specific back-and-forth identity *swap* is gone (the
  man's track no longer gets handed to the woman and back), but the
  underlying cause-2 instability remains: one person's track can still
  briefly get reassigned to a different person's detection during a close
  pass, then self-correct (or not) via a later reconciliation, now
  manifesting as the *same* practical failure mode documented below under
  "Not done yet" rather than a new one. Net distinct identities on this
  clip went from 3 (two of which were actually scrambled composites of
  different real people, not genuinely separate/correct counts) to 4 for
  3 real people (one person briefly double-counted) - a real regression
  in *recall* on this specific clip, but a safer failure mode for the
  awareness use case: an extra fragment never misattributes one person's
  gaze/attention to someone else, where the pre-fix swap did exactly that.

- **A second 3-person clip reproduced cause-2 from above directly, with a
  user-observed, concrete symptom: two people's original ids (`1` and `4`)
  ended up swapped after both briefly left frame.** Root-caused with the
  same instrumentation style used throughout this file (actual pool-track
  velocity, candidate position, and similarity at the exact match, not
  guessed from the video): the wrong match (candidate 504px away, ~88% of
  the frame's width, appearing from almost the exact opposite direction of
  the lost track's last heading - cosine -0.96 against its Kalman
  velocity) scored 0.653 similarity with **zero** orientation relief
  involved (both sides FRONTAL) - a second, independent confirmation that
  raw ReID similarity alone can cross `reid_threshold` for two different
  real people in an ordinary few-person scene, same as the first clip.
  - **Tried and reverted: a heading gate** (reject/shrink a candidate's
    allowed spatial radius if it's behind the lost track's last-known
    direction of travel, read from the Kalman filter's frozen velocity at
    the moment it was lost - `kalman.x[4:6]`, unaffected by the filter's
    own well-documented tendency to run away on a long-lost prediction,
    since pure `predict()` calls never change the velocity estimate
    itself). This *did* fix the specific reported case (re-verified: the
    reappearing person's detection correctly reconciled back into their
    own original track instead of stealing someone else's id) - but
    regression-testing against the single-person clips (standard practice
    in this file before trusting any tracker change) found it broke an
    already-correct case: the same real person in a different clip simply
    turned around and walked back the way they'd come, which produced a
    nearly identical directional signature (cosine -0.90 to -1.00 against
    their own prior heading, legitimate similarity up to 0.89) to the
    false-merge case the gate was built to catch. **No heading-only signal
    separated "same person turned around" from "different person,
    coincidentally arriving from the opposite direction" in the evidence
    gathered** - both look the same to a pure direction check, and turning
    around to walk back is itself an ordinary, common reappearance
    pattern, not a rare edge case worth breaking for this fix. Reverted
    immediately on that evidence, same standard as every other change in
    this file (see the earlier reverted `consecutive_hits` and
    self-similarity-baseline attempts above).
  - **Found and fixed: the actual driver was the torso-only comparison, not
    the full-crop one.** Rather than guess at a second fix, dumped every
    individual gallery entry's similarity (not just the winning max) for
    the wrongly-matched track at the exact frame: all 8 entries' full-crop
    similarity sat at 0.51-0.58 - consistently *below* `reid_threshold`,
    correctly disagreeing - while every one of the same 8 entries' torso-
    only similarity read 0.65-0.76, comfortably over threshold.
    `Track.best_similarity` took a plain `max(full_sim, torso_sim)` per
    entry, so the weaker, more generic signal (torso: just shirt color/
    texture, no face or hair) could single-handedly override the stronger
    one whenever it happened to read higher - exactly backwards, and with
    every gallery entry agreeing (not one stale outlier), ruling out
    recency-weighting as a fix for this specific failure (there's no
    "old entry dragging it up" pattern to downweight).
    **Fix**: torso now only *partially* rescues a low full-crop score -
    `full_sim + torso_rescue_weight * max(0, torso_sim - full_sim)`
    (`torso_rescue_weight` default 0.4) - instead of a plain `max`. Full
    embedding stays the primary vote; torso can pull a score up toward
    itself but not replace it outright, keeping its original purpose
    (rescuing a genuinely partial/edge-clipped crop comparison) while no
    longer letting a merely-higher torso reading override a full-crop
    embedding that's already confidently saying "different person."
    Re-verified on the clip that found this: the reappearing person's
    detection now correctly creates/rejoins her own original track instead
    of absorbing someone else's id, through to the end of the clip.
    Regression-tested against every other validated clip in this file
    (all three single-person clips, the 2-person clip, and the first
    3-person clip above) - all still net the same identity counts as
    before this change, no regression.
  - **Net**: this specific id-swap failure mode is confirmed, root-caused,
    and fixed for the case found - via the gallery/torso mechanism above,
    not the heading gate (reverted) or recency-weighting (ruled out by the
    same-magnitude-across-all-entries evidence). Still only validated
    against the one real false-merge case plus the existing regression
    suite - not a guarantee against every possible false merge (e.g. two
    people in genuinely similar full-body appearance, not just similar
    torsos, would not be caught by this fix) - the base `reid_threshold`/
    embedding-discrimination limitation described below remains real.
  - **The heading gate's own failure mode - no way to tell "turned around"
    from "different person" by direction alone - had a missing piece: it
    was evaluated as an absolute veto on one candidate, but the real case
    that found it had *two* lost tracks (`id 1` and `id 4`, both real,
    different people) simultaneously spatially- and appearance-plausible
    for the very same reappearing detection - confirmed directly in the
    data (both printed as live candidates for the same `candidate_key` on
    the same frames). That's a meaningfully different question: not "is
    this candidate plausible," but "which of two already-plausible
    candidates is it more likely to be" - direction is real, useful
    evidence for the second question even though it says nothing reliable
    about the first. **Re-added as a ranking-only tie-breaker**
    (`_find_reid_matches`): a lost track's Kalman-velocity heading
    (`reconcile_heading_rank_weight`, default 0.15) and its keypoint-
    derived body orientation (`reconcile_orientation_rank_weight`, default
    0.12, the same `orientation.py` signal `orientation_relief` uses
    elsewhere) now both nudge the ranking/margin comparison *among
    candidates that already separately qualify on appearance alone* -
    appearance-based qualification itself (`qualify_sim`, what's compared
    against `reid_threshold`) is completely unaffected by either signal,
    so a single uncontested candidate is matched or rejected exactly as
    before regardless of its direction or orientation. Re-verified: the
    single-person "turned around and walked back" clip that broke the
    hard gate still nets 1 identity (unaffected, since there's no
    competing candidate to rank against); the id1/id4 clip is unaffected
    too (the torso fix above already drops the wrong candidate below
    threshold there, so there's nothing left to rank between) - this is
    deliberately defense-in-depth for a case the torso fix alone wouldn't
    catch (two genuinely similar-looking different people, both
    appearance-qualifying, competing for the same reappearance), not yet
    exercised by a real example where it was the deciding factor. Full
    regression suite (all single-person clips, the 2-person clip, both
    3-person clips) re-run and unchanged.
  - **A second, different problem found on the same clip: `id 4`'s own
    later reappearance wasn't reconciling back into her own original
    track at all** - instead creating an unnecessary brand-new id (`id 5`)
    for the same physical person. User-reported ("person with id 4 should
    still have id 4 as she reenters the view field"), and root-caused with
    direct instrumentation of the *real* `_find_reid_matches` call (not a
    manual re-derivation, after an earlier manual recomputation gave a
    misleading "should have qualified" read - see below): her own gallery
    similarity to the reappearance climbed steadily from 0.50 to 0.75
    across her first ~1.5s back in frame, genuinely crossing
    `reid_threshold` (0.65) by frame 410 - but the match still wasn't
    accepted. Found why by wrapping the real match function and printing
    its actual inputs/outputs rather than trusting a hand-recomputed
    threshold comparison: the density-gating mechanism's "how many
    contenders exist" count (`density_threshold_step`, see
    `_find_reid_matches`'s docstring) was counting a completely unrelated
    third lost track (`id 3`) as a competing contender purely because it
    passed the cheap spatial-proximity gate - even though that track had
    an **empty gallery** (zero stored embeddings - every one of its own
    detections had failed the `is_plausible_person_crop` gate while it was
    alive) and so could mathematically never win any match
    (`best_similarity` returns -1.0 unconditionally for an empty gallery).
    That phantom "contender" inflated the effective threshold `id 4`
    needed to clear by `density_threshold_step` per extra contender,
    keeping a genuine, strengthening, same-person match just out of reach
    for the entire window it mattered. **Fix**: an empty-gallery pool track
    is now skipped before the spatial gate entirely (`_find_reid_matches`),
    not just excluded from matching - so it can neither "match" (it never
    could) nor silently tax the threshold for every real contender nearby.
    Re-verified directly on the real decision (not a manual recomputation)
    this time: `id 4` now correctly reclaims her own reappearance at frame
    410, persists as `id 4` through to the end of the clip, and net
    identity count on this clip dropped from 4 to **3** - matching the
    true number of real people in it exactly. Full regression suite
    (all single-person clips, the 2-person clip, both 3-person clips)
    re-run and unchanged.
    **Methodological note, since it mattered twice in one investigation**:
    a manual recomputation of "should this have qualified" against the
    bare `reid_threshold` gave a wrong, misleading answer here because it
    missed the density-adjusted effective threshold - the fix was only
    found by instrumenting the *real* `_find_reid_matches` call directly
    (its actual inputs and actual return value) rather than trusting a
    parallel hand-written reimplementation of its logic.

### `reid_threshold`: the real tradeoff, found empirically

- At **0.75**: a clean, single-person-at-a-time test (with a simulated
  detection dropout) recovered correctly. But spot-checking a crowded
  scene (~10+ simultaneous `LOST_GALLERY` candidates) found real, confirmed
  false-positive merges between
  different people even after the spatial gate + margin + plausibility
  fixes above - the scores for *wrong* matches spread into the 0.5-0.84
  range once there are many candidates, overlapping with correct-match
  scores.
- **Real same-person reappearances can score lower than that clean test
  suggested.** Two real phone-recorded test clips (person walks out of a
  camera's view and back) scored only 0.52-0.71 on genuine reappearances -
  driven by motion blur, partial/edge-clipped crops, and pose/rotation
  changes between the "before" and "after" views, not a bug. A bare
  0.75 threshold missed all of these.
- Lowering to **0.65** (current default) correctly recovered the cleanest
  real test clip (1 person, 1 identity throughout, instead of splitting
  into 2) without breaking either clean synthetic/controlled test case
  re-checked at that threshold.
- **Re-validated at 0.65 against a real crowded scene - confirmed worse,
  not just theoretically risky.** Re-ran the same Shibuya-crossing-style
  footage used for the original 0.75 finding above. Result: 5 of 6
  reconciliation merges were real false merges between different people
  (visually confirmed via the actual crop images - different clothing,
  different people, not an edge case), collapsing into a single "magnet"
  identity whose gallery got progressively more contaminated as it
  absorbed more unrelated people - a compounding failure, not an
  independent one, since a contaminated gallery's `best_similarity` (a
  max over all entries) only gets easier to clear over time.
- **Two mitigations tried, both failed, for instructive reasons** (see
  `density_threshold_step`/`density_margin_step` below for what shipped
  anyway):
  1. *Density gating* (raise the effective threshold/margin when several
     spatially-plausible candidates compete for the same match): checked
     every one of the 6 false merges and found essentially none of them
     involved real competition - there was almost always exactly one
     above-threshold candidate at the moment of the wrong match. The
     failure mode is a single confident-looking-but-wrong match, not
     several candidates confusing the margin check, so gating on
     candidate *count* doesn't touch the actual problem. Kept anyway as a
     free, no-regression partial safety net (it's a no-op whenever there's
     only one plausible candidate, which covers every single/few-person
     case tested).
  2. *Self-similarity baseline* (require a reclaim to beat the young
     track's own gallery similarity to the same detection, not just beat
     other lost tracks): tried and reverted after it broke the main
     correct recovery case from real footage. The reason is structural,
     not a tuning miss - a track's own gallery is built from consecutive,
     highly-correlated frames of itself (measured self-similarity ~0.95-1.0
     once it has more than a couple of entries), while even a *correct*
     cross-session reappearance match (the whole point of reconciliation)
     tops out around 0.85-0.87 in real footage, because real appearance
     naturally drifts between separate sightings (pose, lighting, motion
     blur). Comparing those two numbers head-to-head isn't comparing two
     candidate identities fairly - it's comparing "how similar are two
     frames taken a thirtieth of a second apart" against "how similar are
     two frames taken 10 seconds apart," which is never a fair fight. No
     margin value fixes this; the comparison itself doesn't work.
- **Net effect**: for a single-or-few-person robot-awareness scenario
  (the actual target use case) the 0.65 threshold's recall tradeoff is
  still justified and both mitigations are harmless (no regression on any
  tested clip). But the crowd-scene risk itself remains real and
  unresolved - neither attempted fix closed it, and it should not be
  trusted anywhere a crowd of several similarly-dressed people is
  plausible (see "Not done yet").

## ROS2 integration (`human_tracking_node`)

- **Reuses `PersonDetection`/`PersonDetectionArray` for output instead of a
  parallel `TrackedPerson` type.** `PersonDetection` already had a
  `track_id` field (-1 if untracked) and reused `Keypoint2D[]`; the only
  real gap was a `state` field for the lifecycle status, so that's what
  got added to the existing message rather than duplicating its
  bbox/confidence/keypoints schema in a second type.
- **Subscribes to two topics, synchronized**: the raw camera image
  (`image_topic`) and `human_perception`'s detections (`detections_topic`),
  via `message_filters.ApproximateTimeSynchronizer`. The tracker needs
  actual pixel crops for ReID, which a detections-only message doesn't
  carry - approximate (not exact) sync tolerates the small clock/
  serialization jitter expected between two separately-published topics
  from what is logically the same source frame.
- **`dt_seconds` comes from consecutive image header stamps**, not an
  assumed fixed frame rate (unlike the standalone test script, which
  reads it once from the video file) - the lifecycle timeouts
  (`kalman_timeout_s` etc.) are real-time thresholds and need the actual
  elapsed time between processed frames, which can vary with real
  message-passing overhead.
- **Verified end-to-end** (not just "it builds"): a synthetic Image +
  `PersonDetectionArray` pair, published in-process to a live
  `HumanTrackingNode` for 5 frames, correctly produced a confirmed
  `track_id=1` with `state="ACTIVE"` after exactly `confirm_frames` (3)
  frames, with the published bbox/keypoints round-tripping correctly.
- **GPU path tested, not just present**: `ReIDEmbedder` already accepted a
  `device` argument; actually running it with `device="cuda:0"` (and the
  node's `device` parameter) on real test clips gave identical tracking
  results to CPU (same ids, same reconciliation events) with a 5-7x
  speedup (e.g. 8.3fps -> 43.1fps on a 5-person clip) - confirms the path
  is both correct and worth using once a GPU is available in deployment.

## Not done yet

- **Orientation fusion is only synthetically validated so far** - no real
  footage of a person leaving and reappearing at a different body
  orientation was available to test against (prior test clips were
  deleted, see below). The synthetic checks (direct `Tracker` calls with
  constructed-similarity embeddings) confirm the mechanism does what it's
  designed to - rescue a genuine cross-orientation match that the plain
  threshold would miss, without eroding margin-based disambiguation
  against a same-orientation distractor - but real OSNet embeddings on
  real footage could behave differently (e.g. if front/back similarity
  for the *same* person is more or less separated from different-person
  similarity than assumed here). Should be re-validated the same way the
  reconciliation stage and `reid_threshold` were: real phone-recorded
  exit/re-entry clips, ideally with a deliberate orientation change on
  reappearance, plus a re-check against the crowd-scene footage for new
  false merges.
- **False merges are confirmed real, not just theoretical, and confirmed
  not crowd-exclusive** (see above; crowd findings below). Two crowd-scene
  mitigations were tried and both failed for instructive, structural
  reasons (not tuning misses). Two separate, real occurrences in ordinary
  3-person clips (see "Multi-person validation" above) showed the same
  fundamental risk - raw ReID similarity exceeding `reid_threshold` for
  two different real people during a brief occlusion - happening in "a
  robot seeing one or a few people at a time," the exact scenario earlier
  validation assumed was safe. Two specific contributing causes found so
  far are both fixed: `orientation_relief` wrongly applying during
  reconciliation (first clip), and a per-entry `max(full_sim, torso_sim)`
  letting the weaker torso-only channel override a correctly-disagreeing
  full-crop embedding (second clip, now a weighted partial rescue instead
  of a full override - see "Multi-person validation"). A heading-based
  gate was also tried for the second case and reverted - no heading-only
  signal distinguished "the same person turned around and walked back"
  from "a different person, coincidentally arriving from the opposite
  direction." **Both fixes are validated only against the specific real
  cases that found them, plus the existing regression suite - not a
  guarantee against every possible false merge.** In particular, two
  people with genuinely similar *full-body* appearance (not just similar
  torsos) would not be caught by the torso fix, and the base
  threshold/embedding-discrimination limitation itself is unchanged. This
  still needs either a different approach entirely (e.g. a part-based
  ReID model trained for multi-person discrimination, or refusing to run
  ReID/reconciliation at all above some simultaneous-person count) or an
  explicit, no-longer-assumption-based decision about what identity-
  stability guarantee this project actually needs for its target use case.
- The ROS2 node has not been tested against a live camera topic (same
  `/dev/video0` permissions blocker noted in `docs/context/bbox_keypoints.md`
  applies here too) or against `human_perception`'s node actually running
  live - only verified with synthetic messages published directly, per
  the ROS2 integration section above.
- No automated test suite (`ament_copyright`/`ament_flake8`/`ament_pep257`
  are declared as `test_depend` in `package.xml`, matching `human_perception`'s
  convention, but no actual test files exist yet in either package).
- **No camera-motion compensation (CMC) from odometry.** `docs/context/design_doc.md`
  ("A · Detection + tracking", `cmc_source: odom`) calls this mandatory for a
  moving camera: the Kalman motion model here assumes a static camera, so a
  robot rotation/translation between frames shows up to the tracker as every
  person having suddenly moved, degrading the IoU/Kalman motion match exactly
  when the robot itself is moving - the scenario this project exists for.
  `Tracker` has no odometry input at all; nothing here subscribes to
  `nav_msgs/Odometry` or warps Kalman predictions by the robot's own
  delta-yaw. Every clip validated so far (see "Multi-person validation"
  above) was recorded handheld/stationary-camera-relative-to-the-scene, not
  from a moving robot base, so this gap hasn't yet shown up as a measured
  failure - it's an architectural gap confirmed by inspection, not yet
  exercised against real moving-robot footage.

## Ridgeback idea: bystander negative memory (co-occurrence veto + negative-appearance check)

Ported from the DyNAMO-Ridgeback leader-follower's "non-leader" memory
(`leader_identity_node.py`): while their leader is verified, every other
visible track is remembered as *not the leader*, and is only admissible for
leader recovery later if it looks more like the leader than like its own
remembered bystander features by a margin. They protect one identity; we
protect all of them, so it was reformulated into two separate, optional
constraints in `Tracker` (`tracker.py`):

1. **Co-occurrence veto** (`cooccurrence_veto`, default **on**): every
   track gets an internal `uid` (NEW tracks included) and a `co_visible`
   set. At the end of each `update`, every pair of tracks matched to
   separate detections in that frame records each other as known-different
   people, unless their boxes overlap by more than `cooccurrence_max_overlap`
   (0.5, intersection over the smaller box). That skip covers fragment
   boxes that escaped containment suppression and people mid-crossing.
   Reconciliation (stage 1.5) may never merge a young track into a lost
   identity it was ever co-visible with. Such pairs are dropped before the
   density count, since a provably different track isn't a contender. On a
   merge, the reclaimed identity inherits the young track's `co_visible`
   set. This is a hard constraint the tracker didn't have before. It costs
   no appearance model and no threshold tuning. Stage 2 is untouched,
   because its candidates are raw detections with no history to compare.
   The Ridgeback "self-history" soft version (allow the merge if the
   detection resembles the lost id more than the young track's own past)
   was not implemented. Our young track's gallery already contains the
   current detection, so that test would always block. It would also
   repeat the self-similarity baseline this file already reverted (see
   `reid_threshold` above).
2. **Negative-appearance ("bystander") check** (`bystander_negative_check`,
   default **off**; `bystander_margin` 0.0): a lost identity may only be
   revived on a detection (stage 1.5 or 2) if that detection scores higher
   against the lost identity's gallery than against any *other* track
   visible elsewhere in the same frame (those are certainly not this
   detection). Unlike the reverted self-similarity baseline, this compares
   a cross-person score against cross-person scores.

Both are exposed as node/yaml parameters and test-script flags
(`--cooccurrence-veto`, `--cooccurrence-max-overlap`,
`--bystander-negative-check`, `--bystander-margin`). Vetoes that rejected an
otherwise above-threshold pair are logged in `Tracker.veto_events` and
printed by the test script. The script also gained `--dump-tracks PATH.jsonl`
(a per-frame id/state/bbox dump, plus the merge map and veto events) for
diffing runs frame by frame.

**Measured** (8-run protocol; baseline = unchanged code; "veto" = 1 only;
"veto+neg" = 1 and 2). Each cell is confirmed ids / net identities /
motion+ReID recoveries:

| run | baseline | veto | veto+neg |
|---|---|---|---|
| A | 1 / 1 / 0+1 | identical | identical |
| B | 2 / 1 / 0+1 | identical | identical |
| C | 1 / 1 / 1+1 | identical | identical |
| D | 7 / 4 / 8+5 | identical | 7 / 5 / 9+3 |
| D `--blackout 150:45` | 7 / 4 / 11+5 | identical | 7 / 5 / 12+3 |
| D `--blackout 300:90` | 7 / 4 / 8+6 | identical | 7 / 5 / 9+4 |
| D `--blackout 450:150` | 5 / 4 / 7+4 | identical | identical |
| C `--blackout 200:120` | 1 / 1 / 1+2 | identical | identical |

- **Co-occurrence veto: never fired.** The per-frame dumps are
  byte-identical to baseline on all 8 runs. On this footage, no
  reconciliation candidate was ever co-visible with the identity it tried
  to rejoin. It is kept on by default because it is neutral here and can
  only remove merges between tracks that were provably two people at
  once, the 3-person false-merge shape documented above. It is also
  covered by synthetic tests.
  - Known blind spot: if a young track's motion match hops onto another
    person inside its 1.5 s reconcile window, the veto would block the
    corrective merge. This hasn't been seen on real footage.
  - Not yet validated on footage where it actually fires.
- **Negative check: a confirmed regression, so it stays off.** It fires on
  one real decision, D f565-570 (also in the f150 and f300 blackout
  variants). The man in the white tee (id 2) is lost at f552 and reappears
  as young id 7. Baseline correctly reclaims 7 into 2 at f565
  (similarity 0.66-0.77). The check vetoes it because his crop scores
  0.75-0.82 against another visible person's gallery: the woman in a white
  top, a similar torso.
  - Knock-on effect: the man stays fragmented as id 7. Then at f571 the
    still-lost id 2 is falsely reclaimed onto the navy-hoodie woman
    (baseline id 4) during her brief loss. She keeps the wrong id to the
    end of the clip.
  - The check turned one correct recovery into a fragmentation plus an
    identity swap. It was checked against annotated stills at f540, f562,
    f567 and f590.
  - Why it fails: with similarly dressed people, "resembles a visible
    bystander" is common for genuine reappearances too. A fresh gallery of
    someone in view right now has a recency advantage over a lost
    identity's older samples.
- Tests: `src/human_tracking/test/test_bystander_memory.py` (8 synthetic
  cases), covering:
  - the baseline false merge of a lookalike co-visible bystander, and the
    veto blocking it;
  - a genuine reappearance still being reconciled;
  - overlap-guarded recording, symmetric recording, and inheritance on
    merge;
  - the stage-2 negative check blocking and not blocking.
- Caveat: one multi-person clip with about 3 people (P1/P2 nearly
  throughout, P3 coming and going). "Neutral" for the veto means "never
  exercised", not "proven helpful".
