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
- **Crowd-scene false merges are confirmed real, not just theoretical, and
  still unresolved** (see above) - two mitigations were tried and both
  failed for instructive, structural reasons (not tuning misses), so this
  needs either a different approach entirely (e.g. a part-based ReID model
  trained for multi-person discrimination, or simply refusing to run ReID/
  reconciliation at all above some simultaneous-person count) or an
  explicit decision to only deploy where a crowd is implausible (the
  actual target use case: a robot seeing one or a few people at a time).
- The ROS2 node has not been tested against a live camera topic (same
  `/dev/video0` permissions blocker noted in `docs/context/bbox_keypoints.md`
  applies here too) or against `human_perception`'s node actually running
  live - only verified with synthetic messages published directly, per
  the ROS2 integration section above.
- No automated test suite (`ament_copyright`/`ament_flake8`/`ament_pep257`
  are declared as `test_depend` in `package.xml`, matching `human_perception`'s
  convention, but no actual test files exist yet in either package).
