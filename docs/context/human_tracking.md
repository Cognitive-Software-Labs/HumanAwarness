# Human tracking / ReID (persistent identity across occlusion)

## What it is

Takes raw per-frame person detections (from `human_perception`) and turns
them into *persistent identities* that survive brief occlusion or the
person temporarily leaving the camera's view — so that "awareness" can be
tracked as a property of a person over time, not reset every time
detection drops out for a frame or two.

## Status

**Core tracking/ReID logic only, validated via a standalone test script
(`scripts/test_human_tracking_on_video.py`). Not yet wired into a ROS2
node or packaged** (`src/human_tracking` currently has no `package.xml`/
`setup.py` — it's a plain importable Python module, not yet buildable
with `colcon`). That packaging is the clear next step, not done yet.

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

### `reid_threshold`: the real tradeoff, found empirically

- At **0.75**: a clean, single-person-at-a-time test (`man_followed_city.mp4`,
  with a simulated detection dropout) recovered correctly. But spot-checking
  a crowded scene (`japan_street.mp4`, ~10+ simultaneous `LOST_GALLERY`
  candidates) found real, confirmed false-positive merges between
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
- **This default has not been re-validated against the crowded-scene
  false-positive case at 0.65** (only checked at 0.75, where it already
  showed real false merges). A lower threshold gives weaker evidence more
  benefit of the doubt, which plausibly makes a crowded scene *worse*,
  not better. For a single-or-few-person robot-awareness scenario
  (the actual target use case) this tradeoff favors recall; it may need
  to be raised again, or made context-dependent, before trusting it
  anywhere a crowd is plausible.

## Not done yet

- No ROS2 package/node - core logic only, see Status above.
- No message type designed yet for publishing tracked identities (would
  need a `TrackedPerson`/`TrackedPersonArray`-style addition to
  `human_awareness_msgs`, analogous to `PersonDetection`/`PersonDetectionArray`).
- `reid_threshold=0.65` not re-validated against the crowded-scene
  false-positive case (see above) - likely needs revisiting once there's
  a concrete multi-person test scenario that matters for the real
  deployment (not just the extreme crowd stress-test).
- OSNet runs on CPU by default in `ReIDEmbedder`; no GPU path tested.
