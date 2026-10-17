# Head pose relative to the robot

## Design reference

`~/Downloads/Pipeline awareness robot2.pdf` ("Pipeline de percepție pentru
cooperarea om-robot") is the authoritative design doc for the *whole*
awareness pipeline this project is building toward (Romanian) - gaze/iris
(MediaPipe + L2CS-Net), environment context (SigLIP2 + a label-free CBM),
temporal fusion into a cooperation score, and all the geometry/parameters in
between. Only a slice of it is built so far: `human_perception` +
`human_tracking` (the doc's "A · Detecție + tracking") and this package
(the doc's "B1 · Orientarea capului"). The rest (gaze/iris, environment,
fusion) is not started - don't assume it exists because the doc describes
it. Where this file says something matches or diverges from "the design
doc," that PDF is what's meant.

## What it is

For each tracked person, estimates their head orientation (pitch/yaw/roll)
via 6DRepNet360, combines it with that person's position in frame to get
their bearing from the camera, and derives a single "gaze_robot_deg" angle:
how far the person's head is turned away from directly facing the robot.
This is the first piece of the actual *awareness* signal the whole project
is for - everything before this (`human_perception`, `human_tracking`)
answers "where are the people," this answers "are they looking at us."

## Why 6DRepNet360, not the original 6DRepNet

The original 6DRepNet is trained/evaluated on a narrower yaw range (roughly
±99°). A person who has turned most of the way around - showing mostly the
back of their head - is exactly the case `human_tracking`'s ReID already
struggles with (see `docs/context/human_tracking.md`'s orientation-fusion
entry) and exactly the case this project most needs head pose *for*
(detecting "not aware, facing away" is as important as detecting "aware,
facing the robot"). 6DRepNet360 is the same authors' full-range (-180°..180°)
retrain of the same architecture on the same representation, specifically to
cover that case - so it was picked over the narrower original for the same
reason orientation-fusion didn't reach for a face-only model: the back-of-
head case has to actually be representable, not just the near-frontal range.

## Inputs / outputs

- **Input**: a raw camera image + a `PersonDetectionArray` (bbox + COCO-17
  keypoints + track_id/state), time-synchronized the same way
  `human_tracking_node` syncs image + detections. Defaults to subscribing to
  `human_tracking`'s tracked output (`human_tracking/tracked_detections`),
  not `human_perception`'s raw per-frame detections - head pose readings are
  meant to accumulate against a persistent identity, not a raw per-frame
  blob, matching `human_tracking`'s own framing of awareness as a property
  of a person over time.
- **Output**: the same `PersonDetectionArray`, same message type again
  (reused for the third time now - see `human_tracking`'s and
  `human_perception`'s own notes on this pattern), with
  `head_pose_valid`/`head_yaw_deg`/`head_pitch_deg`/`head_roll_deg`/
  `bearing_deg`/`gaze_robot_deg` filled in per detection (all zero/false
  when a head crop couldn't be extracted that frame).

## Design decisions

- **Head crop from keypoints, not a face detector.** 6DRepNet360's own demo
  pipeline (per the upstream repo) expects a face detector's crop. Running a
  second detector just to re-find a face inside a region YOLO26-pose already
  localized would be redundant work and a second place for detection to
  fail. Instead, `head_crop.py` derives the head region directly from the
  keypoints `human_perception` already produces: a "neck" point (COCO-17 has
  no neck keypoint, so it's approximated as the midpoint of the two shoulder
  keypoints - the standard stand-in when only body keypoints are available).
- **Crop sized from body scale, not the bbox - and square, not bbox-width.**
  Matches the design doc's "B1 Orientarea capului" / "B Cap + Gating" slides
  exactly: a square crop anchored at the neck, extending upward by
  `head_k_h * scale` (+ `crop_margin` padding on each side), where `scale =
  max(shoulder_width, head_scale_alpha * neck_to_nose_distance)`. An earlier
  version used the full bbox width down to the neck y-coordinate instead -
  replaced because the bbox itself is an unreliable size reference (it
  changes with raised arms, a leg out of frame, or a lean), while the
  neck-to-shoulder geometry stays stable. The `neck_to_nose` term specifically
  is what keeps the scale from collapsing in profile, where shoulder width
  visually shrinks but the neck-to-nose distance doesn't (falls back to
  shoulder width alone when the nose isn't confident). Below `min_head_px`
  pixels, no crop at all (`head_pose_valid=false`) - too small to trust,
  rather than feeding the model a crop it was never evaluated on. One
  simplification vs. the doc: the doc crops "along the hip-to-neck axis" (so
  a leaning body gets a correspondingly tilted crop); this always crops
  along the vertical image axis instead - see "Not done yet".
  Requires both shoulder keypoints confident to even find the neck (reusing
  the `min_keypoint_conf=0.3` convention `human_tracking`'s own
  keypoint-gated helpers use) - no shoulders confident, no crop.
- **Adapters, named and separated per the design doc's "Adaptoare"
  principle** (`adapters.py`): "a component's output is rarely exactly the
  next one's input... an adapter is a pure function, tested on its own, so
  changing a model only ever touches its adapters." Two arrows get one
  here: `person_detection_to_bbox_keypoints` (a ROS `PersonDetection`
  message's fields -> this package's plain `(bbox_xyxy, keypoints)` tuples -
  used by `head_pose_node`) and `extract_head_crop` (keypoints + the source
  frame -> an actual pixel crop, combining `head_crop_bbox`'s geometry with
  the frame-slicing every caller previously repeated inline). The same
  `extract_head_crop` adapter is what lets `scripts/test_tracking_and_head_
  pose_on_video.py` feed a `human_tracking.Track`'s own `last_keypoints`
  straight in, with no ROS message in between - the Track already exposes
  plain tuples in the same shape the adapter expects, so no separate
  Track-specific adapter was needed, just reuse of this one.
- **Bearing-corrected gaze angle, not raw yaw.** A head-pose network trained
  on cropped-and-centered face patches has no way to know where in the frame
  that crop came from - appearance alone can't distinguish "centered and
  facing the camera" from "off to the side and turned to still look at the
  camera" (both produce a visually frontal crop). So raw yaw alone measures
  orientation *relative to the camera-to-face viewing ray*, not relative to
  the camera's fixed optical axis - correct for a centered subject, wrong by
  exactly that person's bearing angle otherwise. `geometry.py` computes that
  bearing (pure pinhole-model azimuth from the bbox's x-position and a
  configurable `camera_hfov_deg`- doesn't need depth, since azimuth from a
  horizontal pixel offset is depth-independent) and subtracts it:
  `gaze_robot_deg = head_yaw_deg - bearing_deg`. This is standard practice in
  gaze-target-estimation work that needs to detect "looking at the camera"
  for off-center subjects, not a correction invented for this project.
- **`camera_hfov_deg` is a bare parameter, not read from `CameraInfo`.** The
  bearing calculation needs *some* notion of the camera's field of view;
  reading real intrinsics from a `CameraInfo` topic would be more correct
  but is more plumbing than a first version needs. Defaulted to 70° (a
  common webcam/robot-camera ballpark) and documented as needing real
  calibration per actual deployment camera - see "Not done yet".
- **Vendored model, not a pip dependency** (`sixdrepnet360.py`, adapted from
  `github.com/thohemp/6DRepNet360`, MIT-licensed): the official repo only
  ships training/eval scripts, no demo/inference entry point ("Usage: Coming
  soon!" as of this writing) and no pip package for the 360 variant - so the
  architecture (ResNet-50-Bottleneck backbone + 6D rotation head) and the
  rotation-matrix-to-Euler conversion are reproduced directly from
  `sixdrepnet360/test.py`/`utils.py`, trimmed to inference only (no
  training/dataset/plotting code), matching the same vendoring rationale
  `human_tracking/osnet.py` already documents for OSNet.
- **Weights: the "Full-Rotation" checkpoint** (fine-tuned on 300W-LP +
  Panoptic), the widest-range one the repo publishes - matches the whole
  reason 360 was picked over the original model. A plain HTTPS download
  (`cloud.ovgu.de`), cached at `~/.cache/human_head_pose/` on first use - no
  `gdown`/Google Drive step needed here, unlike OSNet's weights.
- **New package (`human_head_pose`), new branch, reusing `PersonDetection`
  again** rather than folding this into `human_tracking`: head pose is a
  genuinely separate concern (different model, different crop, no lifecycle/
  gallery state of its own) that only *consumes* a track's existence, so it
  gets its own package the same way `human_tracking` got its own package
  rather than living inside `human_perception`.

## Empirical validation so far

No controlled test photo (known head-turn direction, known camera position)
existed yet, so validation used real frames from the `human_tracking`
whatsapp2 test clip instead:

- **A clearly frontal frame** (both shoulders confident, person looking
  straight at the camera): `yaw=1.2°, pitch=9.3°, roll=-2.6°` - correctly
  near zero, matching the dataset convention that yaw≈0 means facing the
  camera.
- **A frame mid-turn** (person turning their head up and to one side):
  `yaw=40.9°, pitch=25.2°` - a real partial turn, correctly sized as
  "partial," not pinned to 0 or ±90.
- **A frame in full profile** (confirmed by eye - the crop visually shows a
  clean side-on view, nose pointing left, opposite ear hidden):
  `yaw=75.8°, pitch=106.2°, roll=87.3°` - yaw correctly reads as "close to a
  full 90° turn," with pitch/roll both blowing up alongside it.
- **That blow-up was wrongly written off as "expected degeneracy, not a
  bug" here - it was actually a real bug, confirmed on a second, harder
  clip.** A multi-person clip of 5 people walking *away* from the camera
  (backs fully turned, unambiguous ground truth) initially showed most of
  them labeled "LOOKING AT ROBOT." Checked the raw model output directly:
  three of the five had decomposed `yaw` near 0° (reading as "facing the
  camera") while `pitch`/`roll` had swung to near ±180° instead - the
  rotation hadn't vanished, it had aliased onto the other two angles. This
  is a real, known degeneracy of recovering 3 sequential Euler angles from
  a rotation matrix (several different (pitch, yaw, roll) triples can
  describe the identical matrix), and it gets *worse*, not better, the
  further a head turns from frontal - exactly backwards from "only matters
  at the extremes, trust the common case." The near-90° profile case above
  was the same bug in a milder form, misdiagnosed as instability instead of
  wrongness.
- **Fixed in `head_pose_estimator.py`**: pitch and yaw are no longer read
  from `sixdrepnet360`'s 3-angle Euler decomposition at all. Instead,
  `_face_direction_pitch_yaw` converts the face-normal vector
  (`R @ [0,0,1]`) straight to azimuth/elevation - recovering only the 2
  degrees of freedom that actually determine which way the face points
  (dropping roll/twist, which doesn't change that), so there's no third
  angle left for the rotation to alias onto. Roll is still read from the
  Euler decomposition (unaffected - it's yaw/pitch that aliased into each
  other and into roll, not roll in isolation) and isn't used by
  `gaze_robot_deg` anyway, only shown for display.
  Re-verified on the same 5-person back-facing clip: all five now read
  large-magnitude yaw (90°-179°, all correctly "not facing the camera"),
  and the aggregate "LOOKING AT ROBOT" rate on that clip dropped from 67%
  to 0.6%. Re-checked the frontal/profile ground-truth frames above
  (yaw≈1° frontal, yaw≈76° profile) to confirm the fix doesn't regress
  them - both still read correctly (now `gaze=-13°`/`107°` with the
  bearing correction applied, same conclusion either way). The "looking"
  counts on all three previously-tested clips (whatsapp2/3, the 4-person
  meeting) also dropped once this shipped - read as correcting more
  previously-mis-aliased borderline/profile frames, not a regression,
  since every specific ground-truth frame checked by eye still matches.
- **Sign convention confirmed, not assumed**: in the full-profile frame, the
  person's nose visibly points toward image-left (confirmed by looking at
  the actual crop) and 6DRepNet360 reports `yaw=+75.8°` (positive) - so
  positive yaw here means the face has turned toward image-left. This
  matches the common Hopenet/300W-LP-family convention, now confirmed
  against this project's own crop pipeline rather than assumed from the
  paper.
- **Not yet confirmed**: the bearing-correction term's sign specifically -
  that needs a frame where someone off-center is *actually looking at the
  camera* (the one case raw yaw and bearing-corrected yaw disagree on), which
  none of the available footage happens to contain. Until that's checked,
  trust `gaze_robot_deg`'s magnitude more than cases near its sign boundary.
- **Full-video run, with the on-screen "looking at the robot" label**
  (`scripts/test_head_pose_on_video.py`, `--aware-threshold-deg` default 45°):
  run on both real whatsapp test clips (the same ones `human_tracking` used -
  see `docs/context/human_tracking.md`). whatsapp2: 442 frames, 295 with a
  valid head crop, 193 labeled "LOOKING AT ROBOT". whatsapp3: 637 frames, 511
  valid, 403 looking. Spot-checked two frames by eye: a clearly frontal frame
  correctly labeled "LOOKING AT ROBOT" (gaze -6°), and the same full-profile
  frame from the single-photo check above correctly labeled "NOT LOOKING"
  (gaze 88°) once run through the full per-frame pipeline instead of a single
  extracted photo. The `--aware-threshold-deg` cutoff itself is a provisional
  demo value for this script, not a validated awareness threshold - see "Not
  done yet".
- **Combined with `human_tracking`** (`scripts/test_tracking_and_head_pose_
  on_video.py`, see "Design decisions" for the adapter this demonstrates):
  re-ran both clips through detection -> `Tracker` -> head pose in one pass,
  instead of each part running its own independent detection. Results
  essentially unchanged from the head-pose-only run (whatsapp2: 293 valid/
  199 looking vs. 295/193 before; whatsapp3: 509/402 vs. 511/403) - the
  small difference is the new neck-anchored square crop (see "Design
  decisions") being slightly more/less willing to extract a crop on a given
  frame than the old bbox-width one, not a tracking-integration effect.
  Spot-checked the same two frames again with both the track id and the
  gaze label drawn together: frontal frame, `id 1` / "LOOKING AT ROBOT" /
  gaze -12°; full-profile frame, `id 1` (same identity, confirming ReID held
  through the turn) / "NOT LOOKING" / gaze 91°. Also visibly confirmed the
  new crop geometry is doing its job - the orange crop box stays tightly on
  the head in the profile frame instead of including a lot of shoulder/
  background the way the old bbox-width crop did.

## Parameters (`head_pose_node`)

| param | default | meaning |
|---|---|---|
| `image_topic` | `/camera/image_raw` | raw camera feed |
| `detections_topic` | `human_tracking/tracked_detections` | synced detections source |
| `head_poses_topic` | `human_head_pose/head_poses` | output topic |
| `sync_slop_s` | 0.05 | image/detections sync tolerance |
| `weights_path` | `""` (use cached/downloaded default) | 6DRepNet360 weights override |
| `device` | `cpu` | `cuda:0` etc. for the pose estimator |
| `min_keypoint_conf` | 0.3 | shoulder confidence required to trust the neck estimate |
| `camera_hfov_deg` | 70.0 | assumed horizontal field of view, for the bearing calculation |
| `head_k_h` | 0.8 | head crop height, as a multiple of `scale` |
| `head_scale_alpha` | 1.7 | weight on the neck-to-nose term in `scale = max(shoulder_width, alpha * neck_to_nose)` |
| `crop_margin` | 0.2 | fractional padding added to the crop on each side |
| `min_head_px` | 35.0 | below this crop size (pixels), skip the pose estimate entirely |

## Not done yet

- **No real camera intrinsics** - `camera_hfov_deg` is a guessed default,
  not calibrated per deployment camera. Wrong FOV directly biases
  `bearing_deg` and therefore `gaze_robot_deg` for anyone off-center. The
  design doc also undistorts via a real `CameraInfo` (K, D) before computing
  the radius direction - not done here at all, FOV-only.
- **Bearing-correction sign unconfirmed** (see "Empirical validation" above)
  - needs a deliberate test photo, not just real footage of opportunity.
- **`gaze_robot_deg` is a simplified 2D approximation, not the design doc's
  full 3D treatment.** The doc computes an actual 3D face-direction vector
  (`R · [0,0,1]`, in the camera's optical frame) and takes
  `arccos(face_direction · -radius_direction)` against the real 3D
  direction to the robot - which naturally folds in pitch (a person tilting
  their head up/down while still looking at the robot is still "looking,"
  which the formula below can't currently represent) and doesn't need the
  dataset's own axis convention to line up with anything by assumption.
  `head_pose_estimator.py` now also derives yaw from that same `R · [0,0,1]`
  vector (see "Empirical validation" above - this started as a correctness
  fix for a real aliasing bug, not an attempt at the doc's full treatment),
  but only its azimuth component, flattened back to 2D before combining
  with `bearing_deg` - so it's closer to the doc's approach than before,
  still short of it. Skips the "rotație fixă de convenție" (dataset axes ->
  camera optical frame) step the doc calls for, which wasn't implemented
  here since guessing that fixed rotation's exact convention wrong would be
  exactly the kind of plausible-but-silently-wrong bug this project has
  otherwise been careful to avoid. Revisit if pitch-while-looking-away
  turns out to matter, or once there's a way to validate the fixed rotation
  against real data.
- **No distance/3D-position estimate, so no cone-based "vede robotul"
  threshold.** The doc's actual decision rule is geometric: a visual cone
  `atan(robot_radius_m / distance)` (`robot_radius_m=0.3`) plus
  `peripheral_buffer_deg=5` and `theta_contact_deg=12` for direct contact,
  widened by `head_only_extra_deg=15` when only head pose (no gaze) is
  available - not a flat angle cutoff. This project has no distance
  estimate yet (the doc's own fallback chain is ground-plane height ->
  `shoulder_width_prior_m=0.40` -> `person_height_prior_m=1.70`, none of
  it built), so `scripts/test_tracking_and_head_pose_on_video.py`'s
  `--aware-threshold-deg` (default 45°) is a flat-angle stand-in for demo
  purposes only, explicitly not the calibrated decision rule.
- **Crop isn't oriented to a leaning body** (see "Design decisions") -
  always a vertical-axis crop, not one tilted along the hip-to-neck axis.
- **Not tested against a live camera topic or alongside `human_tracking_node`
  actually running live** - same gap `human_tracking`'s own "Not done yet"
  notes for itself, inherited here since this node depends on that one's
  output.
- **No automated test suite**, consistent with the other packages so far.
- **No notion yet of what "aware" actually means as a boolean/threshold** -
  this node produces the angle; turning `gaze_robot_deg` (plus, presumably,
  some time-window logic reusing `human_tracking`'s per-identity state) into
  an actual awareness decision is future work, not started here.
- **Gaze/iris (L2CS-Net + MediaPipe) and environment context (SigLIP2 +
  CBM)** from the design doc are entirely separate, not-yet-started logical
  parts - not a gap in this package specifically, just scope not reached yet.
