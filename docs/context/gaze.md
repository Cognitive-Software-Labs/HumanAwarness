# Iris / gaze relative to the robot

## Design reference

Same source as `docs/context/head_pose.md`: `~/Downloads/Pipeline awareness
robot2.pdf`, this time its "B2 · Iris si privire" slide ("MediaPipe
pregateste, L2CS-Net estimeaza") plus the matching "Adaptoare · B2" data-
contract slide, and - for phase 2 below - the "PARAMETRI · B PRIVIRE IN TIMP
+ UNGHIURI" and "PARAMETRI · GEOMETRIE + TRAIECTORIE" parameter slides.

Built in two explicitly separated phases, per the project owner's own
scoping call ("first build 1 and then 2"):
- **Phase 1**: the iris/gaze vector itself (gating, estimation, fixation,
  fusion with head pose), using a flat-angle demo threshold for "sees the
  robot" - `aware-threshold-deg` in the test scripts.
- **Phase 2**: the doc's real geometric decision rule - a distance estimate
  (the doc's own separate "3 · Geometrie" stage) feeding a robot-sized cone
  (`atan(robot_radius_m / distance)`), replacing the flat threshold. Built
  after phase 1 surfaced a concrete, visually-confirmed problem with it (see
  "Empirical validation: phase 1 vs. phase 2").

## What it is

For each tracked person whose head pose (B1) is already valid, estimates
where their eyes are actually pointed - not just which way their head is
turned - and uses that to *replace* the head-only "facing the robot" angle
whenever a trustworthy iris reading is available this frame, falling back
to head-only otherwise. This is the project's first temporally-filtered
signal: unlike head pose (recomputed fresh every frame, no memory), gaze
needs a held fixation before it's trusted at all, and is smoothed across
frames once it is.

## Why MediaPipe for gating, MobileGaze for the angle (not one model doing both)

MediaPipe Face Landmarker gives 478 3D landmarks including 5-point iris
detail per eye, is cheap, and is directly interpretable (eye corners, iris
center, blink state) - but it doesn't estimate gaze direction itself; its
own geometric approximation for that has a published 5-10 degree error,
per the design doc's own model-comparison slide. MobileGaze (L2CS-Net-style
bin-classification gaze estimation, trained on Gaze360) is accurate and
covers large angles including near-profile, but has no opinion on whether
a given frame is even trustworthy to run on (open eyes? face actually
visible? not motion-blurred?). Splitting the two lets each model do only
what it's actually good at - MediaPipe decides *whether* to trust this
frame, MobileGaze decides *what angle* if so.

## Design decisions

- **MobileGaze variant: `mobileone_s0`, not a ResNet backbone.** The design
  doc names "L2CS-Net / MobileGaze" together as one chosen line item;
  confirmed via research (not assumed) that MobileGaze
  (github.com/yakhyo/gaze-estimation) is a from-scratch reimplementation of
  L2CS-Net's exact method (same bin-classification head, same Gaze360
  training, same 90-bin/4-degree-width/-180..180-degree decode formula)
  offering several backbone sizes - `mobileone_s0` is the smallest
  published-weights variant (1.24M params, ~4.8MB checkpoint vs. the
  ResNet-50-equivalent's ~91MB), chosen for the same resource-appropriate-
  variant reasoning already used for `osnet_x1_0` and `yolo26n-pose` in
  this project, at a modest accuracy cost (12.58 vs. 11.34 degrees MAE on
  the model's own Gaze360 test set, per its published results).
- **Vendored the architecture, not pip-installed** (`mobileone_gaze.py`,
  adapted from the upstream repo's `models/mobileone.py`, itself adapted
  from Apple's MobileOne - MIT-licensed, matching OSNet/6DRepNet360's own
  vendoring rationale in this project). Trimmed further than those two:
  the published checkpoint is already saved in *fused* (inference-mode)
  form - confirmed empirically, loads with zero missing/unexpected keys
  directly into a model built without any training-time branch structure
  at all - so the upstream repo's `reparameterize()`/`_fuse_bn_tensor()`
  machinery and the s1-s4 size variants are genuinely dead code for this
  project's single always-inference-mode use case, not just unused-for-now.
- **Weights license is the same tradeoff the design doc already accepted
  for this model choice, now independently confirmed rather than assumed.**
  Code is MIT. The published weights are trained exclusively on Gaze360,
  whose own license (read directly, not inferred) restricts "models
  trained on the dataset" to research/non-commercial use - the same
  "licenta necomerciala" caveat the design doc's own model-comparison
  slide already notes for this exact line item. Not a new decision made
  here; flagging it again at the point it actually gets vendored, same as
  how OSNet/6DRepNet360's own licensing is documented at their vendoring
  point.
- **Preprocessing: aspect-preserving resize to the shorter side = 448, not
  a square resize.** The upstream repo has two inference paths that
  disagree with each other - confirmed by reading both directly, not
  assumed from the README: the PyTorch path (what training actually used)
  does `transforms.Resize(448)`, which scales the shorter side and crops
  nothing, so the tensor is often non-square; the separate ONNX-export
  wrapper instead does a hard `cv2.resize(img, (448, 448))`, which distorts
  aspect ratio and does *not* match what the model was trained on - a
  convenience shortcut baked into that one wrapper class, not a property
  of the model itself. `gaze_estimator.py` replicates the PyTorch path's
  behavior (both backbones tolerate variable spatial size via
  `AdaptiveAvgPool2d`). Known, accepted approximation: `cv2.resize` isn't
  bit-identical to PyTorch/PIL's own resize implementation - measured an
  ~8-degree yaw shift between the two on the same crop in informal testing
  - not chased further since what matters is internal consistency with
  real-footage validation, not exact parity with the original research
  code's exact pixels.
- **MediaPipe Face Landmarker: `--no-deps` install, pin `protobuf>=4.25.3,
  <5`.** Both confirmed empirically after real failures, not assumed from
  docs - see `src/human_gaze/requirements.txt` for the exact fix and why a
  plain `pip install mediapipe` breaks `cv_bridge` the same way
  ultralytics/numpy did in `human_perception` (docs/context/bbox_keypoints.md).
  The protobuf one was specifically confusing to track down: it doesn't
  fail at import time, only the first time a face is actually detected
  (`AttributeError: module 'google.protobuf.message_factory' has no
  attribute 'GetMessageClass'`), so a quick smoke test with no face in
  frame can look like success right up until real footage hits it.
- **Known mediapipe library bug, confirmed not a code bug of this
  project's:** `FaceLandmarkerOptions(num_faces=1)` returns zero faces on
  frames where `num_faces=2` correctly finds the exact same single face -
  reproduced on the same image, same confidence threshold, both mediapipe
  0.10.21 and 1.1.1 (tried pinning to the older, more established release
  specifically to rule out a regression in the brand-new 1.x line - same
  bug in both). Worked around in `mediapipe_gate.py` by always requesting
  `num_faces=2` and using the first result; a per-person head-crop-cropped
  frame only ever has one real face in it anyway, so the extra capacity
  costs nothing.
- **Face crop for MobileGaze comes from the *original* frame, not from the
  head-crop sub-image** - matches the design doc's adapter spec exactly
  ("crop fata din cadrul original"): MediaPipe's landmarks are already
  normalized relative to the head crop, so the face bbox they imply is
  converted back to frame-absolute coordinates
  (`adapters.crop_relative_to_frame_xyxy`) before slicing, avoiding
  compounding the head crop's own (small) resolution loss on top of an
  already-small face region.
- **Fixation detection runs on the *raw* gaze angle; the One Euro filter
  runs on the fixation's confirmed mean, not the raw per-frame angle.**
  Deliberate ordering, not arbitrary: feeding already-smoothed values into
  fixation detection would blunt genuine saccades into looking like slow
  drifts, undermining the `fixation_max_dev_deg` check's whole purpose
  (telling a real held gaze apart from one mid-movement). The One Euro
  filter's job is display/fusion stability frame-to-frame *once* a
  fixation is already confirmed trustworthy, not deciding trustworthiness
  itself.
- **A gap longer than a blink ends a fixation even when nothing was fed
  during it, and each new fixation starts with fresh One Euro filters.**
  `gaze_node` and the video script don't call the pipeline at all for a
  LOST track or a frame without a valid head pose, so `FixationTracker`
  sees such a gap as missing updates, not `None` readings. It applies
  `blink_gap_max_ms` to both: a reading more than that after the last one
  starts a new fixation, which must then hold for `fixation_min_ms` on
  its own. Before, the old fixation absorbed the reading and counted as
  held across the gap, so someone coming back from a 3 s occlusion looking
  roughly where they had been was iris-fixating on their first frame
  back. The yaw/pitch smoothers restart with each new fixation, after a
  saccade as well as after a gap. Carried over, they pulled the new
  fixation toward the previous one's angle (17.4° on the first frame after
  a 0° -> 20° saccade), since `dt_seconds` is the frame interval, not the
  time since they last ran.
- **Iris never partially overrides head pose - it's all-or-nothing per
  frame** (`fusion.fuse`): an iris reading that's gated, estimated, and
  fixating but implausible against head pose (`max_eye_head_deg`) falls
  back to head-only entirely, not some blended value. Matches the design
  doc's own framing ("privire - cap peste asta -> privirea se arunca" -
  thrown away, not down-weighted) and avoids inventing an untested blending
  rule where the doc specifies a hard switch.
- **`gaze_robot_deg` itself is kept as a 2D, yaw-only correction**
  (`gaze_yaw_deg - bearing_deg`, mirroring `human_head_pose.geometry`'s own
  simplification exactly, not re-litigated here) - the real 3D treatment
  lives entirely in the phase-2 geometric decision below (`sees_robot`/
  `direct_contact`), not retrofitted into this field, so existing consumers
  of `gaze_robot_deg` see no behavior change from phase 2's addition.
- **LOST tracks get no gaze, whatever head pose they arrive with.** Like
  `head_pose_node`, `gaze_node` only processes ACTIVE tracks and untracked
  detections (`state == ""`). A `LOST_KALMAN`/`LOST_GALLERY` track carries
  its last match's keypoints, so its head crop, face crop and phase-2
  reference pixel would all come from wherever the person used to be. It
  is republished with `head_pose_valid=false` and its head-pose fields
  zeroed (the "all zero" `PersonDetection.msg` promises for an invalid
  head pose), every gaze field at its default (`sees_robot=false`,
  `distance_method=""`), and everything else unchanged. The node checks
  `state` itself (`human_head_pose.adapters.has_fresh_keypoints`) instead
  of relying on `head_pose_valid`, so input from an older
  `head_pose_node`, or a bag recorded before it skipped LOST tracks, can't
  leak through. A LOST track's fixation/smoothing state is still kept
  (`prune` gets every published `track_id`), but the gap itself ends its
  fixation (see the fixation-gap point above). The video script already
  skipped non-ACTIVE tracks this way.

### Phase 2: the geometric decision rule

Three new modules, each a pure, independently-testable function per the
design doc's own "Adaptoare" principle - no vendored model involved, this
is all project-owned geometry:

- **`camera_model.py`**: a minimal pinhole model. `intrinsics_from_hfov`
  derives focal length from `camera_hfov_deg` + image size - the same
  approximation `bearing_deg` already makes, extended to give a focal
  length in pixels, not just an angle-per-pixel-column. `pixel_ray_camera_
  frame` back-projects a pixel into a 3D unit ray, in the *same*
  camera-optical-frame convention (`+x` right, `+y` down, `+z` forward)
  `gaze_estimator.gaze_yaw_pitch_to_vector` already uses - chosen
  deliberately so the two are directly comparable with a plain dot product,
  no extra frame conversion needed between "which way the person is
  looking" and "where the robot actually is from them."
- **`distance_estimator.py`**: the design doc's own three-tier fallback
  (its "PARAMETRI · GEOMETRIE + TRAIECTORIE" slide), cascading by what's
  actually visible this frame: ground-plane projection from an ankle
  keypoint (preferred - it's measuring this specific person, not assuming
  average proportions), then a shoulder-width size prior
  (`shoulder_width_prior_m=0.40`) if feet aren't confident, then a
  person-height size prior (`person_height_prior_m=1.70`, using the
  detection bbox height) as the last resort. Returns which tier fired
  alongside the number (`distance_method`), not just the distance, so a
  bad estimate can be traced to its source rather than trusted blindly.
- **`geometric_decision.py`**: `robot_cone_half_angle_deg` is literally
  `atan(robot_radius_m / distance_m)` from the slide. `evaluate()` computes
  the real angle between the gaze vector and the direction to the robot
  (`angle_between_deg`, a plain 3D dot product/arccos - no small-angle
  approximation), then applies the doc's two separate, independently-useful
  tests: `sees_robot` (peripheral - within the cone + `peripheral_buffer_
  deg`, widened by `head_only_extra_deg` when the fused source this frame
  is head-only rather than iris) and `direct_contact` (strict,
  distance-independent - within `theta_contact_deg` regardless of distance
  or source). `head_only_extra_deg` only ever widens `sees_robot`, never
  `direct_contact` - a coarser signal should get more peripheral-vision
  benefit of the doubt, not an easier claim to "direct eye contact."
- **The camera isn't assumed to sit exactly at the robot's center anymore**
  (`camera_model.CameraRobotOffset`, `config/robot_rig.yaml`). The person's
  full 3D position is reconstructed in camera-optical-frame coordinates
  (`camera_model.person_position_camera_frame` - a unit-ray scale for
  `ground_plane`'s true straight-line distance, a depth-correct
  unnormalized-ray scale for `shoulder_width`/`height`'s focal-length
  depth estimate) and compared directly against the robot's own position,
  also in camera coordinates (`camera_model.robot_center_camera_frame`).
  Both `theta_to_robot_deg` and `distance_m` (now genuinely "distance from
  the *person* to the *robot*," not to the camera) fall out of that single
  3D subtraction - no separate "direction" computation needed.
- **The offset is measured along the camera's own axes (right/down/forward
  from the lens), not the robot's body frame.** Deliberately sidesteps
  ever needing to know which way the robot's body itself faces - the cone
  test treats the robot as a plain sphere of `robot_radius_m`, with no
  orientation, matching the design doc's own model exactly - and it's
  arguably the easier thing to actually measure on a real robot: stand at
  the camera, measure forward/right/down to the robot's body center, done.
  Correct as long as the camera is rigidly bolted to the robot body (not a
  separately-moving gimbal/pan-tilt - not supported, see "Not done yet").
  All-zero offset (the project's original simplifying assumption) is still
  the default, and was confirmed to reduce the math to exactly the same
  result as before this change (`direction_to_robot == -unit_ray` to full
  float precision) - this is a strict generalization, not a rewrite of
  already-validated behavior.
- **Verified the offset matters more up close than far away, as it
  physically should** - not just assumed to work from the math. A fixed
  0.3m lateral camera-to-robot offset: at 10m, a gaze aimed exactly at the
  camera is only 1.7° off the true direction to the robot (passes both
  `sees_robot` and, barely, `direct_contact`); at 1m, the same offset
  produces a 16.7° error (still clears the wider `sees_robot` cone at that
  distance, but fails `direct_contact`'s stricter 12° regardless of
  distance). Confirmed again on real footage (today's close-range,
  handheld clip): a realistic offset (camera 0.3m above and 0.15m forward
  of the robot's own body) drops `sees_robot` from 361 to 99 of the same
  1184 frames - this clip's subjects are mostly close to the camera
  (median ~1.25m, see "Empirical validation: phase 1 vs. phase 2"), exactly
  the regime where the offset should matter most.
- **Reference pixel for the ray: nose, falling back to the eye midpoint**
  (`_evaluate_geometric` in `gaze_pipeline.py`) - picked straight from
  keypoints, not from the MediaPipe/head-crop pipeline, specifically so the
  geometric decision still runs on frames where iris gating failed outright
  (the flat `gaze_robot_deg` already does this; phase 2 matches that
  behavior rather than silently narrowing coverage).
- **Gaze vector uses real pitch, not an assumed zero.** `head_pitch_deg` is
  threaded through from B1 (previously computed but discarded everywhere -
  `_pitch, yaw, _roll = pose`); iris pitch is now smoothed with its own One
  Euro filter instance, riding along whenever a fixation's yaw already
  confirmed trustworthy (`_pitch_smoother_for` in `gaze_pipeline.py`) -
  fixation itself stays a yaw-only decision, matching the design doc,
  pitch just isn't thrown away anymore once yaw says to trust the frame.
- **Calibration is placeholder values, explicitly, not a real deployment
  rig's measurements** (`camera_height_m=1.2`, `camera_pitch_deg=0.0`,
  `robot_radius_m=0.3`) - this project has only ever run against recorded
  phone video with no known camera mount. `sees_robot`/`direct_contact` are
  only as physically meaningful as these three values; everything else in
  phase 2 (the math, the fallback cascade, the two-test structure) is
  deployment-independent and already verified against hand-computed
  expected values (see "Empirical validation").

### Iris-position veto

Head pose alone can't tell "head toward the camera, eyes looking elsewhere"
apart from real looking. On the manually labeled WhatsApp clip
(`data/ground_truth/whatsapp_1017_headpose.*`, see `labeling_tool.md`) that
case was 271 of the man's (track 2) false positives - head yaw within a few
degrees of the camera, eyes turned clearly to the side. MobileGaze couldn't
fix it: his eyes are under `min_eye_px` (12px) in 445 of 647 frames, and
even with the floor lowered to 8px every one of his 379 MobileGaze readings
then failed fixation (the reading jitters more than `fixation_max_dev_deg`
at that size). A yes/no "are the eyes aimed at the camera" check doesn't
need fixation-level precision, so it's done separately:

- `mediapipe_gate` reports `iris_horiz_pos`: where the iris sits on the line
  between the eye's two corners (0 = image-left corner, 1 = image-right,
  ~0.5 centred), averaged over both eyes. Set whenever MediaPipe found a
  face, even on frames the gate otherwise rejects - the veto is not bound by
  `min_eye_px`.
- `gaze_pipeline._apply_iris_veto`: someone looking at the camera with their
  head turned by `yaw` degrees has their iris near
  `0.5 + iris_shift_per_yaw_deg * yaw`. On a frame whose fused source is
  head-only, an iris more than `iris_veto_max_offset` eye widths from that
  sets `eyes_averted` and forces the geometric `sees_robot`/`direct_contact`
  to False (`dataclasses.replace`, so `gaze_node` publishes the vetoed
  decision with no message change). Since 2026-11-16 this applies to
  iris-sourced frames too (see "Veto on iris-sourced frames" below). `gaze_robot_deg` (the
  angle) is never touched; a consumer that thresholds that angle itself
  doesn't get the veto.

Choices, each from the same labeled data (iris landmarks recorded on every
labeled frame where MediaPipe found a face - 893 of them):

- **Correct for head turn with the raw head-pose yaw, not `yaw - bearing`.**
  The slope fitted against `yaw - bearing` came out ~0 and vetoed no more
  than no correction at all (207 vs 211 of 317 false positives at the same
  threshold); raw yaw vetoed 246. Consistent with 6DRepNet360 reading yaw
  from the face's appearance, i.e. relative to the viewing ray. The slope's
  sign is tied to `HeadPoseEstimator`'s yaw sign and the image-left-to-right
  corner ordering; `head_pose_node` passes that yaw through unchanged.
- **Centre fixed at 0.5, only the slope fitted** (0.0027 per degree). A
  fitted centre (0.494) gained nothing.
- **Leave-one-person-out** (fit the slope on two people, test on the third,
  threshold 0.12): 248 of 317 head-pose false positives vetoed, 7 of 509 true
  "looking" frames wrongly vetoed - holds across these three people, not
  just in-sample.
- **No extra eye-size floor.** It works from 8px eyes up; below 8px (17
  frames) it never fired. `mp_min_head_px` already keeps eyes at 7px or more
  on this footage.
- **Applies on blink frames too.** Noisier there (3 of the 9 wrong vetoes
  came from 109 blink frames) but still net positive (17 false positives
  removed).
- **No "eyes closed means not looking" rule.** On this clip the not-looking
  frames had slightly *more* open eyes than the looking ones for two of the
  three people, and nobody closed their eyes for long - no evidence for it
  yet.

End to end on the same clip (full pipeline, `head_only_extra_deg=35`,
scored against all three people's labels):

| | precision | recall | F1 | false positives |
|---|---|---|---|---|
| veto off (`iris_veto_max_offset` >= 1.0) | 0.60 | 0.88 | 0.71 | 351 |
| veto on (0.12) | 0.83 | 0.86 | 0.85 | 105 |
| man (track 2), off -> on | 0.38 -> 0.79 | 0.93 -> 0.91 | 0.54 -> 0.84 | 271 -> 44 |

With the veto off the export is identical, row for row, to the run before
it existed.

### Veto on iris-sourced frames (2026-11-16)

The veto first skipped iris-sourced frames, assuming MobileGaze had already
accounted for the eyes. The first live run on the Unitree R1 showed
otherwise: close up, with eyes large enough for the iris path to engage,
the person looked away with the eyes only, the iris landmarks were clearly
off-centre, and it still said "looking". MobileGaze is trained on Gaze360,
whose gaze mostly follows the head, and the `mobileone_s0` variant has
~12.6° error, so an eyes-only glance can stay inside the iris branch's
`sees_robot` allowance (robot cone + `peripheral_buffer_deg`, ~22° at 1 m).
The veto now runs on every frame with face landmarks, whatever the source.

Checked on the labeled WhatsApp clip (full chain, `--head-only-extra-deg
35`): the score is unchanged (TP 510, FP 105, FN 82). Its 11 iris-sourced
frames are all labeled "looking" and sit at iris offset 0.000-0.024, far
under 0.12, so none were wrongly vetoed. That clip has no iris-sourced
false positives, so it can't show the gain; the R1 footage would.

## Interfaces

Reuses `human_awareness_msgs/PersonDetection` a fourth time, extending it
again rather than introducing a parallel type (same reasoning
`human_tracking` and `human_head_pose` already give for their own
extensions):

- `iris_valid` (bool): an iris reading was gated through and estimated this
  frame at all (independent of whether it ended up trusted).
- `iris_fixating` (bool): `iris_valid` *and* held steady long enough
  (`fixation_min_ms`) to trust.
- `gaze_source` (string): `"head"` or `"iris"` - which signal actually
  produced `gaze_robot_deg` this frame.
- `iris_yaw_deg` (float32): the fixation-confirmed, One-Euro-filtered iris
  yaw - raw, *not* bearing-corrected (unlike `gaze_robot_deg` itself) - 0
  if not currently fixating.
- `gaze_robot_deg` (reused, now fused): unchanged in meaning (~0 = facing
  the robot) but now sourced from iris whenever `gaze_source` is `"iris"`,
  head-only otherwise - the same field downstream awareness-scoring logic
  already expects to read, now a better estimate when it can be.
- `distance_m` (float32) / `distance_method` (string): phase 2's estimated
  distance from the **person to the robot** (not to the camera - these
  differ once `config/robot_rig.yaml`'s `camera_offset_*` is nonzero) and
  which fallback tier produced the underlying camera-to-person measurement
  (`"ground_plane"`/`"shoulder_width"`/`"height"`/`"none"`) - 0/`"none"` if
  no reference pixel was even available this frame.
- `theta_to_robot_deg` (float32): the real 3D angle between the person's
  gaze vector and the true direction to the robot - the phase-2 analogue of
  `gaze_robot_deg`, but not bearing-subtracted (it's already a true angle,
  not a yaw-vs-bearing difference).
- `gaze_gate_reason` (string), `iris_offset` (float32, -1 = no face
  landmarks) and `eyes_averted` (bool), added 2026-11-16 for debugging live
  runs (`docs/context/run_recorder.md`): why the iris path did or didn't
  run, the veto's measurement, and whether it fired. Diagnostics only;
  nothing downstream decides on them.
- `sees_robot` (bool) / `direct_contact` (bool): phase 2's two decision
  outputs - peripheral ("robot is in view") and strict ("looking right at
  it") respectively. **Placeholder-calibrated** - see "Design decisions:
  Phase 2" before trusting these on anything but relative comparisons.

## Empirical validation so far

- **Full pipeline verified end-to-end at every layer, not just unit-tested
  in isolation**, each confirmed against real output rather than assumed
  from the code: `mediapipe.FaceLandmarker` on a real image (face + all 5
  iris landmarks per eye, confirmed by index); the vendored `MobileOneGaze`
  loading the real published checkpoint (zero missing/unexpected keys);
  the One Euro filter on synthetic noisy-constant vs. fast-ramp signals
  (variance cut from ~1.0 to ~0.014 on jitter, only ~1.5 degrees of lag on
  a real 42-degree movement over 0.5s); the fixation state machine
  (confirms at exactly `fixation_min_ms`, survives a blink-length gap,
  correctly resets on a genuine 70-degree jump); the fusion logic's three
  branches (iris/no-fixation/implausible) with direct assertions; a real
  ROS2 in-process run (`GazeNode` + a synthetic publisher, 20 published
  frames, 19 received, final frame shows a genuine `gaze_source="iris"`
  with a confirmed fixation) - the same "actually run it as a node, not
  just import it" standard `human_tracking_node`'s own docs hold to.
- **Real footage: engages genuinely, not just falls back to head-only
  every time.** On the first validated single-person clip (whatsapp,
  stationary, frontal, same clip used throughout `human_tracking.md`): 177
  track-frames with valid head pose, 43 (24%) reached a confirmed iris
  fixation and sourced `gaze_robot_deg` from it - visually spot-checked,
  the MediaPipe-derived face bbox sits tightly and correctly on the face
  in every checked frame.
- **A second, more dynamic multi-person clip (handheld, close-range,
  people often at an angle or moving) engaged far less: 0 of 1184 valid
  head poses reached a confirmed fixation.** Root-caused with direct
  instrumentation rather than left unexplained: 76% of attempts failed
  specifically at `min_eye_px` ("eyes too small"), measuring consistently
  12-15px against the design doc's own recommended 20px floor - confirmed
  by eye (not just numbers) that this isn't a crop/scaling bug: the saved
  head crops show a clearly visible, reasonably-sized, mostly-frontal face
  each time, just with a real corner-to-corner eye width smaller than the
  threshold at this crop size/distance/camera resolution. **This is a
  genuine calibration finding, not a defect** - `min_eye_px=20` (the
  design doc's own recommended value) is measurably strict in practice on
  real handheld footage at 640px detector input resolution, not just in
  theory.
- **Upscaling the head crop before MediaPipe gating was tested and
  rejected as a fix for the above, not assumed to help.** Raised the gate
  pass rate from 0.6% to 52% on the second clip, which looked promising -
  but overlaying the 1x and 2x-rescaled-back-to-1x landmark positions on
  the same crop showed them landing at essentially the same pixel (10.8px
  vs. 10.7px measured eye width, 1x-equivalent) - MediaPipe isn't finding
  new detail, it's reporting a larger *raw* pixel count because the
  threshold compares against whatever resolution it's handed. Upscaling
  before gating would game the metric, not fix it - confirms `min_eye_px`
  is doing its job correctly and the real eyes in this footage are simply
  small at native resolution.
- **`min_eye_px` lowered from 20.0 to 12.0, with the same kind of
  real-footage, both-directions evidence this project requires before
  changing a threshold** (e.g. `reid_threshold`'s own documented history) -
  not guessed. Swept 20/15/12/10/8 on the third (close-range, casual
  conversation) clip and checked, at every level, whether the iris
  readings that passed gating actually *agreed* with head pose
  (`|iris_yaw - head_yaw|`, independent of the `max_eye_head_deg=30`
  plausibility gate that would otherwise silently discard disagreeing
  ones) - median agreement stayed ~7 degrees at every tested value
  (15/12/10/8px), i.e. lower thresholds aren't letting noise through, the
  landmarks stay trustworthy even on small eyes (consistent with the
  upscaling finding above - MediaPipe's placement precision doesn't
  actually degrade here). Picked 12.0, not 8.0: engagement plateaus below
  ~10px (20 fixations at both 10px and 8px, no further gain) because two
  *other* gates take over as the dominant blockers at that point - on the
  same clip at 12px, `"eyes too small"` (471/1184) and `"head turned away
  from gaze range"` (468/1184, `max_yaw_gaze_deg`) are roughly tied, with
  `"iris not yet fixating or implausible vs head"` (200/1184) a distant
  third - going below 12px chases a threshold that's no longer the real
  bottleneck. Net effect on this clip: 0 -> 11 confirmed fixations (still
  a small fraction of 1184 - this is genuinely casual, handheld
  conversation footage, not a calibration/staring-at-the-robot scenario,
  so no threshold change makes iris the dominant signal here. The honest
  takeaway is in "Design decisions: Phase 2" below, not this threshold).
  Regression-checked on the first validated clip (24% fixation rate,
  min_eye_px was never the limiting factor there) - unaffected, as
  expected: 43/177, unchanged.
- **LOST tracks skipped, checked over ROS** (the same in-process run as
  `head_pose.md`'s: `HeadPoseNode` -> `GazeNode`, synthetic publisher, 5
  frames of an ACTIVE, a `LOST_KALMAN`, a `LOST_GALLERY` and an untracked
  detection with identical bbox and keypoints). On `human_gaze/gazes`,
  ACTIVE and untracked got a phase-2 decision (`ground_plane`); both LOST
  tracks came out with `head_pose_valid=false`, every gaze field at its
  default and id/state/bbox/keypoints unchanged. `GazeNode` alone, fed
  LOST tracks marked `head_pose_valid=true` with a facing head pose (yaw
  10°): `sees_robot=true` for them before the change, `head_pose_valid=
  false` with the head-pose fields zeroed, `sees_robot=false` and no
  phase-2 decision after, while the ACTIVE track with the same input
  still got `sees_robot=true`.
- **Fixation across a LOST gap** (scripted iris readings, with MediaPipe
  and MobileGaze stubbed, through the real `process_track` and through
  `GazeNode` over ROS): 0.4 s at 10°, 3 s LOST, then 10.5°. Before the
  fix the first frame back was already fixating and iris-sourced
  (10.0065° over ROS). Now it's head-only, the new fixation confirms 0.3 s
  later at exactly 10.5°, and the pitch smoother restarts too. A 0.25 s
  unfed gap (under `blink_gap_max_ms`) still keeps the fixation. On the
  labeled WhatsApp clip (full chain, `--head-only-extra-deg 35`) the
  score is unchanged: precision 0.829 / recall 0.861 (TP 510, FP 105,
  FN 82). 5 of 2001 export rows differ, all in `gaze_robot_deg`, by at
  most 2.2°, on iris frames of one person's second and later fixations,
  where the smoothers now restart. Fixation counts, `sees_robot` and
  awareness are identical.

### Empirical validation: phase 1 vs. phase 2

- **The flat phase-1 threshold produces real false positives, confirmed
  visually on real footage, not just suspected.** On a third clip (today's
  WhatsApp video, handheld, multi-person, casual conversation - iris never
  engaged at all here, 0 of 740 "looking" calls were iris-sourced), two
  concrete frames: a person turned ~33 degrees away mid-sentence, visibly
  talking to someone beside him, labeled `LOOKING AT ROBOT`; a person in
  profile, visibly facing and talking to two other people in the room, also
  labeled `LOOKING AT ROBOT`. Both are a direct consequence of the flat
  ±45-degree threshold having no notion of distance - the motivating case
  for building phase 2.
- **Phase 2's math independently verified against hand-computed expected
  values before trusting it on real footage** (not just "it runs without
  crashing"): `intrinsics_from_hfov` + `pixel_ray_camera_frame` at the
  principal point returns exactly `[0, 0, 1]`; `ground_plane_distance_m`
  with a level camera and an ankle pixel at the horizon correctly returns
  `None` (can't hit the ground), and with a known height/pixel offset
  matches the hand-computed value; `size_based_distance_m` round-trips a
  known real size/pixel width back to the exact distance it was derived
  from; `angle_between_deg` returns exactly 0 for identical vectors and 90
  for orthogonal ones; `robot_cone_half_angle_deg` matches `atan` by hand at
  two different distances; `evaluate()`'s `head_only_extra_deg` widening
  was confirmed to actually change the `sees_robot` outcome at the same
  real angle, depending only on `source` - all before ever running it on a
  frame of real video.
- **On the same two confirmed false-positive frames above, phase 2
  correctly rejects both** (`sees_robot=False`): the 33-degree-off frame
  (`theta=38.9` vs. a `cone=14.8` degree allowance at an estimated 1.1m) and
  the profile-view frame (`theta=51.8` vs. a `cone=1.5` degree allowance at
  an estimated 11.4m) - not a coincidence of the specific threshold value,
  the real angle genuinely doesn't clear the real cone in either case.
- **Across the full clip, phase 2 cuts "sees robot" calls by about
  half with the (default) zero camera-to-robot offset** (740 flat-
  threshold "looking" calls -> 361 `sees_robot=True`) and reports **zero
  `direct_contact`** the entire clip - expected, since iris never engaged
  on this footage at all (see above) and head-only gaze has no confirmed
  pitch-plus-yaw precision to support a strict, distance-independent
  claim. With a realistic nonzero offset (camera 0.3m above, 0.15m forward
  of the robot's own body - a plausible mast-mounted camera), `sees_robot`
  drops further, to 99 - this clip's subjects are mostly close to the
  camera (median ~1.56m), the regime where the offset matters most (see
  "Design decisions: Phase 2"). Consistent with phase 2 being strictly
  more conservative than the flat threshold it replaces, not a
  free-floating separate number.
- **One real robustness gap, found and deliberately left open rather than
  silently patched: an unclamped outlier distance estimate.** Across the
  same clip (zero offset), `distance_m` ranged from 1.09m to a
  clearly-wrong 378.7m (median 1.56m) - almost certainly a near-degenerate
  shoulder-width measurement (a near-profile view collapsing the pixel
  width toward zero). This doesn't cause an incorrect `sees_robot=True`
  claim (an inflated distance only *shrinks* the cone, making both tests
  strictly harder to pass that frame - fails safe, not unsafe), so it was
  not an urgent fix, but it's a real data-quality gap worth fixing before
  this distance value is used for anything beyond the cone test itself
  (e.g. logged, or consumed by a future stage) - see "Not done yet".
- **Distance-tier usage on real handheld footage**: `shoulder_width` fired
  far more often than `ground_plane` (887 vs. 173 of 1060 frames with a
  distance estimate) - ankles are frequently out of frame or
  low-confidence in handheld vertical video, so the fallback tier, not the
  preferred one, is realistically the common case on this kind of footage.
- **A real bug, found by the project owner re-watching the rendered
  video, not caught by any of the numeric checks above: the test script's
  main "LOOKING AT ROBOT" label was still driven by the flat phase-1
  threshold, not phase 2, even after phase 2 was built and verified.**
  Phase 2 was only ever drawn as a *second*, separate diagnostic line
  (`[phase2] ...`) next to the still-phase-1-driven headline label - so
  every numeric validation above was checking phase 2's own output
  correctly, but the thing actually displayed to a human watching the
  video never used it. Fixed in `test_tracking_head_pose_and_gaze_on_
  video.py`: the label now uses `geometric.sees_robot` whenever a distance
  estimate exists this frame (true for essentially every active-track
  frame, 1184/1184 before this specific fix was involved - the geometric
  result is almost always available once the head pose itself is valid),
  falling back to the flat threshold only for frames phase 2 genuinely
  couldn't evaluate at all. The two separate `[phase1]`/`[phase2]` overlay
  lines were removed at the same time (per the project owner's request) -
  there's now one label, and it's the correct one. **Lesson, not just a
  fix**: a numeric regression suite (what every check in this doc up to
  this point was) verifies the *computation*; it doesn't verify the
  computation is actually *wired to what gets shown*. Both need checking.
- **`head_only_extra_deg` lowered from 15.0 to 10.0, found via the same
  re-watch.** Even after the label bug above was fixed, id2 still showed
  genuine false "LOOKING" moments close to the camera (frame 30 of the
  third clip: head turned ~25 degrees, `theta=31`, still passed because
  `cone(13.2) + peripheral_buffer_deg(5) + head_only_extra_deg(15) =
  33.2`). Root cause: at close range the robot's own cone is already wide,
  and since iris essentially never engages on this footage (see above),
  every decision falls back to head-only, which stacks another 15 degrees
  on top - generous enough that a real 25-31 degree head turn still
  cleared it. `head_only_extra_deg=15` is the design doc's own default,
  never validated against real footage (same situation `min_eye_px` was
  in). Swept 15/10/5/0 on the third clip (361/255/51/2 `sees_robot` calls)
  and picked 10 by checking specific frames, not just the aggregate count:
  frame 30 (the false positive above) correctly flips to NOT LOOKING;
  frame 60 (head turned only ~17 degrees, genuinely oriented toward the
  camera, a hand visibly gesturing in frame) correctly stays LOOKING;
  frame 2 (head yaw ~5 degrees, squarely facing the camera) stays LOOKING
  while the person beside him, in profile and talking to someone else in
  the same frame, correctly shows NOT LOOKING. Regression-checked on the
  three earlier clips: a small, proportionate drop on each (53->45,
  166->165, 292->275) - nothing like the large, specifically-false-positive
  drop on the problem clip (361->255) - the signature of a genuine
  calibration fix, not an over-correction.

## Parameters (`GazePipeline.__init__` / `gaze_node`)

| param | default | meaning |
|---|---|---|
| `mp_min_head_px` | 80.0 | below this head-crop size, MediaPipe isn't even attempted (separate from, and larger than, B1's own `min_head_px=35`) |
| `min_eye_px` | 12.0 | eye corner-to-corner width floor - lowered from the design doc's 20.0, see "Empirical validation" |
| `max_yaw_gaze_deg` | 45.0 | head turned more than this (B1's own yaw) - skip gaze entirely, eyes aren't usable |
| `max_eye_head_deg` | 30.0 | iris yaw vs. head yaw disagreement beyond this - discard the iris reading, not partially trust it |
| `ear_min` | 0.2 | eye-aspect-ratio floor (Soukupova & Cech 2016) - below this, treated as a blink/closed eye |
| `blur_min` | 0.0 (placeholder) | Laplacian-variance floor on the eye region - design doc says "calibrated," not yet measured against real footage; 0.0 effectively disables this gate until it is |
| `fixation_min_ms` | 300.0 | minimum dwell time before a gaze direction counts as a real fixation |
| `fixation_max_dev_deg` | 4.0 | max angle drift still considered the same ongoing fixation |
| `blink_gap_max_ms` | 300.0 | a gap this short (a blink, one bad gating frame) doesn't reset an ongoing fixation |
| `one_euro_min_cutoff` | 1.0 | One Euro filter: smoothing strength while nearly still |
| `one_euro_beta` | 0.3 | One Euro filter: how much real movement speed cuts through the smoothing |
| `theta_contact_deg` | 12.0 | `direct_contact`'s strict, distance-independent angle allowance |
| `peripheral_buffer_deg` | 5.0 | degrees of slack added to the robot's own cone for `sees_robot` |
| `head_only_extra_deg` | 10.0 | extra `sees_robot` slack when the fused source this frame is head-only, not iris - lowered from the design doc's 15.0, see "Empirical validation" |
| `shoulder_width_prior_m` | 0.40 | distance-estimator fallback tier 2: assumed real shoulder width |
| `person_height_prior_m` | 1.70 | distance-estimator fallback tier 3 (last resort): assumed real person height |
| `iris_veto_max_offset` | 0.12 | on any frame with face landmarks, an iris more than this many eye widths from where it'd sit looking at the camera vetoes `sees_robot`/`direct_contact`; >= 1.0 disables - see "Iris-position veto" |
| `iris_shift_per_yaw_deg` | 0.0027 | how far (in eye widths) the iris sits off-centre per degree of head yaw while still looking at the camera |

Physical rig calibration - same `GazePipeline`/`gaze_node` constructor, but
sourced from the separate `config/robot_rig.yaml` (not `human_gaze.yaml`),
loaded as a second params file by `human_gaze.launch.py` - see "Design
decisions: Phase 2" for why these are split out:

| param | default | meaning |
|---|---|---|
| `camera_height_m` | 1.2 (placeholder) | camera height above ground, for ground-plane distance |
| `camera_pitch_deg` | 0.0 (placeholder) | camera downward tilt from level |
| `robot_radius_m` | 0.3 (placeholder) | half the robot's width - sets the cone size via `atan(robot_radius_m / distance)` |
| `camera_offset_right_m` | 0.0 | robot body center's position right of the camera (negative = left), along the camera's own axes |
| `camera_offset_down_m` | 0.0 | robot body center's position below the camera (negative = above) |
| `camera_offset_forward_m` | 0.0 | robot body center's position in front of the camera (negative = behind - the common case for a camera that sticks out past the robot's own front face) |

## Robot rig: Unitree R1 (2026-11-16)

`config/robot_rig.yaml` now holds values for a Unitree R1 using its eye
camera instead of generic placeholders. Unitree publishes only the overall
size (1230 x 357 x 190 mm) and the camera (binocular depth, 146 x 124 deg
FOV, RGB 1280x1080 @ 30 Hz). Eye height (1.10 m) and the body-center offset
(0.25 m down, 0.08 m back, and 0.03 m left since only the right eye is used) are estimates from those numbers, marked in the
file for measuring on the real robot. The center is on the upper chest with
`robot_radius_m` 0.3, so a gaze anywhere from the face to the waist counts
as looking at the robot.

`camera_hfov_deg` was deliberately left at 70. `bearing_deg` and
`intrinsics_from_hfov` assume a pinhole image, and a 146 deg lens is
fisheye-like. With 146 typed in, a person halfway to the image edge would
get a pinhole bearing of ~58 deg against ~37 deg for an equidistant fisheye,
a ~20 deg error, larger than `theta_contact_deg`. The fix is to feed a
rectified image and set the FOV from its `camera_info` fx (formula in the
yaml). The R1's actual camera topic and whether it publishes rectified
images is unknown until it's checked on the robot. The head is also 2-DOF,
and `camera_pitch_deg` is static, so tests should keep the head level.

## Not done yet

- **The iris-position veto is fitted and validated on one clip** (three
  people, one phone camera, indoor light). Leave-one-person-out supports it
  generalizing across people, but not across cameras, resolutions or
  lighting - re-score it on a second labeled clip before trusting the
  numbers above. Per-frame only (no smoothing over time).
- **`blur_min` is an unset placeholder** (0.0 - the gate never actually
  rejects anything on blur right now), consistent with the design doc's
  own "calibrat" (calibrated) note rather than a fixed recommended value -
  needs real footage with both sharp and genuinely blurry eye crops to set
  meaningfully, the same way `reid_threshold` and others were tuned
  against contrasting real examples, not guessed.
- **`min_eye_px` was recalibrated (20.0 -> 12.0) against this project's own
  ad hoc phone clips, not real target-camera footage** - still worth
  revisiting once real deployment-camera footage exists, since the right
  value is a property of that camera/resolution/distance, not a universal
  constant. Even at the recalibrated value, iris engagement on casual,
  handheld, multi-person footage stays low (11/1184 on the third clip) -
  not because of this threshold specifically (see "Empirical validation":
  `"head turned away from gaze range"` and `"eyes too small"` are now
  roughly tied as the dominant blockers), but because that kind of footage
  genuinely has few held, camera-directed fixations to find. Crop/
  resolution choices upstream (detector `imgsz`, head crop sizing) remain
  unexplored as an alternative lever.
- **Phase 2's calibration is placeholder values, not a real rig's
  measurements** (`camera_height_m`, `camera_pitch_deg`, `robot_radius_m`,
  `camera_offset_*`, all in `config/robot_rig.yaml`) - `sees_robot`/
  `direct_contact` are only physically meaningful once these are set from
  the actual deployment robot/camera, see "Design decisions: Phase 2."
- **Distance estimates aren't outlier-clamped** - confirmed on real footage
  (see "Empirical validation: phase 1 vs. phase 2"), a near-degenerate
  shoulder-width measurement can produce a wildly-wrong distance (379m seen
  in practice). Fails safe today (only ever shrinks the cone, making both
  tests stricter, never more lenient) but is a real data-quality gap if
  `distance_m` is ever consumed for anything beyond the cone test itself.
- **Real `tf` not used - the camera-to-robot offset is a fixed, manually
  entered constant** (`config/robot_rig.yaml`), not read from a live `tf`
  tree to a `robot_base_frame`. Correct for a camera rigidly bolted to the
  robot body; wrong for a camera on an independently-moving gimbal/pan-tilt
  unit, where the true offset would vary with the gimbal's own angle - not
  supported, since there's no real robot/tf tree to test against yet, only
  recorded phone video.
- **No real camera calibration (`camera_info`/`K`/`D`)** - intrinsics are
  still derived from `camera_hfov_deg` + image size (`camera_model.
  intrinsics_from_hfov`), the same approximation `bearing_deg` already
  made; a real lens has distortion and a possibly-non-square pixel aspect
  this doesn't account for.
- **Bearing-correction sign for iris gaze (`gaze_robot_deg`) is
  unconfirmed**, same caveat `head_pose.md` already carries for head yaw -
  inherited here since the same `bearing_deg` subtraction is reused
  unchanged. Phase 2's `theta_to_robot_deg` doesn't have this ambiguity
  (it's a true 3D angle, not a signed subtraction), but `gaze_robot_deg`
  itself still does.
- **Not tested against a live camera topic or the real `gaze_node` running
  alongside the rest of the pipeline live** - same gap every other
  package's "Not done yet" notes; only verified with a synthetic
  publisher in-process (see "Empirical validation") and the standalone
  video test script.
- **No automated test suite**, consistent with every other package so far
  - phase 2's math was checked with ad hoc hand-verified scripts (see
  "Empirical validation: phase 1 vs. phase 2"), not a committed test file.
