# HumanAwareness

A ROS2 system that estimates whether a human is aware of a robot's presence
(human awareness), built as a multi-package ROS2 workspace.

See `CLAUDE.md` for a running index of what's been built, and
`docs/context/` for the detailed design notes behind each part.

## Status

- **bbox + keypoints detection** — monocular RGB person detection (bbox +
  COCO-17 keypoints) via YOLO26-pose. See `docs/context/bbox_keypoints.md`.
- **human tracking / ReID** — persistent identity across brief occlusion
  or leaving/re-entering frame (Kalman motion + OSNet appearance matching,
  fused with a keypoint-derived body-orientation signal so a reappearance
  at a different facing direction isn't missed), wired into a ROS2 node
  (`human_tracking_node`). See `docs/context/human_tracking.md`.
- **head pose relative to the robot** — per-tracked-person head
  orientation (6DRepNet360) combined with that person's bearing from the
  camera into a single "facing the robot or not" angle, wired into a ROS2
  node (`human_head_pose` package, `head_pose_node`). See
  `docs/context/head_pose.md`.
- **iris / gaze relative to the robot** — MediaPipe Face Landmarker gates
  (eye size, blink, blur, head-turn range), MobileGaze estimates eye
  direction, fused with head pose into the final "facing the robot" angle
  (iris takes over once a fixation is confirmed and plausible, head-only
  otherwise), wired into a ROS2 node (`human_gaze` package, `gaze_node`).
  See `docs/context/gaze.md`.
- **manual labeling / review tool** — frame-by-frame ground-truth labeling
  for gaze/awareness and ReID identity against real footage, scored
  against exported predictions for precision/recall and false-merge/
  ID-switch counts. See `docs/context/labeling_tool.md`.
- **awareness over time** — the first per-person state (stage D): for each
  track_id, a value + confidence for "knows the robot is there", rising
  within about a second while the gaze stage says the person sees the
  robot, fading over ~8 s afterwards, and kept across LOST/ReID gaps
  (`human_awareness` package, `awareness_node`). See
  `docs/context/awareness.md`.
- **live visualization** — the live counterpart of `test_output/`'s
  annotated videos: the camera image with each person's bbox, id,
  looking label and awareness bar drawn on, published as a ROS image topic
  to watch in `rqt_image_view` / RViz2 / Foxglove (`human_visualization`
  package, `visualization_node`). See `docs/context/visualization.md`.

## Setup

`human_perception` needs `ultralytics`/`torch`, which don't mix cleanly
with ROS2's apt-installed, NumPy-1.x-compiled `cv_bridge`/`python3-opencv`.
Use a venv built with `--system-site-packages` so it can still see those,
and build with `python3 -m colcon build` (not plain `colcon build`) so the
venv's interpreter — not the system one — ends up in the generated node
script. See `docs/context/bbox_keypoints.md` for why.

```bash
source /opt/ros/<distro>/setup.bash   # e.g. jazzy

# ROS2 deps for the custom message package
rosdep install --from-paths src --ignore-src -r -y

# venv for the Python deps human_perception + human_tracking need
# (ultralytics, gdown for OSNet weights, etc.)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r src/human_perception/requirements.txt
pip install -r src/human_tracking/requirements.txt
pip install -r src/human_head_pose/requirements.txt
# human_gaze's mediapipe dependency needs a non-standard install - see
# src/human_gaze/requirements.txt for why (a plain `pip install -r` would
# break cv_bridge the same way ultralytics/numpy did for human_perception):
pip install --no-deps mediapipe
pip install "protobuf>=4.25.3,<5"

# Build — must run via the venv's python so colcon bakes in the right
# interpreter (a bare `colcon build` always uses system Python here)
python3 -m colcon build --symlink-install
source install/setup.bash
```

Each new shell needs both `source /opt/ros/<distro>/setup.bash` and
`source .venv/bin/activate` (in that order) before `source
install/setup.bash`.

### Running offline (no wifi)

Every model is downloaded on first use and cached under `~/.cache/human_*`
(about 150 MB in total; the `.venv` with torch/CUDA is the big part, ~5.7 GB).
Fetch them all once while online:

```bash
python3 scripts/download_models.py            # add --extras for yolo26s-pose + OSNet-AIN
```

After that the whole pipeline runs with no network. To move it to a machine
that has never been online, copy `~/.cache/human_perception`,
`human_tracking`, `human_head_pose` and `human_gaze` over (same paths), plus
a `.venv` built with the same Python version. The detection node sets
`YOLO_OFFLINE=1` itself; for the no-ROS test scripts, `export YOLO_OFFLINE=1`
avoids Ultralytics' online check stalling on a network without internet.

## Launch

### Everything at once

```bash
ros2 launch human_awareness_bringup full_pipeline.launch.py viewer:=true
```

Starts all six stages below plus, with `viewer:=true`, a window showing what
the robot sees with each person's box, id, LOOKING / NOT LOOKING and
awareness drawn on. Other arguments:
- `image_topic:=/your/camera/topic` if the robot's camera doesn't publish on
  `/camera/image_raw`.
- `webcam:=true` to use a laptop webcam (`webcam_device:=N` to pick one).
- `robot_rig_file:=...` for your robot's calibration.

The window needs a screen. On a robot without one, leave `viewer` off and
run the viewer on a laptop on the same network with the same
`ROS_DOMAIN_ID`:
`ros2 run rqt_image_view rqt_image_view /human_visualization/debug_image`.
To use a custom yaml for one stage, launch that stage's own launch file
below instead (details: `docs/context/bringup.md`).

### bbox + keypoints detection

```bash
ros2 launch human_perception bbox_keypoints.launch.py
```

Defaults to subscribing on `/camera/image_raw` and publishing
`human_perception/detections` (`human_awareness_msgs/PersonDetectionArray`).

Override all params with your own yaml file:

```bash
ros2 launch human_perception bbox_keypoints.launch.py \
  params_file:=/path/to/my_params.yaml
```

Or tweak a couple of params directly via `ros2 run` (bypasses the launch
file, useful for quick testing):

```bash
ros2 run human_perception bbox_keypoints_node --ros-args \
  -p image_topic:=/my_camera/image_raw -p device:=cuda:0
```

(Default params live in `src/human_perception/config/bbox_keypoints.yaml`.)
For multiple people close together (e.g. seated at a table, bodies
overlapping from the camera's angle), `model_path:=yolo26s-pose.pt` +
`imgsz:=1280` measurably helped detect/separate them on real test footage —
see `docs/context/human_tracking.md`'s multi-person validation for the
actual numbers and why that combination (not just "bigger is always
better") was the one that won. Not adopted as the default - it's a
compute/accuracy tradeoff only the target hardware's capability can decide.

### human tracking / ReID

```bash
ros2 launch human_tracking human_tracking.launch.py
```

Subscribes to the raw camera image (`image_topic`, default
`/camera/image_raw`) *and* `human_perception`'s detections
(`detections_topic`, default `human_perception/detections`) — run both
nodes together, since the tracker needs actual pixel crops for ReID, not
just boxes. Publishes `human_tracking/tracked_detections`
(`human_awareness_msgs/PersonDetectionArray`, reusing the same message
type `human_perception` uses, with `track_id`/`state` populated for every
currently-confirmed identity).

```bash
ros2 launch human_tracking human_tracking.launch.py \
  params_file:=/path/to/my_params.yaml
```

```bash
ros2 run human_tracking human_tracking_node --ros-args \
  -p device:=cuda:0 -p reid_threshold:=0.6
```

`device` controls the OSNet ReID embedder (`cpu` by default); `cuda:0`
gives a 5-7x speedup with identical tracking results if a GPU is
available. (Default params live in
`src/human_tracking/config/human_tracking.yaml`; see
`docs/context/human_tracking.md` for what each one trades off.)

### head pose relative to the robot

```bash
ros2 launch human_head_pose human_head_pose.launch.py
```

Subscribes to the raw camera image (`image_topic`, default
`/camera/image_raw`) *and* `human_tracking`'s tracked detections
(`detections_topic`, default `human_tracking/tracked_detections`) — run all
three nodes together. Publishes `human_head_pose/head_poses`
(`human_awareness_msgs/PersonDetectionArray` again, with
`head_yaw_deg`/`head_pitch_deg`/`head_roll_deg`/`bearing_deg`/
`gaze_robot_deg` populated per detection — see `docs/context/head_pose.md`
for what each means and how `gaze_robot_deg` is derived). Only ACTIVE
tracks (and untracked detections) get a head pose: `LOST_KALMAN`/
`LOST_GALLERY` tracks carry their last match's keypoints, so they pass
through with `head_pose_valid` false.

```bash
ros2 launch human_head_pose human_head_pose.launch.py \
  params_file:=/path/to/my_params.yaml
```

```bash
ros2 run human_head_pose head_pose_node --ros-args \
  -p device:=cuda:0 -p camera_hfov_deg:=78.0
```

`camera_hfov_deg` (default 70°) is an assumed horizontal field of view used
to compute each person's bearing from the camera — set it to your actual
camera's real FOV, it isn't read from `CameraInfo` yet (see
`docs/context/head_pose.md`'s "Not done yet"). (Default params live in
`src/human_head_pose/config/human_head_pose.yaml`.)

### iris / gaze relative to the robot

```bash
ros2 launch human_gaze human_gaze.launch.py
```

Subscribes to the raw camera image *and* `human_head_pose`'s output
(`detections_topic`, default `human_head_pose/head_poses`) — run all four
nodes together; gaze is only attempted for a detection whose head pose is
already valid, and never for a `LOST_*` track. Publishes `human_gaze/gazes`
(`human_awareness_msgs/PersonDetectionArray` again, with `gaze_robot_deg`
now fused from iris whenever `gaze_source` is `"iris"`, plus
`iris_valid`/`iris_fixating`/`iris_yaw_deg` — see `docs/context/gaze.md`
for what each means and why iris only ever fully replaces head pose, never
partially blends with it).

Also publishes the phase-2 geometric decision (`distance_m`,
`distance_method`, `theta_to_robot_deg`, `sees_robot`, `direct_contact`) —
a real robot-sized cone test at the person's estimated distance, replacing
`gaze_robot_deg`'s flat angle threshold, and accounting for the camera not
necessarily being mounted exactly at the robot's own center.

When that decision comes from head pose alone, an iris-position check can
veto it: if the iris sits clearly off to one side of the eye (for the
person's head turn), `sees_robot`/`direct_contact` are set to false. This
catches "head toward the camera, eyes looking elsewhere". On the labeled
test clip it raised precision from 0.60 to 0.83, while recall dropped from
0.88 to 0.86. Tune or disable it with `iris_veto_max_offset` (0.12;
`>= 1.0` turns it off) — see `docs/context/gaze.md`'s "Iris-position veto".

Physical rig calibration (camera height/tilt/mounting offset, robot
radius) lives in its own file, `src/human_gaze/config/robot_rig.yaml` —
loaded automatically alongside `human_gaze.yaml` by the launch file above,
kept separate specifically so setting up a new robot only means filling in
that one small, heavily-commented file:

```bash
ros2 launch human_gaze human_gaze.launch.py \
  robot_rig_file:=/path/to/my_robot_rig.yaml
```

**Every value in `robot_rig.yaml` is a placeholder, not a real
calibration** — see `docs/context/gaze.md`'s "Design decisions: Phase 2"
before trusting `sees_robot`/`direct_contact` on anything but relative
comparisons.

```bash
ros2 launch human_gaze human_gaze.launch.py \
  params_file:=/path/to/my_params.yaml
```

```bash
ros2 run human_gaze gaze_node --ros-args \
  -p device:=cuda:0 -p min_eye_px:=15.0
```

`min_eye_px` (design doc's recommended default was 20px; this project's
own default is now 12px, lowered against real handheld footage — see
`docs/context/gaze.md`'s "Empirical validation") and `head_only_extra_deg`
(design doc's default 15°, this project's now 10°, same reasoning) were
both measured stricter/looser in practice than they looked on paper —
check that section before trusting either default on your own camera.
(Default params live in `src/human_gaze/config/human_gaze.yaml`.)

### awareness over time

```bash
ros2 launch human_awareness human_awareness.launch.py
```

Subscribes to `gaze_node`'s output (`gazes_topic`, default
`human_gaze/gazes`) — run it after the four nodes above. Publishes
`human_awareness/awareness`: the same `PersonDetectionArray`, one message
per input message with the same header, with `awareness_value` and
`awareness_confidence` (both 0..1) filled in for every tracked person,
LOST ones included.

- `awareness_value` rises while the person sees the robot (`sees_robot`)
  and fades over `tau_awareness_s` (8 s) after they stop. A new person
  starts at 0.5 ("don't know"), and someone the camera can't see drifts
  back to 0.5, never to 0.
- `awareness_confidence` says how much recent observation backs the
  value: 0 for a new or long-unseen person. Ignore the value when it's low
  (the design doc uses 0.3).
- Iris-sourced decisions count double head-only ones.
- Only fresh observations count: an ACTIVE track with a valid head pose
  and a phase-2 decision.

The state is kept per `track_id` while `human_tracking` still publishes
it, so it carries on when ReID brings a person back. Time comes from the
message stamps, so a recorded bag gives the same result at any playback
speed. Unstamped input (stamp 0) is timed by arrival instead. A message
delivered out of order is dropped, and a jump back of more than 1 s (a
looping bag) resets the state. See `docs/context/awareness.md` for the
model, its parameters (`src/human_awareness/config/human_awareness.yaml`)
and how it behaves on the labeled clip.

```bash
ros2 run human_awareness awareness_node --ros-args -p tau_awareness_s:=5.0
```

Unit tests for the core (no ROS needed):

```bash
PYTHONPATH=src/human_awareness python3 -m pytest src/human_awareness/test -q
```

### live visualization (see what the robot sees)

```bash
ros2 launch human_visualization human_visualization.launch.py
```

Subscribes to the camera image (`image_topic`, default `/camera/image_raw`)
and the awareness output (`detections_topic`, default
`human_awareness/awareness`). It publishes `human_visualization/debug_image`:
the camera frame each result was computed on, with the same overlay as the
full-chain video script. That's the bbox and `id N` (green, or orange
"predicted" while the tracker coasts on Kalman), LOOKING AT ROBOT /
NOT LOOKING, the awareness value + bar, and a top-left people count and
pipeline lag. It runs no models.

Watch it live:

```bash
ros2 run rqt_image_view rqt_image_view /human_visualization/debug_image
```

(or add an Image display in RViz2, or open it in Foxglove through
`foxglove_bridge`). Record a session to look at or score later:

```bash
ros2 bag record /camera/image_raw /human_visualization/debug_image /human_awareness/awareness
```

To see one stage alone, point it at that stage's topic, e.g.
`-p detections_topic:=human_tracking/tracked_detections -p show_gaze:=false
-p show_awareness:=false` for tracking ids only. If the chain is slow and the
node warns that detections are older than its buffer, raise
`image_buffer_size` (default 60 frames). Params:
`src/human_visualization/config/human_visualization.yaml`.

Unit tests (no ROS needed):

```bash
PYTHONPATH=src/human_visualization python3 -m pytest src/human_visualization/test -q
```

### Running on a live camera

Every node takes its frames from `/camera/image_raw`, so a live camera
just means running a camera driver that publishes there, e.g.:

```bash
ros2 run usb_cam usb_cam_node_exe --ros-args -r image_raw:=/camera/image_raw
```

Then start the five pipeline nodes above and the visualization. Notes:

- Raw `sensor_msgs/Image` only, not `CompressedImage`.
- The nodes subscribe with default (reliable) QoS. A driver that publishes
  best-effort (sensor-data QoS) is silently never received: check with
  `ros2 topic info -v /camera/image_raw`.
- At 30 fps, about 11% of frames are dropped along the chain (see
  `docs/context/visualization.md`). The design doc targets ~10-15 fps.
- Set `camera_hfov_deg` and `robot_rig.yaml` to your real camera and robot.
- Packages are installed as copies, not symlinks, even with
  `--symlink-install` here: rebuild after editing a node.

## Testing on a static image (no ROS2 needed)

To quickly check detection quality on your own photo(s) — no camera, no
`ros2 launch`, just the model:

```bash
source .venv/bin/activate
python3 scripts/test_bbox_keypoints_on_image.py path/to/photo.jpg
# or a whole folder of images:
python3 scripts/test_bbox_keypoints_on_image.py path/to/folder/
```

Saves an annotated copy of each image to `test_output/` (bbox, keypoint
skeleton, confidence, and a per-image `id N` label per detected person),
and prints a per-image detection count to the terminal. `test_output/` is
gitignored — it's generated output, not something to commit.

## Testing on a video (no ROS2 needed)

All four video test scripts below auto-correct for phone-video rotation
metadata (e.g. a WhatsApp clip recorded in portrait but stored as
landscape pixels + a rotation tag) via `scripts/_video_rotation.py` - a
rotated copy is cached under `~/.cache/human_awareness_scripts/` on first
use. OpenCV and Ultralytics both ignore that metadata on their own, so
without this a rotated clip gets silently processed sideways and every
downstream result (bbox, keypoints, head pose) is meaningless - see
`docs/context/human_tracking.md` for a confirmed real case.

Same idea as testing on an image, but for a video file, with persistent
per-person tracking IDs (via Ultralytics' built-in tracker) instead of
per-frame numbering:

```bash
source .venv/bin/activate
python3 scripts/test_bbox_keypoints_on_video.py path/to/video.mp4
```

Saves an annotated copy to `test_output/<name>_annotated.mp4` and prints
the frame count + number of unique tracked people. Detection/tracking
quality depends a lot on camera angle and how large people are in frame —
see `docs/context/bbox_keypoints.md` for what was actually tested.

## Testing human tracking / ReID (no ROS2 needed)

One extra dependency beyond `human_perception`'s (`gdown`, to download
OSNet's weights on first use):

```bash
source .venv/bin/activate
pip install -r src/human_tracking/requirements.txt
python3 scripts/test_human_tracking_on_video.py path/to/video.mp4
```

Saves an annotated copy to `test_output/<name>_tracked.mp4` — only
`ACTIVE` tracks are drawn (a lost track has no reliable position to show),
with the pose skeleton overlaid — and prints the number of confirmed
identities (and how many were later reclaimed back into an older identity
by reconciliation — see below) plus how many were recovered via simple
motion re-match vs. via ReID. Useful flags:

```bash
# simulate a detection dropout (e.g. to force/demo a LOST_GALLERY -> ACTIVE
# ReID recovery) without needing footage with a real occlusion of the
# right duration - ignores real detections for COUNT frames from START:
python3 scripts/test_human_tracking_on_video.py path/to/video.mp4 \
  --blackout START:COUNT

# tune the lifecycle/matching parameters (see docs/context/human_tracking.md
# for what each one trades off, especially --reid-threshold)
python3 scripts/test_human_tracking_on_video.py path/to/video.mp4 \
  --reid-threshold 0.65 --kalman-timeout 2.5 --gallery-timeout 45 \
  --reconcile-window 1.5

# run the ReID embedder on a GPU instead of CPU (5-7x faster, verified
# identical tracking results - see docs/context/human_tracking.md)
python3 scripts/test_human_tracking_on_video.py path/to/video.mp4 \
  --reid-device cuda:0

# multiple people close together (bodies overlapping from the camera's
# angle) - a bigger detector + higher input resolution measurably helped
# separate them on real footage, see docs/context/human_tracking.md
python3 scripts/test_human_tracking_on_video.py path/to/video.mp4 \
  --det-model yolo26s-pose.pt --imgsz 1280
```

`--reconcile-window` controls how long (in seconds) a just-created track
can still lose its detection to an older lost identity via ReID instead of
staying a separate person — this is what lets a reappearance get
recognized a few frames late (e.g. once enough of the body becomes
visible again) instead of permanently splitting into two identities. See
`docs/context/human_tracking.md` for what was actually tested (including
real phone-recorded footage) and the known tradeoffs — in particular, the
default `--reid-threshold` was lowered from an initially more
conservative value after real testing, and that threshold choice plus the
reconciliation stage were both re-tested against a real crowded scene and
**confirmed to produce real false merges between different people** -
`--density-threshold-step`/`--density-margin-step` (raise the bar when
several people are simultaneously plausible candidates) are there as a
partial, no-regression mitigation, but testing found they don't fully fix
it - see `docs/context/human_tracking.md` for exactly why, and don't trust
this in a scene where several similarly-dressed people are likely.

## Testing head pose (no ROS2 needed)

```bash
source .venv/bin/activate
python3 scripts/test_head_pose_on_image.py path/to/photo.jpg
```

Saves an annotated copy to `test_output/<name>_head_pose.jpg` (person bbox
in green, the extracted head crop in orange) and prints the per-person
yaw/pitch/roll, bearing, and combined gaze angle. Downloads 6DRepNet360's
weights to `~/.cache/human_head_pose/` on first use. Useful for checking
6DRepNet360's output against a known head-turn direction before trusting it
in the live node — see `docs/context/head_pose.md`'s "Empirical validation"
section for what's been checked so far and what hasn't.

Same idea, but for a video file, with an on-screen "LOOKING AT ROBOT" /
"NOT LOOKING" label per person (green/red) based on `gaze_robot_deg`:

```bash
python3 scripts/test_head_pose_on_video.py path/to/video.mp4 \
  --aware-threshold-deg 45
```

Saves an annotated copy to `test_output/<name>_head_pose.mp4` and prints
how many frames got a valid head pose and how many of those were labeled
"looking." `--aware-threshold-deg` is a provisional demo cutoff for this
script, not a validated awareness threshold.

## Testing human tracking + head pose together (no ROS2 needed)

The two scripts above each run their own independent detection pass, with
no shared identity between them. This instead runs one detection pass
through `human_tracking`'s tracker first, then feeds each confirmed track
into `human_head_pose`, so you see both logical parts together: a
persistent track id next to whether that person is currently looking at
the robot.

```bash
source .venv/bin/activate
python3 scripts/test_tracking_and_head_pose_on_video.py path/to/video.mp4 \
  --reid-device cuda:0 --head-pose-device cuda:0
```

Saves an annotated copy to
`test_output/<name>_tracking_and_head_pose.mp4` (green box + `id N` from
tracking, blue head crop, green/red "LOOKING AT ROBOT"/"NOT LOOKING" +
gaze angle from head pose, all on the same person). A track currently
`LOST_KALMAN` (no fresh detection this frame, but the motion model still
has a recent predicted position) is drawn too, in orange with
`id N (predicted)` and no head pose, rather than vanishing outright.

For multiple people close together, `--det-model yolo26s-pose.pt --imgsz
1280` measurably helped (see `docs/context/human_tracking.md`'s
multi-person validation) - not the default, since it's a compute/accuracy
tradeoff:

```bash
python3 scripts/test_tracking_and_head_pose_on_video.py path/to/video.mp4 \
  --det-model yolo26s-pose.pt --imgsz 1280 \
  --reid-device cuda:0 --head-pose-device cuda:0
```

## Testing iris / gaze (no ROS2 needed)

```bash
source .venv/bin/activate
python3 scripts/test_gaze_on_image.py path/to/photo.jpg
```

Saves an annotated copy to `test_output/<name>_gaze.jpg` (green person box,
orange head crop implied by head pose, magenta face crop from MediaPipe)
and prints, per person: the head-only vs. fused `[phase1]` gaze angle and
which source won, plus the `[phase2]` geometric decision (estimated
distance + method, the real angle to the robot, `sees`/`contact`). A single
image can't hold a real fixation, so `fixation_min_ms` is set to 0 here
specifically so you can still see a raw iris reading — real temporal
fixation behavior is what the combined video script below actually
exercises.

## Testing human tracking + head pose + gaze together (no ROS2 needed)

```bash
source .venv/bin/activate
python3 scripts/test_tracking_head_pose_and_gaze_on_video.py path/to/video.mp4 \
  --reid-device cuda:0 --head-pose-device cuda:0 --gaze-device cuda:0
```

Saves an annotated copy to
`test_output/<name>_tracking_head_pose_and_gaze.mp4` — same as the
tracking+head-pose script above, plus a magenta MediaPipe face crop, the
`[phase1]` flat gaze angle with `(iris)`/`(head)` showing which source won
that frame, and the `[phase2]` geometric decision (estimated distance +
method, angle to the robot, `sees`/`contact`) in a second line below it.
Prints how many track-frames reached a confirmed iris fixation, how many
actually had `gaze_robot_deg` sourced from iris (not always the same number
— a fixation can still fail the plausibility-vs-head check), and both the
`[phase1]` "LOOKING AT ROBOT" tally and the `[phase2]` `sees_robot`/
`direct_contact` tallies side by side. See `docs/context/gaze.md`'s
"Empirical validation" sections for real numbers on three different real
clips, including a case where the flat phase-1 threshold produced
visually-confirmed false positives that phase 2 correctly rejects, and
where `min_eye_px` turned out stricter in practice than expected.

Each person also gets their awareness over time, drawn above the box as
`aware 0.82 conf 0.61` plus a bar (red at 0, green at 1). It is computed
the same way `awareness_node` does it. `--export-jsonl` rows carry it as
`awareness_value` / `awareness_confidence`. Tune it with
`--tau-awareness-s`, `--cue-max-age-s`, `--awareness-rise-time-s`,
`--awareness-confidence-time-s` and `--awareness-weight-iris` /
`--awareness-weight-head` (see `docs/context/awareness.md`).

(ReID got a dedicated toggle-based labeling tool briefly, then it was
dropped — real ReID errors turned out visually obvious just watching the
tracked video, not worth a structured labeling pass for this project's
footage. `label_review_tool.py`'s `i`/`x` keys below still handle an
identity correction when you spot one by eye.)

## Manually labeling head-pose-only ground truth (no ROS2 needed)

For checking head pose's own "looking at the robot" signal specifically
(no iris/gaze/MediaPipe models to load, so this is faster to iterate on
than the combined tool below if gaze/iris isn't what you're validating
right now):

```bash
source .venv/bin/activate
python3 scripts/test_tracking_and_head_pose_on_video.py path/to/video.mp4 \
  --export-jsonl data/ground_truth/video_headpose.jsonl
python3 scripts/label_review_tool.py path/to/video.mp4 \
  --predictions data/ground_truth/video_headpose.jsonl
```

Same tool, same keys as below — `l`/`k`/`u` to label looking/not-looking/
unsure. `gaze_source` is always `"head"` (no iris fusion - that needs
MediaPipe, see the combined script below), but the phase-2 geometric
decision (distance estimate -> real 3D angle to the robot) still runs,
straight off head pose - it only needs keypoints + camera params, not
iris, so `sees_robot`/`direct_contact`/`distance_m` are real here too, not
placeholders. Tune its angular tolerance with `--head-only-extra-deg` if
recall looks too low for your footage - see `docs/context/gaze.md` for
what each geometric-decision parameter trades off.

## Manually labeling ground truth and scoring predictions (no ROS2 needed)

First export structured predictions alongside the usual annotated video —
one JSON record per visible track per frame (bbox, track_id/state, head
pose, fused gaze, geometric decision, the final `is_looking` call):

```bash
source .venv/bin/activate
python3 scripts/test_tracking_head_pose_and_gaze_on_video.py path/to/video.mp4 \
  --export-jsonl data/ground_truth/video.jsonl
```

(Use a tracked directory like `data/ground_truth/`, not `test_output/` —
the latter is gitignored, but your labels below are real manual work worth
keeping.) Then review it frame-by-frame and label ground truth:

```bash
python3 scripts/label_review_tool.py path/to/video.mp4 \
  --predictions data/ground_truth/video.jsonl
```

Click a person's box to select their track, then `l`/`k`/`u` to label the
current frame looking / not-looking / unsure (held until the next label),
`i` to correct an ID switch (merge two track_ids into one person_id), `x`
to correct a false merge (split a track into two person_ids from this
frame on), `]`/`[`/arrows to navigate, `n`/`N` to jump straight to the next
disagreement once some labels exist, `e`/`E` to jump to the next track
appearing/disappearing, `h` for the full key list, `s` to save. Labels
are saved to `data/ground_truth/video.labels.json`, and also on any exit:
`q`/Esc, closing the window, or the process being terminated.

Once you've labeled some frames (on one video or several), score the
predictions against them:

```bash
python3 scripts/score_predictions.py data/ground_truth/*.jsonl
```

Prints a gaze/awareness confusion matrix (precision, recall, false-positive
rate) plus ReID error counts (ID switches, false merges) per video and in
aggregate, with the exact frame ranges of every disagreement so you can
jump straight back into `label_review_tool.py` at an edge case instead of
re-scrubbing the whole clip. See `docs/context/labeling_tool.md` for the
on-disk label format and design rationale.
