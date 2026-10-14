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

# Build — must run via the venv's python so colcon bakes in the right
# interpreter (a bare `colcon build` always uses system Python here)
python3 -m colcon build --symlink-install
source install/setup.bash
```

Each new shell needs both `source /opt/ros/<distro>/setup.bash` and
`source .venv/bin/activate` (in that order) before `source
install/setup.bash`.

## Launch

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

Same idea, but for a video file, with persistent per-person tracking IDs
(via Ultralytics' built-in tracker) instead of per-frame numbering:

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
