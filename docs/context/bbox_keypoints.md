# bbox + keypoints detection

## What it is

First logical part of the HumanAwareness pipeline: detects people in a
monocular RGB frame and outputs, per person, a 2D bounding box and the
COCO-17 body keypoints. This is the raw perception input that later parts
(gaze/attention estimation, proximity, awareness scoring) will consume.

## Packages

- `src/human_awareness_msgs` (ament_cmake) — custom message interfaces
  shared across the project.
- `src/human_perception` (ament_python) — the detection node itself.

## Design decisions

- **Model**: Ultralytics YOLO26-pose. One model call gives both bbox and
  keypoints together (no separate detector + pose-estimator pipeline),
  which keeps latency and package count down.
- **Input**: single monocular RGB `sensor_msgs/Image` topic. No depth
  dependency at this stage — that's deferred to whichever part needs
  distance (e.g. proximity/awareness scoring).
- **Custom messages instead of `vision_msgs`**: `vision_msgs` has no native
  keypoints field, and bolting keypoints on as a parallel array keyed by
  index is error-prone. `human_awareness_msgs/PersonDetection` bundles bbox
  + keypoints + confidence + an optional `track_id` in one message, so
  downstream nodes never have to re-correlate across topics.
- **Keypoint order**: fixed COCO-17 order (nose, eyes, ears, shoulders,
  elbows, wrists, hips, knees, ankles), matching what YOLO-pose models
  output. Each `Keypoint2D` also carries its own `name` string so consumers
  don't have to hardcode index positions.
- **No class filtering needed**: YOLO-pose models are trained single-class
  (person), so every detection above the confidence threshold is a person.

## Interfaces

- `human_awareness_msgs/msg/Keypoint2D` — one named keypoint (`x`, `y`,
  `confidence`).
- `human_awareness_msgs/msg/PersonDetection` — bbox (`bbox_x/y/width/height`),
  `confidence`, `track_id`, `keypoints[]`.
- `human_awareness_msgs/msg/PersonDetectionArray` — `header` +
  `detections[]`, published once per processed frame.

## Node: `bbox_keypoints_node`

- Subscribes: `image_topic` (`sensor_msgs/Image`, default `/camera/image_raw`).
- Publishes: `detections_topic`
  (`human_awareness_msgs/PersonDetectionArray`, default
  `human_perception/detections`).
- Optional debug publish: `annotated_image_topic` (`sensor_msgs/Image`),
  gated by `publish_annotated` (default `false`).
- Params: `model_path` (default `yolo26n-pose.pt` — nano size, for
  real-time CPU/edge use; override to `yolo26{s,m,l,x}-pose.pt` for more
  accuracy at the cost of speed), `device` (default `cpu`),
  `confidence_threshold` (default `0.5`).

## Environment: venv + ROS2 + colcon gotchas

ROS2 Jazzy (apt) and `ultralytics`/`torch` (pip) can't share one
environment cleanly. Two issues hit during setup, both resolved — see
`requirements.txt` and `README.md` for the resulting install steps:

1. **NumPy ABI conflict.** `cv_bridge` and `python3-opencv` are apt-compiled
   against NumPy 1.x. `ultralytics`/`torch` pull in NumPy 2.x by default,
   which breaks `cv_bridge`'s native extension. Fix: pin `numpy<2` in the
   venv, and don't pip-install `opencv-python` — use the system
   `python3-opencv` (visible via `--system-site-packages`) instead.
2. **colcon uses system Python regardless of the active venv.** The
   installed `colcon` binary has a hardcoded `#!/usr/bin/python3` shebang,
   so a plain `colcon build` always runs under system Python and bakes
   that interpreter into the generated node script — which then can't
   import `ultralytics` (only installed in the venv). Fix: build with
   `python3 -m colcon build` instead of `colcon build`, from inside the
   activated venv (the venv was created with `--system-site-packages`, so
   it can see the apt-installed `colcon_core` modules too).

## Verified (this was actually run, not just written)

- `colcon build` succeeds for both packages.
- `ros2 interface show human_awareness_msgs/msg/PersonDetectionArray` and
  `ros2 pkg executables human_perception` confirm the interfaces and node
  are registered correctly.
- `yolo26n-pose.pt` downloads and loads via Ultralytics.
- Ran `bbox_keypoints_node` as a real ROS2 node, published a real
  `sensor_msgs/Image` (a two-person test photo) on `/camera/image_raw`, and
  confirmed correct `PersonDetectionArray` output on
  `human_perception/detections` via `ros2 topic echo` — 2 people, 17
  keypoints each, correct bbox/confidence values.

## Known gaps / not done yet

- No multi-frame tracking — `track_id` is populated only if the underlying
  Ultralytics result provides one (not enabled by default); otherwise -1.
- Not tested against a live camera driver / real camera topic, only a
  single published test frame — confirm with an actual camera node before
  trusting this in the field.
- Model weights (`yolo26n-pose.pt`) are not vendored; Ultralytics will
  download them on first run if not already cached locally.
