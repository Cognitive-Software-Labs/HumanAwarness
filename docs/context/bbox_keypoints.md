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
- Params: `model_path` (default `yolo26-pose.pt`), `device` (default
  `cpu`), `confidence_threshold` (default `0.5`).

## Known gaps / not done yet

- No multi-frame tracking — `track_id` is populated only if the underlying
  Ultralytics result provides one (not enabled by default); otherwise -1.
- Not tested against ROS2 or `ultralytics` in the dev environment this was
  written in (neither was installed) — verify build + a live camera topic
  before trusting this in the field.
- Model weights (`yolo26-pose.pt`) are not vendored; Ultralytics will
  download them on first run if not already cached locally.
