# Live visualization (annotated camera view)

## What it is

The live counterpart of the video scripts' annotated `.mp4` in
`test_output/`. On recorded clips the scripts draw the overlay themselves.
On a robot the pipeline runs as ROS nodes on a camera topic, and until now
nothing showed what it saw: only `human_perception` could publish an
annotated image (raw YOLO boxes and keypoints, with no ids, gaze or
awareness).

`visualization_node` subscribes to the camera image and to the pipeline's
final output (`human_awareness/awareness`), draws each person's overlay on
the frame the detections came from, and publishes the result as a normal
`sensor_msgs/Image` topic (`human_visualization/debug_image`). It's viewed
with the standard tools (`rqt_image_view`, RViz2, Foxglove) and recorded
with `ros2 bag`. There's no custom GUI: a custom one would only be worth
building for controls (live thresholds, clicking a person).

Code: `src/human_visualization/human_visualization/`:
- `overlay.py`: the drawing itself (`PersonOverlay`, `draw_person`,
  `draw_info`). Pure OpenCV, no ROS imports.
- `adapters.py`: `PersonDetection` → `PersonOverlay`, plus `ImageBuffer`.
- `visualization_node.py`: the ROS wrapper.

## What's drawn

Per person, the same layout as
`scripts/test_tracking_head_pose_and_gaze_on_video.py`, which now uses
`overlay.py` too:

| element | when |
|---|---|
| bbox + `id N`, green | ACTIVE track |
| bbox + `id N (predicted)`, orange | LOST_KALMAN: the Kalman prediction, no detection this frame |
| bbox + `untracked`, cyan | `track_id` -1 (when fed straight from `human_perception`) |
| `LOOKING AT ROBOT` / `NOT LOOKING` | `head_pose_valid`; phase-2 `sees_robot` when it ran, else the flat `aware_threshold_deg` fallback, the same rule as the script |
| `NOT LOOKING (eyes away)` | the iris-position veto fired (`eyes_averted`, on the message since 2026-11-16) |
| `no head pose`, grey | tracked but no head pose this frame |
| `aware V conf C` + bar | awareness value/confidence (red 0 → green 1) |
| top-left: `people: N`, `lag: X ms` | frame info; lag = node clock − frame stamp, shown only when plausible (0–60 s) |

LOST_GALLERY tracks aren't drawn: they aren't in the frame, they're only
kept for re-identification. Labels are shifted left to stay inside the
frame for a person at the right edge.

**Not drawn live:** the head-crop (blue) and MediaPipe face (magenta)
boxes the video script shows. They aren't in `PersonDetection`, and
recomputing them would mean running the crop/landmark logic a second time.
The script still draws them.

## Design decisions

- **Exact-stamp pairing, not `ApproximateTimeSynchronizer`.** Every node
  copies the camera frame's header onto its output, so the awareness
  message for frame X carries exactly X's stamp. It just arrives later, by
  the whole chain's processing time. `ImageBuffer` keeps the last
  `image_buffer_size` frames (default 60 = 2 s at 30 fps) and looks the
  stamp up exactly. The other nodes' approximate syncers keep only 10
  frames per topic, which a slow chain can outrun. An exact match against
  a deeper buffer can't pair a wrong frame, and it tells the two failure
  cases apart in its warnings (frame older than the buffer vs. frame never
  received). Unstamped input (stamp 0) falls back to the newest frame, with
  a one-time warning.
- **Draws on the final topic by default, any stage by parameter.**
  `detections_topic` can point at `human_perception/detections`,
  `human_tracking/tracked_detections`, `human_head_pose/head_poses` or
  `human_gaze/gazes` to see one stage alone; `show_gaze` /
  `show_awareness` hide parts that stage doesn't fill in.
- **Image queue depth 30** (the other nodes use 10). Decoding, drawing and
  encoding hold the single-threaded executor long enough that a burst of
  camera frames can queue up, and a frame dropped here is one that can't
  be drawn.
- **Output rate = pipeline rate.** One debug image per awareness message,
  not per camera frame: frames the pipeline itself dropped have nothing to
  draw.
- **One drawing module for both paths**, so the live view and the
  recorded `test_output/` videos look the same. The script's refactor
  left its export byte-identical. The video differs only where two
  people's labels overlap: each person is now drawn in one go, so the
  stacking order changed.

## Inputs / outputs

- In: `image_topic` (`sensor_msgs/Image`, default `/camera/image_raw`),
  `detections_topic` (`PersonDetectionArray`, default
  `human_awareness/awareness`).
- Out: `debug_image_topic` (`sensor_msgs/Image`, bgr8, default
  `human_visualization/debug_image`), with the detections message's header.
- Params: `src/human_visualization/config/human_visualization.yaml`.
- Depends on: `rclpy`, `sensor_msgs`, `cv_bridge`, `human_awareness_msgs`.
  No models, no GPU.

## Empirical validation

- **Unit tests** (`src/human_visualization/test/test_visualization.py`,
  16 cases): the looking rule (phase-2 vs flat fallback vs no head pose),
  the PersonDetection → overlay mapping (LOST_KALMAN predicted,
  LOST_GALLERY skipped, untracked, show flags), the buffer's exact match /
  miss / eviction, box colors, edge-of-frame drawing and label clamping.
- **First full live-chain run.** The 18.35.59 clip (442 frames, 576×1024)
  was published as a fake camera on `/camera/image_raw` at 30 fps, stamped
  with the wall clock. All six nodes ran under ROS (`device:=cuda:0`), with
  message counts per topic:

  | topic | messages |
  |---|---|
  | `/camera/image_raw` | 442 |
  | `human_perception/detections` | 416 |
  | `human_tracking/tracked_detections` | 410 |
  | `human_head_pose/head_poses` | 402 |
  | `human_gaze/gazes` | 392 |
  | `human_awareness/awareness` | 392 |
  | `human_visualization/debug_image` | 392 |

  Every awareness message got its frame and was drawn, with no warnings,
  and the node shut down cleanly on SIGINT. Lag on the drawn frames was
  8–51 ms. The 50 frames lost upstream (11%) are dropped by the other
  nodes' queues and approximate syncers at 30 fps, not by this node. With
  depth-10 image queues in the first runs, this node itself missed 3 of 394.

## Not done yet

- **Not run on a real camera driver**: only a video replayed as a topic.
  See README "Running on a live camera" for the QoS caveat.
- **No combined launch file** for the whole chain. Each node is started
  on its own.
- The 11% upstream frame loss at 30 fps hasn't been looked into (queue
  depths, `sync_slop_s`, sensor-data QoS).
