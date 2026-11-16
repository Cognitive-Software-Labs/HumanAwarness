# Bringup (one launch for the whole pipeline)

## What it is

`human_awareness_bringup` is a launch-only package. Its
`full_pipeline.launch.py` starts all six stages with one command instead of
one terminal per stage: detection, tracking, head pose, gaze, awareness and
visualization. Two extras are optional: a webcam publisher and an
`rqt_image_view` window on the annotated view.

```bash
ros2 launch human_awareness_bringup full_pipeline.launch.py viewer:=true
```

## Arguments

| arg | default | what it does |
|---|---|---|
| `image_topic` | `/camera/image_raw` | camera topic the pipeline reads; remapped for all six stages at once |
| `viewer` | `false` | open `rqt_image_view` on `/human_visualization/debug_image` (needs a display) |
| `webcam` | `false` | start `image_tools cam2image` publishing on `image_topic` |
| `webcam_device` | `0` | webcam index for `cam2image` |
| `robot_rig_file` | `human_gaze`'s `robot_rig.yaml` | camera/robot calibration, forwarded to the gaze stage |
| `record` | `false` | also start `run_recorder_node`, saving the run for debugging (`docs/context/run_recorder.md`) |
| `runs_dir` | `~/human_awareness_runs` | where `record:=true` saves runs |
| `run_name` | `""` | the run's folder name (default: start time) |

## Design decisions

- **Includes each stage's own launch file** instead of re-declaring the
  nodes, so every stage keeps its own default yaml and a change there
  applies to both ways of launching.
- **Each include is in its own scoped `GroupAction`.** Every stage's launch
  file declares an argument called `params_file`. In one shared scope the
  first declaration (`bbox_keypoints.yaml`) wins and every later stage
  silently loads the detection node's yaml instead of its own. It still
  appeared to work only because the stages' built-in defaults happen to
  match their yaml. Seen on the first test run and fixed by the scoping.
  Consequence: passing `params_file:=...` to this launch would reach every
  stage (scoped groups inherit parent values), so don't. To tune one stage
  with a custom yaml, launch that stage's own launch file instead.
- **`image_topic` is a `SetRemap`** of `/camera/image_raw`, applied around
  all includes, rather than a per-node param. That way it needs no change in
  any stage's config.
- **No camera driver by default.** On a robot the camera driver is usually
  already running. `webcam:=true` uses `image_tools cam2image` because it's
  already installed with ROS desktop (`usb_cam` isn't installed here) and
  publishes raw images with reliable QoS, which the nodes require.
- **`rqt_image_view` and `image_tools` aren't in `package.xml`.** Both are
  opt-in, so a headless robot doesn't need Qt.

## Verified (2026-11-16)

Replayed a recorded clip at 10 Hz on `image_topic:=/test/cam` (laptop CPU):
- all six nodes came up, each with its own yaml and the remap;
- `/test/cam` had 5 subscribers, and every stage topic published;
- `debug_image` and `awareness` ran at ~4.7 Hz, the detector's CPU rate;
- a saved `debug_image` frame showed the id, gaze label and awareness bar;
- `viewer:=true` started `rqt_image_view` on the debug topic.

One startup warning from `visualization_node` is expected: the first
detections arrive for frames from before it subscribed. `webcam:=true` was
not run (it would turn on the laptop camera). Its only parts are the
`cam2image` node, the device param and the remap.

`ros2 topic hz` was unreliable on the image topics here (no output within
its timeout). A plain rclpy subscriber counted the messages correctly.
