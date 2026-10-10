# HumanAwareness

A ROS2 system that estimates whether a human is aware of a robot's presence
(human awareness), built as a multi-package ROS2 workspace.

See `CLAUDE.md` for a running index of what's been built, and
`docs/context/` for the detailed design notes behind each part.

## Status

- **bbox + keypoints detection** — monocular RGB person detection (bbox +
  COCO-17 keypoints) via YOLO26-pose. See `docs/context/bbox_keypoints.md`.

## Setup

```bash
# Python deps for human_perception (not available via rosdep)
pip install -r src/human_perception/requirements.txt

# ROS2 deps
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build
source install/setup.bash
```

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
