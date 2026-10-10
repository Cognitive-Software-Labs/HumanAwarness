# HumanAwareness

A ROS2 system that estimates whether a human is aware of a robot's presence
(human awareness), built as a multi-package ROS2 workspace.

See `CLAUDE.md` for a running index of what's been built, and
`docs/context/` for the detailed design notes behind each part.

## Status

- **bbox + keypoints detection** — monocular RGB person detection (bbox +
  COCO-17 keypoints) via YOLO26-pose. See `docs/context/bbox_keypoints.md`.

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

# venv for the Python deps human_perception needs (ultralytics etc.)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r src/human_perception/requirements.txt

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
