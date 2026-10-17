# CLAUDE.md

Context index for the HumanAwareness project. This file stays short — it only
tracks *what exists* and *where to read about it*. Detailed context for each
logical part lives in `docs/context/`.

## Project

A ROS2 system that estimates whether a human is aware of a robot's presence
(human awareness detection), built incrementally as a multi-package ROS2
workspace under `src/`.

## Workflow (how this file is maintained)

- Each new logical part (a package, node, algorithm, or pipeline stage) gets
  its own context file at `docs/context/<part>.md` describing: what it does,
  why it exists, key design decisions, inputs/outputs, and dependencies.
- This file (`CLAUDE.md`) only gets a one-line entry per part, linking to its
  context file — never the full details inline.
- `README.md` gets updated alongside with a user-facing summary of what was
  added and how to launch/run it.

## Parts

- **bbox + keypoints detection** — monocular RGB person detection via
  YOLO26-pose (`human_perception` node, `human_awareness_msgs` interfaces).
  See `docs/context/bbox_keypoints.md`.
- **human tracking / ReID** — persistent identity across occlusion
  (Kalman motion + OSNet appearance matching), wired into a ROS2 node
  (`human_tracking` package, `human_tracking_node`). See
  `docs/context/human_tracking.md`.
- **head pose relative to the robot** — per-tracked-person head
  orientation via 6DRepNet360, combined with camera bearing into a
  "facing the robot or not" angle (`human_head_pose` package,
  `head_pose_node`). See `docs/context/head_pose.md`.

<!--
Template for new entries:
- **<part name>** — one-line description. See `docs/context/<part>.md`.
-->
