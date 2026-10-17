#!/usr/bin/env python3
"""Run the full chain - detection, human_tracking's ReID tracker, and
human_head_pose's head pose - on one video, and save a single annotated
copy showing both logical parts together: each confirmed track's persistent
id/state (from human_tracking) next to whether they're currently looking at
the robot (from human_head_pose). No ROS2 required.

Unlike running the two per-part test scripts separately (each of which runs
its own independent YOLO pass with no shared identity), this feeds a single
detection pass through human_tracking.Tracker first, then adapts each
confirmed Track's last known bbox/keypoints into human_head_pose's crop
pipeline - the same shape of adaptation head_pose_node does for a track
arriving over ROS (human_head_pose/adapters.py), just without the message
round-trip.

Usage:
    source .venv/bin/activate
    python3 scripts/test_tracking_and_head_pose_on_video.py path/to/video.mp4
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "human_tracking"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_head_pose"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_perception"))

import cv2
from ultralytics import YOLO

from _video_rotation import ensure_upright
from _yolo_tracking_adapter import yolo_result_to_detections
from human_head_pose.adapters import extract_head_crop
from human_head_pose.geometry import bearing_deg, gaze_robot_deg
from human_head_pose.head_pose_estimator import HeadPoseEstimator
from human_tracking.reid_embedder import ReIDEmbedder
from human_tracking.track import ACTIVE, LOST_KALMAN
from human_tracking.tracker import Tracker

ACTIVE_COLOR = (0, 200, 0)
PREDICTED_COLOR = (0, 165, 255)  # LOST_KALMAN: Kalman-predicted position, no fresh detection this frame
HEAD_CROP_COLOR = (255, 165, 0)
LOOKING_COLOR = (0, 220, 0)
NOT_LOOKING_COLOR = (0, 0, 230)
NO_HEAD_POSE_COLOR = (160, 160, 160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--det-model", default="yolo26n-pose.pt")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="detector input resolution - higher helps separate people whose boxes overlap/merge "
        "(e.g. sitting close together) at the cost of more compute; see docs/context/human_tracking.md's "
        "multi-person validation for measured tradeoffs",
    )
    parser.add_argument("--reid-weights", default=None, help="path to osnet_x1_0 weights (default: ~/.cache)")
    parser.add_argument("--reid-device", default="cpu")
    parser.add_argument("--head-pose-weights", default=None, help="path to 6DRepNet360 weights (default: ~/.cache)")
    parser.add_argument("--head-pose-device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--reid-threshold", type=float, default=0.65)
    parser.add_argument("--camera-hfov-deg", type=float, default=70.0)
    parser.add_argument(
        "--aware-threshold-deg",
        type=float,
        default=45.0,
        help="|gaze_robot_deg| at or under this is labeled 'LOOKING AT ROBOT' - a provisional demo "
        "threshold, not a validated awareness decision (see docs/context/head_pose.md)",
    )
    parser.add_argument("--output-dir", default="test_output")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}_tracking_and_head_pose.mp4"

    print("Loading detector + ReID embedder + head pose estimator...")
    detector = YOLO(args.det_model)
    reid_embedder = ReIDEmbedder(weights_path=args.reid_weights, device=args.reid_device)
    head_pose_estimator = HeadPoseEstimator(weights_path=args.head_pose_weights, device=args.head_pose_device)
    tracker = Tracker(reid_threshold=args.reid_threshold)

    input_path = ensure_upright(video_path)

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = width * height
    dt_seconds = 1.0 / fps
    cap.release()

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    stream = detector(source=str(input_path), conf=args.conf, imgsz=args.imgsz, stream=True, verbose=False)

    t_start = time.time()
    frame_count = 0
    looking_frames = 0
    tracked_pose_frames = 0
    for result in stream:
        frame = result.orig_img
        detections = yolo_result_to_detections(result, reid_embedder)
        published = tracker.update(detections, frame_area=frame_area, dt_seconds=dt_seconds)
        # LOST_KALMAN still has a live, continuously-updated Kalman position
        # estimate (t.kalman.predict() runs every frame regardless of
        # whether a detection matched it) - showing it (distinctly, as a
        # prediction rather than a fresh detection) during the brief gaps
        # that are extremely common in a crowded/overlapping scene is far
        # more representative of what the tracker actually knows than
        # letting the box vanish on the very first missed detection.
        # LOST_GALLERY has no trustworthy motion estimate left (that's the
        # point of the timeout into that state) and stays undrawn.
        visible_tracks = [t for t in published if t.state in (ACTIVE, LOST_KALMAN)]

        # This is the adapter this script exists to demonstrate: a
        # human_tracking.Track's own last-known bbox/keypoints, straight into
        # human_head_pose's crop + pose pipeline - no ROS message in between.
        # Only meaningful for ACTIVE: a LOST_KALMAN track has no fresh
        # keypoints this frame, so there's no new crop to extract pose from -
        # reusing last_keypoints would pair stale keypoints with the
        # Kalman-predicted (moved-on) box, which don't correspond anymore.
        head_crops, crop_boxes = [], []
        for t in visible_tracks:
            if t.state == ACTIVE:
                crop, crop_bbox = extract_head_crop(frame, t.last_keypoints)
            else:
                crop, crop_bbox = None, None
            head_crops.append(crop)
            crop_boxes.append(crop_bbox)
        poses = head_pose_estimator.estimate(head_crops) if head_crops else []

        annotated = frame.copy()
        for t, crop_bbox, pose in zip(visible_tracks, crop_boxes, poses):
            box_color = ACTIVE_COLOR if t.state == ACTIVE else PREDICTED_COLOR
            x1, y1, x2, y2 = (int(v) for v in t.kalman.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            id_label = f"id {t.track_id}" if t.state == ACTIVE else f"id {t.track_id} (predicted)"
            cv2.putText(
                annotated, id_label, (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2,
            )
            if crop_bbox is not None:
                cx1, cy1, cx2, cy2 = (int(v) for v in crop_bbox)
                cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), HEAD_CROP_COLOR, 2)

            if pose is None:
                cv2.putText(
                    annotated, "no head pose", (x1, max(y1 - 34, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, NO_HEAD_POSE_COLOR, 2,
                )
                continue

            tracked_pose_frames += 1
            _pitch, yaw, _roll = pose
            bearing = bearing_deg((x1 + x2) / 2.0, width, args.camera_hfov_deg)
            gaze = gaze_robot_deg(yaw, bearing)
            is_looking = abs(gaze) <= args.aware_threshold_deg
            if is_looking:
                looking_frames += 1

            label = "LOOKING AT ROBOT" if is_looking else "NOT LOOKING"
            color = LOOKING_COLOR if is_looking else NOT_LOOKING_COLOR
            cv2.putText(annotated, label, (x1, max(y1 - 58, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(
                annotated, f"gaze {gaze:.0f} deg", (x1, max(y1 - 34, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        writer.write(annotated)
        frame_count += 1

    writer.release()
    elapsed = time.time() - t_start
    print(f"Processed {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.1f} fps)")
    print(f"Track-frames with a valid head pose: {tracked_pose_frames}")
    print(f"Of those, labeled 'LOOKING AT ROBOT' (|gaze| <= {args.aware_threshold_deg}deg): {looking_frames}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
