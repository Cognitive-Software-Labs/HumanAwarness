#!/usr/bin/env python3
"""Run bbox+keypoints detection + head pose (6DRepNet360) on a video file and
save an annotated copy, with a per-person on-screen "looking at the robot or
not" label. No ROS2 required - exercises the same logic as head_pose_node,
just without the ROS wiring.

Usage:
    source .venv/bin/activate
    python3 scripts/test_head_pose_on_video.py path/to/video.mp4
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "human_head_pose"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_perception"))

import cv2
from ultralytics import YOLO

from _video_rotation import ensure_upright
from human_head_pose.adapters import extract_head_crop
from human_head_pose.geometry import bearing_deg, gaze_robot_deg
from human_head_pose.head_pose_estimator import HeadPoseEstimator

PERSON_COLOR = (0, 200, 0)
HEAD_CROP_COLOR = (0, 165, 255)
LOOKING_COLOR = (0, 220, 0)
NOT_LOOKING_COLOR = (0, 0, 230)
NO_CROP_COLOR = (160, 160, 160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--det-model", default="yolo26n-pose.pt")
    parser.add_argument("--head-pose-weights", default=None, help="path to 6DRepNet360 weights (default: ~/.cache)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--camera-hfov-deg", type=float, default=70.0)
    parser.add_argument("--head-k-h", type=float, default=0.8)
    parser.add_argument("--head-scale-alpha", type=float, default=1.7)
    parser.add_argument("--crop-margin", type=float, default=0.2)
    parser.add_argument("--min-head-px", type=float, default=35.0)
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
    out_path = output_dir / f"{video_path.stem}_head_pose.mp4"

    print("Loading detector + head pose estimator...")
    detector = YOLO(args.det_model)
    estimator = HeadPoseEstimator(weights_path=args.head_pose_weights, device=args.device)

    input_path = ensure_upright(video_path)

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    stream = detector(source=str(input_path), conf=args.conf, stream=True, verbose=False)

    frame_count = 0
    valid_pose_frames = 0
    looking_frames = 0
    for result in stream:
        frame = result.orig_img
        annotated = frame.copy()
        boxes, keypoints_batch = result.boxes, result.keypoints

        head_crops, crop_boxes, person_boxes = [], [], []
        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                person_boxes.append((x1, y1, x2, y2))
                kp_xy = keypoints_batch.xy[i].tolist() if keypoints_batch is not None else []
                kp_conf = (
                    keypoints_batch.conf[i].tolist()
                    if keypoints_batch is not None and keypoints_batch.conf is not None
                    else [1.0] * len(kp_xy)
                )
                kp = [(x, y, c) for (x, y), c in zip(kp_xy, kp_conf)]
                crop, crop_bbox = extract_head_crop(
                    frame, kp, 0.3, args.head_k_h, args.head_scale_alpha, args.crop_margin, args.min_head_px
                )
                crop_boxes.append(crop_bbox)
                head_crops.append(crop)

        poses = estimator.estimate(head_crops) if head_crops else []

        for (x1, y1, x2, y2), crop_bbox, pose in zip(person_boxes, crop_boxes, poses):
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), PERSON_COLOR, 2)
            if crop_bbox is not None:
                cx1, cy1, cx2, cy2 = (int(v) for v in crop_bbox)
                cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), HEAD_CROP_COLOR, 2)

            if pose is None:
                cv2.putText(
                    annotated, "no head crop", (int(x1), max(int(y1) - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, NO_CROP_COLOR, 2,
                )
                continue

            valid_pose_frames += 1
            _pitch, yaw, _roll = pose
            bearing = bearing_deg((x1 + x2) / 2.0, width, args.camera_hfov_deg)
            gaze = gaze_robot_deg(yaw, bearing)
            is_looking = abs(gaze) <= args.aware_threshold_deg
            if is_looking:
                looking_frames += 1

            label = "LOOKING AT ROBOT" if is_looking else "NOT LOOKING"
            color = LOOKING_COLOR if is_looking else NOT_LOOKING_COLOR
            cv2.putText(
                annotated, label, (int(x1), max(int(y1) - 34, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )
            cv2.putText(
                annotated, f"gaze {gaze:.0f} deg", (int(x1), max(int(y1) - 10, 34)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        writer.write(annotated)
        frame_count += 1

    writer.release()
    print(f"Processed {frame_count} frames")
    print(f"Frames with a valid head pose: {valid_pose_frames}")
    print(f"Of those, labeled 'LOOKING AT ROBOT' (|gaze| <= {args.aware_threshold_deg}deg): {looking_frames}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
