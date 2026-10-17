#!/usr/bin/env python3
"""Run bbox+keypoints detection + head pose (6DRepNet360) on static images and
save annotated results. No ROS2 required - exercises the same logic as
head_pose_node, just without the ROS wiring.

Primarily for empirically checking 6DRepNet360's yaw sign convention (see
docs/context/head_pose.md's "Not done yet") before trusting gaze_robot_deg's
sign in the real node: take a photo of yourself turning your head a known
way (e.g. to your own right) at a known rough position in frame, and check
that head_yaw_deg and gaze_robot_deg move the direction the design doc
predicts.

Usage:
    source .venv/bin/activate
    python3 scripts/test_head_pose_on_image.py path/to/photo.jpg
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "human_head_pose"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_perception"))

import cv2
from ultralytics import YOLO

from human_head_pose.adapters import extract_head_crop
from human_head_pose.geometry import bearing_deg, gaze_robot_deg
from human_head_pose.head_pose_estimator import HeadPoseEstimator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_image_paths(inputs: list[str]) -> list[Path]:
    paths = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"warning: '{raw}' not found, skipping", file=sys.stderr)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="image file(s) or a folder of images")
    parser.add_argument("--det-model", default="yolo26n-pose.pt")
    parser.add_argument("--head-pose-weights", default=None, help="path to 6DRepNet360 weights (default: ~/.cache)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--camera-hfov-deg", type=float, default=70.0)
    parser.add_argument("--head-k-h", type=float, default=0.8)
    parser.add_argument("--head-scale-alpha", type=float, default=1.7)
    parser.add_argument("--crop-margin", type=float, default=0.2)
    parser.add_argument("--min-head-px", type=float, default=35.0)
    parser.add_argument("--output-dir", default="test_output")
    args = parser.parse_args()

    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        print("no valid images found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading detector + head pose estimator...")
    detector = YOLO(args.det_model)
    estimator = HeadPoseEstimator(weights_path=args.head_pose_weights, device=args.device)

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"warning: could not read '{image_path}', skipping", file=sys.stderr)
            continue
        image_width = frame.shape[1]

        result = detector(frame, conf=args.conf, verbose=False)[0]
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
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
            if crop_bbox is not None:
                cx1, cy1, cx2, cy2 = (int(v) for v in crop_bbox)
                cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 165, 255), 2)
            if pose is None:
                label = "no head crop"
            else:
                pitch, yaw, roll = pose
                bearing = bearing_deg((x1 + x2) / 2.0, image_width, args.camera_hfov_deg)
                gaze = gaze_robot_deg(yaw, bearing)
                label = f"yaw {yaw:.0f} pitch {pitch:.0f} roll {roll:.0f} | bearing {bearing:.0f} gaze {gaze:.0f}"
            cv2.putText(
                annotated, label, (int(x1), max(int(y1) - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
            )

        out_path = output_dir / f"{image_path.stem}_head_pose.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"{image_path.name}: {len(person_boxes)} person(s) -> {out_path}")


if __name__ == "__main__":
    main()
