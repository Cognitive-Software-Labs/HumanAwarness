#!/usr/bin/env python3
"""Run bbox+keypoints + head pose + iris/gaze on static images and save
annotated results. No ROS2 required - exercises the same logic as
gaze_node, just without the ROS wiring.

Primarily for checking MediaPipe gating and MobileGaze's yaw sign
convention against a known head-turn direction, the same way
test_head_pose_on_image.py does for 6DRepNet360 - see docs/context/gaze.md.

Usage:
    source .venv/bin/activate
    python3 scripts/test_gaze_on_image.py path/to/photo.jpg
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "human_gaze"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_head_pose"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_perception"))

import cv2
from ultralytics import YOLO

from human_gaze.gaze_pipeline import GazePipeline
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
    parser.add_argument("--mp-min-head-px", type=float, default=80.0)
    parser.add_argument("--min-eye-px", type=float, default=12.0)
    parser.add_argument("--max-yaw-gaze-deg", type=float, default=45.0)
    parser.add_argument("--max-eye-head-deg", type=float, default=30.0)
    parser.add_argument(
        "--camera-height-m", type=float, default=1.2,
        help="phase-2 geometric decision - placeholder, not a real calibration (see docs/context/gaze.md)",
    )
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--robot-radius-m", type=float, default=0.3)
    parser.add_argument("--theta-contact-deg", type=float, default=12.0)
    parser.add_argument("--peripheral-buffer-deg", type=float, default=5.0)
    parser.add_argument("--head-only-extra-deg", type=float, default=35.0)
    parser.add_argument(
        "--camera-offset-right-m", type=float, default=0.0,
        help="robot center's position relative to the camera, along the camera's own axes - "
        "see src/human_gaze/config/robot_rig.yaml",
    )
    parser.add_argument("--camera-offset-down-m", type=float, default=0.0)
    parser.add_argument("--camera-offset-forward-m", type=float, default=0.0)
    parser.add_argument("--output-dir", default="test_output")
    args = parser.parse_args()

    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        print("no valid images found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading detector + head pose estimator + MediaPipe + MobileGaze...")
    detector = YOLO(args.det_model)
    head_estimator = HeadPoseEstimator(weights_path=args.head_pose_weights, device=args.device)
    # A single static image can't hold a fixation (fixation_min_ms needs
    # several consecutive frames) - lowered to near-zero here so this
    # per-image script can still show a raw gaze reading instead of always
    # falling back to head-only. Real temporal behavior is what
    # test_tracking_head_pose_and_gaze_on_video.py actually validates.
    gaze_pipeline = GazePipeline(
        device=args.device,
        mp_min_head_px=args.mp_min_head_px,
        min_eye_px=args.min_eye_px,
        max_yaw_gaze_deg=args.max_yaw_gaze_deg,
        max_eye_head_deg=args.max_eye_head_deg,
        fixation_min_ms=0.0,
        one_euro_beta=0.0,  # no temporal history to speak of on a single frame
        camera_height_m=args.camera_height_m,
        camera_pitch_deg=args.camera_pitch_deg,
        robot_radius_m=args.robot_radius_m,
        theta_contact_deg=args.theta_contact_deg,
        peripheral_buffer_deg=args.peripheral_buffer_deg,
        head_only_extra_deg=args.head_only_extra_deg,
        camera_offset_right_m=args.camera_offset_right_m,
        camera_offset_down_m=args.camera_offset_down_m,
        camera_offset_forward_m=args.camera_offset_forward_m,
    )

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"warning: could not read '{image_path}', skipping", file=sys.stderr)
            continue
        image_width = frame.shape[1]

        result = detector(frame, conf=args.conf, verbose=False)[0]
        annotated = frame.copy()

        boxes, keypoints_batch = result.boxes, result.keypoints
        n_people = 0
        if boxes is not None:
            for i in range(len(boxes)):
                n_people += 1
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                kp_xy = keypoints_batch.xy[i].tolist() if keypoints_batch is not None else []
                kp_conf = (
                    keypoints_batch.conf[i].tolist()
                    if keypoints_batch is not None and keypoints_batch.conf is not None
                    else [1.0] * len(kp_xy)
                )
                keypoints = [(x, y, c) for (x, y), c in zip(kp_xy, kp_conf)]

                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)

                _crop, crop_bbox = extract_head_crop(frame, keypoints)
                pose = head_estimator.estimate([_crop])[0] if _crop is not None else None
                if pose is None:
                    cv2.putText(
                        annotated, "no head pose", (int(x1), max(int(y1) - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2,
                    )
                    continue

                pitch, yaw, roll = pose
                image_height = frame.shape[0]
                result_gaze = gaze_pipeline.process_track(
                    track_id=i, frame=frame, keypoints=keypoints, head_yaw_deg=yaw,
                    bbox_center_x=(x1 + x2) / 2.0, image_width=image_width,
                    camera_hfov_deg=args.camera_hfov_deg, timestamp_s=0.0, dt_seconds=1.0 / 30.0,
                    head_pitch_deg=pitch, bbox_xyxy=(x1, y1, x2, y2), image_height=image_height,
                )
                bearing = bearing_deg((x1 + x2) / 2.0, image_width, args.camera_hfov_deg)
                head_only = gaze_robot_deg(yaw, bearing)

                if result_gaze.face_bbox_xyxy is not None:
                    fx1, fy1, fx2, fy2 = (int(v) for v in result_gaze.face_bbox_xyxy)
                    cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 165, 0), 2)

                label1 = f"head_yaw {yaw:.0f} head_only_gaze {head_only:.0f}"
                label2 = (
                    f"iris: {result_gaze.gate_reason} | source={result_gaze.source} "
                    f"iris_yaw={result_gaze.iris_yaw_deg:.0f} fused={result_gaze.gaze_robot_deg:.0f} [phase1]"
                )
                g = result_gaze.geometric
                if g is not None and g.distance_m is not None:
                    label3 = (
                        f"[phase2] d={g.distance_m:.1f}m ({g.distance_method}) theta={g.theta_to_robot_deg:.0f} "
                        f"sees={g.sees_robot} contact={g.direct_contact}"
                    )
                else:
                    label3 = "[phase2] no distance estimate"
                cv2.putText(
                    annotated, label1, (int(x1), max(int(y1) - 58, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2,
                )
                cv2.putText(
                    annotated, label2, (int(x1), max(int(y1) - 34, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2,
                )
                cv2.putText(
                    annotated, label3, (int(x1), max(int(y1) - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2,
                )

        out_path = output_dir / f"{image_path.stem}_gaze.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"{image_path.name}: {n_people} person(s) -> {out_path}")


if __name__ == "__main__":
    main()
