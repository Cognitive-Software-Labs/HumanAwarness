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

Also runs human_gaze's phase-2 geometric decision (distance estimate ->
real 3D angle to the robot, replacing the flat gaze_robot_deg threshold)
straight off head pose alone - the same code path
human_gaze.gaze_pipeline.GazePipeline already uses for its own head-only
fallback frames, reused here directly since it needs no iris/MediaPipe
input at all. Lets this script validate the geometric decision's effect
without paying for the full gaze/iris pipeline's model loading.

Usage:
    source .venv/bin/activate
    python3 scripts/test_tracking_and_head_pose_on_video.py path/to/video.mp4
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "human_tracking"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_head_pose"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_gaze"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_perception"))

import cv2
from ultralytics import YOLO

from _video_rotation import ensure_upright
from _yolo_tracking_adapter import yolo_result_to_detections
from human_gaze.camera_model import (
    CameraRobotOffset,
    intrinsics_from_hfov,
    person_position_camera_frame,
    robot_center_camera_frame,
)
from human_gaze.distance_estimator import estimate_distance_m
from human_gaze.gaze_estimator import gaze_yaw_pitch_to_vector
from human_gaze.geometric_decision import evaluate as evaluate_geometric
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

_NOSE, _LEFT_EYE, _RIGHT_EYE = 0, 1, 2


def _reference_pixel(keypoints, min_keypoint_conf: float):
    """Nose if confident, else the eye midpoint if both are - same
    reference-point choice human_gaze.gaze_pipeline uses for the phase-2
    geometric decision, reused here so the head-only path produces the
    same kind of estimate without needing a GazePipeline/MediaPipe at all."""
    if len(keypoints) > _NOSE and keypoints[_NOSE][2] >= min_keypoint_conf:
        return keypoints[_NOSE][0], keypoints[_NOSE][1]
    if (
        len(keypoints) > _RIGHT_EYE
        and keypoints[_LEFT_EYE][2] >= min_keypoint_conf
        and keypoints[_RIGHT_EYE][2] >= min_keypoint_conf
    ):
        return (keypoints[_LEFT_EYE][0] + keypoints[_RIGHT_EYE][0]) / 2.0, \
            (keypoints[_LEFT_EYE][1] + keypoints[_RIGHT_EYE][1]) / 2.0
    return None


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
        help="|gaze_robot_deg| at or under this is labeled 'LOOKING AT ROBOT' whenever the phase-2 "
        "geometric decision below has no distance estimate to work with this frame - a provisional "
        "demo threshold, not a validated awareness decision (see docs/context/head_pose.md)",
    )
    parser.add_argument(
        "--camera-height-m", type=float, default=1.2,
        help="phase-2 geometric decision - placeholder, not a real calibration (see docs/context/gaze.md)",
    )
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--robot-radius-m", type=float, default=0.3)
    parser.add_argument("--theta-contact-deg", type=float, default=12.0)
    parser.add_argument("--peripheral-buffer-deg", type=float, default=5.0)
    parser.add_argument(
        "--head-only-extra-deg", type=float, default=35.0,
        help="widens sees_robot only (not direct_contact) - this script is always head-only (no "
        "iris), so this always applies when a geometric decision fires at all. Raised from 10.0 "
        "to 35.0 - real-footage labeling found 10.0 too tight for head-only noise (recall "
        "0.287), 35.0 best F1 (~0.85); same default as the full gaze/iris pipeline. See "
        "docs/context/labeling_tool.md's 'Validating against real footage'",
    )
    parser.add_argument("--shoulder-width-prior-m", type=float, default=0.40)
    parser.add_argument("--person-height-prior-m", type=float, default=1.70)
    parser.add_argument(
        "--camera-offset-right-m", type=float, default=0.0,
        help="robot center's position relative to the camera, along the camera's own axes - "
        "see src/human_gaze/config/robot_rig.yaml",
    )
    parser.add_argument("--camera-offset-down-m", type=float, default=0.0)
    parser.add_argument("--camera-offset-forward-m", type=float, default=0.0)
    parser.add_argument("--output-dir", default="test_output")
    parser.add_argument(
        "--export-jsonl", default=None, metavar="PATH",
        help="also write one JSON record per visible track per frame (bbox, track_id/state, head "
        "pose, head-only gaze/is_looking, phase-2 geometric decision) to PATH, in the same schema "
        "the combined tracking+head-pose+gaze script's --export-jsonl uses (gaze_source is always "
        "\"head\" here - no iris/MediaPipe in this script - but sees_robot/direct_contact/distance_m "
        "are real, computed straight off head pose) - consumed by scripts/label_review_tool.py and "
        "scripts/score_predictions.py. Omit to skip export entirely.",
    )
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
    robot_center = robot_center_camera_frame(
        CameraRobotOffset(args.camera_offset_right_m, args.camera_offset_down_m, args.camera_offset_forward_m)
    )

    input_path = ensure_upright(video_path)

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = width * height
    dt_seconds = 1.0 / fps
    cap.release()

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
    export_file = open(args.export_jsonl, "w") if args.export_jsonl else None
    intrinsics = intrinsics_from_hfov(width, height, args.camera_hfov_deg)

    stream = detector(source=str(input_path), conf=args.conf, imgsz=args.imgsz, stream=True, verbose=False)

    t_start = time.time()
    frame_count = 0
    looking_frames = 0
    tracked_pose_frames = 0
    sees_robot_frames = 0
    direct_contact_frames = 0
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
        timestamp_s = frame_count * dt_seconds
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
                if export_file is not None:
                    export_file.write(json.dumps({
                        "frame": frame_count, "timestamp_s": timestamp_s,
                        "track_id": t.track_id, "state": t.state,
                        "bbox": [x1, y1, x2, y2], "head_pose_valid": False,
                        "head_yaw_deg": None, "head_pitch_deg": None, "bearing_deg": None,
                        "head_only_gaze_robot_deg": None, "gaze_robot_deg": None, "gaze_source": None,
                        "iris_fixating": False, "distance_m": None,
                        "sees_robot": False, "direct_contact": False,
                        "iris_offset": None, "eyes_averted": False, "is_looking": False,
                    }) + "\n")
                continue

            tracked_pose_frames += 1
            pitch, yaw, _roll = pose
            bearing = bearing_deg((x1 + x2) / 2.0, width, args.camera_hfov_deg)
            gaze = gaze_robot_deg(yaw, bearing)

            # Phase-2 geometric decision, straight off head pose - no iris/
            # MediaPipe (see test_tracking_head_pose_and_gaze_on_video.py for
            # that), just the same distance-estimate + 3D-cone test
            # human_gaze.gaze_pipeline already runs for its own head-only
            # fallback frames (GazePipeline._evaluate_geometric). Needs an
            # actual reference pixel + distance estimate to produce a real
            # result; falls back to the flat angle threshold otherwise.
            ref_pixel = _reference_pixel(t.last_keypoints, 0.3)
            distance_m, distance_method = (None, "none")
            geometric = None
            if ref_pixel is not None:
                distance_m, distance_method = estimate_distance_m(
                    t.last_keypoints, intrinsics, args.camera_height_m, args.camera_pitch_deg,
                    bbox_height_px=(y2 - y1), shoulder_width_prior_m=args.shoulder_width_prior_m,
                    person_height_prior_m=args.person_height_prior_m,
                )
                person_position = (
                    person_position_camera_frame(ref_pixel[0], ref_pixel[1], distance_m, distance_method, intrinsics)
                    if distance_m is not None else None
                )
                gaze_vector = gaze_yaw_pitch_to_vector(yaw, pitch)
                geometric = evaluate_geometric(
                    gaze_vector, person_position, robot_center, distance_method, source="head",
                    robot_radius_m=args.robot_radius_m, theta_contact_deg=args.theta_contact_deg,
                    peripheral_buffer_deg=args.peripheral_buffer_deg, head_only_extra_deg=args.head_only_extra_deg,
                )

            sees_robot = geometric is not None and geometric.sees_robot
            direct_contact = geometric is not None and geometric.direct_contact
            has_geometric = geometric is not None and geometric.distance_m is not None
            is_looking = sees_robot if has_geometric else abs(gaze) <= args.aware_threshold_deg
            if is_looking:
                looking_frames += 1
            if sees_robot:
                sees_robot_frames += 1
            if direct_contact:
                direct_contact_frames += 1

            label = "LOOKING AT ROBOT" if is_looking else "NOT LOOKING"
            color = LOOKING_COLOR if is_looking else NOT_LOOKING_COLOR
            cv2.putText(annotated, label, (x1, max(y1 - 58, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(
                annotated, f"gaze {gaze:.0f} deg", (x1, max(y1 - 34, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            if export_file is not None:
                export_file.write(json.dumps({
                    "frame": frame_count, "timestamp_s": timestamp_s,
                    "track_id": t.track_id, "state": t.state,
                    "bbox": [x1, y1, x2, y2], "head_pose_valid": True,
                    "head_yaw_deg": yaw, "head_pitch_deg": pitch, "bearing_deg": bearing,
                    "head_only_gaze_robot_deg": gaze, "gaze_robot_deg": gaze, "gaze_source": "head",
                    "iris_fixating": False, "distance_m": distance_m,
                    "sees_robot": sees_robot, "direct_contact": direct_contact,
                    "iris_offset": None, "eyes_averted": False, "is_looking": is_looking,
                }) + "\n")

        writer.write(annotated)
        frame_count += 1

    if export_file is not None:
        export_file.close()
    writer.release()
    elapsed = time.time() - t_start
    print(f"Processed {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.1f} fps)")
    print(f"Track-frames with a valid head pose: {tracked_pose_frames}")
    print(
        f"Labeled 'LOOKING AT ROBOT' (phase-2 sees_robot when a distance estimate was available, "
        f"else the flat phase-1 |gaze| <= {args.aware_threshold_deg}deg fallback): {looking_frames}"
    )
    print(f"sees_robot: {sees_robot_frames}, direct_contact: {direct_contact_frames}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
