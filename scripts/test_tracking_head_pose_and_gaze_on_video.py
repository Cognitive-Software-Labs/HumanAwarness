#!/usr/bin/env python3
"""Run the full chain - detection, human_tracking's ReID tracker,
human_head_pose's head pose, human_gaze's iris/gaze, and human_awareness's
per-person awareness over time - on one video, and save a single annotated
copy. No ROS2 required.

Extends scripts/test_tracking_and_head_pose_on_video.py with the gaze stage:
per confirmed track, re-derives the same head crop for MediaPipe gating,
then MobileGaze, then fixation + One Euro filtering + fusion with head
pose - the same shape of adaptation gaze_node does for a track arriving
over ROS, just without the message round-trip. This is the primary way
human_gaze gets validated against real footage (see docs/context/gaze.md)
since fixation detection needs real consecutive frames, which a single
test image can't exercise. Awareness gets the same cue awareness_node builds
from a PersonDetection (an ACTIVE track's phase-2 sees_robot, tagged with its
gaze source), so the two give the same values for the same frames.

Usage:
    source .venv/bin/activate
    python3 scripts/test_tracking_head_pose_and_gaze_on_video.py path/to/video.mp4
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
sys.path.insert(0, str(REPO_ROOT / "src" / "human_awareness"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_visualization"))

import cv2
from ultralytics import YOLO

from _video_rotation import ensure_upright
from _yolo_tracking_adapter import yolo_result_to_detections
from human_awareness.awareness import AwarenessAggregator, AwarenessCue
from human_gaze.gaze_pipeline import GazePipeline
from human_head_pose.adapters import extract_head_crop
from human_head_pose.geometry import bearing_deg, gaze_robot_deg
from human_head_pose.head_pose_estimator import HeadPoseEstimator
from human_tracking.reid_embedder import ReIDEmbedder
from human_tracking.track import ACTIVE, LOST_KALMAN
from human_tracking.tracker import Tracker
from human_visualization.overlay import PersonOverlay, draw_person

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--det-model", default="yolo26n-pose.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--reid-weights", default=None)
    parser.add_argument("--reid-device", default="cpu")
    parser.add_argument("--head-pose-weights", default=None)
    parser.add_argument("--head-pose-device", default="cpu")
    parser.add_argument("--gaze-weights", default=None)
    parser.add_argument("--gaze-device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--reid-threshold", type=float, default=0.65)
    parser.add_argument("--camera-hfov-deg", type=float, default=70.0)
    parser.add_argument("--mp-min-head-px", type=float, default=80.0)
    parser.add_argument("--min-eye-px", type=float, default=12.0)
    parser.add_argument("--max-yaw-gaze-deg", type=float, default=45.0)
    parser.add_argument("--max-eye-head-deg", type=float, default=30.0)
    parser.add_argument(
        "--aware-threshold-deg",
        type=float,
        default=45.0,
        help="|gaze_robot_deg| at or under this is labeled 'LOOKING AT ROBOT' - the flat phase-1 demo "
        "threshold, shown alongside the phase-2 geometric decision for comparison (see docs/context/gaze.md)",
    )
    parser.add_argument(
        "--camera-height-m", type=float, default=1.2,
        help="phase-2 geometric decision - placeholder, not a real calibration (see docs/context/gaze.md)",
    )
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--robot-radius-m", type=float, default=0.3)
    parser.add_argument("--theta-contact-deg", type=float, default=12.0)
    parser.add_argument("--peripheral-buffer-deg", type=float, default=5.0)
    parser.add_argument("--head-only-extra-deg", type=float, default=10.0)
    parser.add_argument(
        "--camera-offset-right-m", type=float, default=0.0,
        help="robot center's position relative to the camera, along the camera's own axes - "
        "see src/human_gaze/config/robot_rig.yaml",
    )
    parser.add_argument("--camera-offset-down-m", type=float, default=0.0)
    parser.add_argument("--camera-offset-forward-m", type=float, default=0.0)
    parser.add_argument(
        "--iris-veto-max-offset", type=float, default=0.12,
        help="veto a 'looking' when the iris sits more than this many eye widths from where it "
        "would looking at the camera (>= 1.0 disables) - see docs/context/gaze.md",
    )
    parser.add_argument("--iris-shift-per-yaw-deg", type=float, default=0.0027)
    parser.add_argument(
        "--tau-awareness-s", type=float, default=20.0,
        help="awareness fades with this time constant once forgetting starts - "
        "see docs/context/awareness.md",
    )
    parser.add_argument(
        "--forget-delay-s", type=float, default=4.0,
        help="awareness doesn't fall at all for this long after the person last saw the robot",
    )
    parser.add_argument("--cue-max-age-s", type=float, default=0.5)
    parser.add_argument("--awareness-rise-time-s", type=float, default=0.5)
    parser.add_argument("--awareness-confidence-time-s", type=float, default=2.0)
    parser.add_argument("--awareness-weight-iris", type=float, default=1.0)
    parser.add_argument("--awareness-weight-head", type=float, default=0.5)
    parser.add_argument("--output-dir", default="test_output")
    parser.add_argument(
        "--export-jsonl", default=None, metavar="PATH",
        help="also write one JSON record per visible track per frame (bbox, track_id/state, "
        "head pose, fused gaze, geometric decision, awareness) to PATH - the structured counterpart to the "
        "annotated video, consumed by scripts/label_review_tool.py and scripts/score_predictions.py. "
        "Omit to skip export entirely.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}_tracking_head_pose_and_gaze.mp4"

    print("Loading detector + ReID embedder + head pose estimator + MediaPipe + MobileGaze...")
    detector = YOLO(args.det_model)
    reid_embedder = ReIDEmbedder(weights_path=args.reid_weights, device=args.reid_device)
    head_pose_estimator = HeadPoseEstimator(weights_path=args.head_pose_weights, device=args.head_pose_device)
    gaze_pipeline = GazePipeline(
        gaze_weights_path=args.gaze_weights,
        device=args.gaze_device,
        mp_min_head_px=args.mp_min_head_px,
        min_eye_px=args.min_eye_px,
        max_yaw_gaze_deg=args.max_yaw_gaze_deg,
        max_eye_head_deg=args.max_eye_head_deg,
        camera_height_m=args.camera_height_m,
        camera_pitch_deg=args.camera_pitch_deg,
        robot_radius_m=args.robot_radius_m,
        theta_contact_deg=args.theta_contact_deg,
        peripheral_buffer_deg=args.peripheral_buffer_deg,
        head_only_extra_deg=args.head_only_extra_deg,
        camera_offset_right_m=args.camera_offset_right_m,
        camera_offset_down_m=args.camera_offset_down_m,
        camera_offset_forward_m=args.camera_offset_forward_m,
        iris_veto_max_offset=args.iris_veto_max_offset,
        iris_shift_per_yaw_deg=args.iris_shift_per_yaw_deg,
    )
    tracker = Tracker(reid_threshold=args.reid_threshold)
    awareness = AwarenessAggregator(
        tau_s=args.tau_awareness_s,
        forget_delay_s=args.forget_delay_s,
        cue_max_age_s=args.cue_max_age_s,
        rise_time_s=args.awareness_rise_time_s,
        confidence_time_s=args.awareness_confidence_time_s,
        weight_iris=args.awareness_weight_iris,
        weight_head=args.awareness_weight_head,
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

    stream = detector(source=str(input_path), conf=args.conf, imgsz=args.imgsz, stream=True, verbose=False)

    t_start = time.time()
    frame_count = 0
    looking_frames = 0
    tracked_pose_frames = 0
    iris_sourced_frames = 0
    fixating_frames = 0
    sees_robot_frames = 0
    direct_contact_frames = 0
    eyes_averted_frames = 0
    awareness_cue_frames = 0
    for result in stream:
        frame = result.orig_img
        detections = yolo_result_to_detections(result, reid_embedder)
        published = tracker.update(detections, frame_area=frame_area, dt_seconds=dt_seconds)
        visible_tracks = [t for t in published if t.state in (ACTIVE, LOST_KALMAN)]
        gaze_pipeline.prune({t.track_id for t in published})

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
        # Awareness needs every visible track's cue before it can be drawn or
        # exported, so people are drawn and rows written after this loop.
        awareness_cues: dict[int, AwarenessCue] = {}
        pending_rows = []
        for t, crop_bbox, pose in zip(visible_tracks, crop_boxes, poses):
            x1, y1, x2, y2 = (int(v) for v in t.kalman.bbox)
            overlay = PersonOverlay(
                bbox_xyxy=(x1, y1, x2, y2), track_id=t.track_id, predicted=t.state != ACTIVE,
                head_crop_bbox=crop_bbox, head_pose_valid=pose is not None,
            )

            if pose is None:
                pending_rows.append((t, overlay, {
                    "frame": frame_count, "timestamp_s": timestamp_s,
                    "track_id": t.track_id, "state": t.state,
                    "bbox": [x1, y1, x2, y2], "head_pose_valid": False,
                    "head_yaw_deg": None, "head_pitch_deg": None, "bearing_deg": None,
                    "head_only_gaze_robot_deg": None, "gaze_robot_deg": None, "gaze_source": None,
                    "iris_fixating": False, "distance_m": None,
                    "sees_robot": False, "direct_contact": False,
                    "iris_offset": None, "eyes_averted": False, "is_looking": False,
                }))
                continue

            tracked_pose_frames += 1
            pitch, yaw, _roll = pose
            bearing = bearing_deg((x1 + x2) / 2.0, width, args.camera_hfov_deg)
            head_only_gaze = gaze_robot_deg(yaw, bearing)

            if t.state == ACTIVE:
                gaze_result = gaze_pipeline.process_track(
                    track_id=t.track_id, frame=frame, keypoints=t.last_keypoints, head_yaw_deg=yaw,
                    bbox_center_x=(x1 + x2) / 2.0, image_width=width, camera_hfov_deg=args.camera_hfov_deg,
                    timestamp_s=timestamp_s, dt_seconds=dt_seconds, head_pitch_deg=pitch,
                    bbox_xyxy=t.kalman.bbox, image_height=height,
                )
                gaze = gaze_result.gaze_robot_deg
                source = gaze_result.source
                if source == "iris":
                    iris_sourced_frames += 1
                iris_fixating = gaze_result.iris_fixating
                if iris_fixating:
                    fixating_frames += 1
                overlay.face_bbox = gaze_result.face_bbox_xyxy
                geometric = gaze_result.geometric
                iris_offset = gaze_result.iris_offset
                eyes_averted = gaze_result.eyes_averted
            else:
                # LOST_KALMAN: no fresh crop this frame, so no new gaze
                # attempt either - head-only fallback only, same reasoning
                # as the head-pose-only script for why this state skips it.
                gaze = head_only_gaze
                source = "head"
                iris_fixating = False
                geometric = None
                iris_offset = None
                eyes_averted = False

            sees_robot = geometric is not None and geometric.sees_robot
            direct_contact = geometric is not None and geometric.direct_contact
            has_geometric = geometric is not None and geometric.distance_m is not None
            # Phase 2 (the real geometric cone test) drives the label
            # whenever it has a distance estimate to work with - it's
            # strictly more accurate than the flat phase-1 threshold, which
            # is only a fallback for frames where phase 2 couldn't run at
            # all (no confident reference pixel/distance this frame).
            # The veto already cleared sees_robot; this also covers the flat fallback.
            is_looking = (sees_robot if has_geometric else abs(gaze) <= args.aware_threshold_deg) and not eyes_averted
            if is_looking:
                looking_frames += 1
            if sees_robot:
                sees_robot_frames += 1
            if direct_contact:
                direct_contact_frames += 1
            if eyes_averted:
                eyes_averted_frames += 1

            overlay.is_looking = is_looking
            overlay.eyes_averted = eyes_averted

            # Same rule as human_awareness.adapters.person_detection_to_cue:
            # only an ACTIVE track whose phase-2 decision actually ran. The
            # flat-threshold fallback above stays a display-only label.
            if t.state == ACTIVE and has_geometric:
                awareness_cues[t.track_id] = AwarenessCue(sees_robot=sees_robot, source=source)
                awareness_cue_frames += 1

            pending_rows.append((t, overlay, {
                "frame": frame_count, "timestamp_s": timestamp_s,
                "track_id": t.track_id, "state": t.state,
                "bbox": [x1, y1, x2, y2], "head_pose_valid": True,
                "head_yaw_deg": yaw, "head_pitch_deg": pitch, "bearing_deg": bearing,
                "head_only_gaze_robot_deg": head_only_gaze, "gaze_robot_deg": gaze, "gaze_source": source,
                "iris_fixating": iris_fixating,
                "distance_m": geometric.distance_m if geometric is not None else None,
                "sees_robot": sees_robot, "direct_contact": direct_contact,
                "iris_offset": iris_offset, "eyes_averted": eyes_averted, "is_looking": is_looking,
            }))

        # Every published track, LOST_GALLERY included (no cue), so its state
        # keeps decaying while it's gone and carries on if ReID brings it back.
        published_ids = {t.track_id for t in published}
        estimates = {tid: awareness.observe(tid, timestamp_s, awareness_cues.get(tid)) for tid in published_ids}
        awareness.prune(published_ids)

        for t, overlay, row in pending_rows:
            estimate = estimates[t.track_id]
            overlay.awareness = (estimate.value, estimate.confidence)
            draw_person(annotated, overlay)
            if export_file is not None:
                row["awareness_value"] = estimate.value
                row["awareness_confidence"] = estimate.confidence
                export_file.write(json.dumps(row) + "\n")

        writer.write(annotated)
        frame_count += 1

    if export_file is not None:
        export_file.close()
    writer.release()
    elapsed = time.time() - t_start
    print(f"Processed {frame_count} frames in {elapsed:.1f}s ({frame_count / elapsed:.1f} fps)")
    print(f"Track-frames with a valid head pose: {tracked_pose_frames}")
    print(f"Of those, iris was fixating: {fixating_frames}, iris actually sourced gaze: {iris_sourced_frames}")
    print(
        f"Labeled 'LOOKING AT ROBOT' (phase-2 sees_robot when a distance estimate was available, "
        f"else the flat phase-1 |gaze| <= {args.aware_threshold_deg}deg fallback): {looking_frames}"
    )
    print(f"sees_robot: {sees_robot_frames}, direct_contact: {direct_contact_frames}")
    print(f"Frames with eyes judged averted (iris-position veto): {eyes_averted_frames}")
    print(f"Track-frames that fed awareness a cue: {awareness_cue_frames}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
