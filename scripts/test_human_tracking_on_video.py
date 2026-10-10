#!/usr/bin/env python3
"""Run the full human_tracking lifecycle (detection + OSNet ReID + Kalman/
gallery lifecycle) on a video file and save an annotated copy. No ROS2
required - imports human_perception/human_tracking package sources
directly via sys.path, exercising the same tracker/track/reid_embedder
code the ROS2 node will use.

Usage:
    source .venv/bin/activate
    python3 scripts/test_human_tracking_on_video.py path/to/video.mp4
"""

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "human_tracking"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_perception"))

import cv2
from ultralytics import YOLO

from human_tracking.reid_embedder import ReIDEmbedder
from human_tracking.track import ACTIVE, LOST_GALLERY, LOST_KALMAN
from human_tracking.tracker import Detection, Tracker

STATE_COLORS = {
    ACTIVE: (0, 200, 0),       # green
    LOST_KALMAN: (0, 165, 255),  # orange
    LOST_GALLERY: (0, 0, 255),   # red
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--det-model", default="yolo26n-pose.pt")
    parser.add_argument("--reid-weights", default=None, help="path to osnet_x1_0 weights (default: ~/.cache)")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--reid-threshold", type=float, default=0.65)
    parser.add_argument("--reid-margin", type=float, default=0.08)
    parser.add_argument("--max-speed", type=float, default=3.0, help="bbox-heights/sec a lost track may have moved")
    parser.add_argument("--kalman-timeout", type=float, default=2.5, help="seconds")
    parser.add_argument("--gallery-timeout", type=float, default=45.0, help="seconds")
    parser.add_argument("--output-dir", default="test_output")
    parser.add_argument(
        "--blackout",
        default=None,
        metavar="START:COUNT",
        help=(
            "Simulate a detection dropout: ignore all real detections for COUNT frames "
            "starting at frame START (the real video/person is unaffected - only what the "
            "tracker is told it sees). Useful for demonstrating LOST_KALMAN/LOST_GALLERY "
            "transitions and ReID recovery without needing footage with a real occlusion "
            "of the right duration."
        ),
    )
    args = parser.parse_args()

    blackout_start, blackout_end = None, None
    if args.blackout:
        start_str, count_str = args.blackout.split(":")
        blackout_start = int(start_str)
        blackout_end = blackout_start + int(count_str)

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}_tracked.mp4"

    print("Loading detector + ReID embedder...")
    detector = YOLO(args.det_model)
    embedder = ReIDEmbedder(weights_path=args.reid_weights)

    tracker = Tracker(
        reid_threshold=args.reid_threshold,
        reid_margin=args.reid_margin,
        max_speed_bbox_heights_per_sec=args.max_speed,
        kalman_timeout_s=args.kalman_timeout,
        gallery_timeout_s=args.gallery_timeout,
    )

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = width * height
    dt_seconds = 1.0 / fps
    cap.release()

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    stream = detector(source=str(video_path), conf=args.conf, stream=True, verbose=False)

    t_start = time.time()
    frame_count = 0
    ever_active_ids = set()
    lost_kalman_recoveries = 0
    reid_recoveries = 0
    # Track which ids we've seen in LOST_KALMAN / LOST_GALLERY, to count recoveries
    seen_lost_kalman: set[int] = set()
    seen_lost_gallery: set[int] = set()

    for result in stream:
        frame = result.orig_img
        boxes = result.boxes
        keypoints = result.keypoints

        crops = []
        dets_meta = []
        if boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                crops.append(frame[max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)])
                if keypoints is not None:
                    kp_xy = keypoints.xy[i].tolist()
                    kp_conf = keypoints.conf[i].tolist() if keypoints.conf is not None else [1.0] * len(kp_xy)
                    kp = [(x, y, c) for (x, y), c in zip(kp_xy, kp_conf)]
                else:
                    kp = []
                dets_meta.append(((x1, y1, x2, y2), float(boxes.conf[i]), kp))

        embeddings = embedder.embed(crops) if crops else []

        detections = [
            Detection(bbox_xyxy=bbox, confidence=conf, keypoints=kp, embedding=emb)
            for (bbox, conf, kp), emb in zip(dets_meta, embeddings)
        ]

        in_blackout = blackout_start is not None and blackout_start <= frame_count < blackout_end
        if in_blackout:
            detections = []

        published = tracker.update(detections, frame_area=frame_area, dt_seconds=dt_seconds)

        annotated = frame.copy()
        for t in published:
            ever_active_ids.add(t.track_id)
            if t.state == LOST_KALMAN and t.track_id not in seen_lost_kalman:
                seen_lost_kalman.add(t.track_id)
            if t.state == LOST_GALLERY and t.track_id not in seen_lost_gallery:
                seen_lost_gallery.add(t.track_id)
            if t.state == ACTIVE:
                if t.track_id in seen_lost_kalman:
                    lost_kalman_recoveries += 1
                    seen_lost_kalman.discard(t.track_id)
                if t.track_id in seen_lost_gallery:
                    reid_recoveries += 1
                    seen_lost_gallery.discard(t.track_id)

            if t.state != ACTIVE:
                # Lost tracks have no reliable current position to show - a
                # stale/drifting box on screen is misleading, not informative.
                continue

            x1, y1, x2, y2 = (int(v) for v in t.kalman.bbox)
            color = STATE_COLORS.get(t.state, (255, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"id {t.track_id} [{t.state}]"
            cv2.putText(annotated, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        if in_blackout:
            cv2.putText(
                annotated,
                "SIMULATED DETECTION DROPOUT",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(annotated)
        frame_count += 1

    writer.release()
    elapsed = time.time() - t_start
    print(f"Processed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    print(f"Total confirmed identities ever seen: {len(ever_active_ids)}")
    print(f"Recoveries via Kalman-only re-match (LOST_KALMAN -> ACTIVE): {lost_kalman_recoveries}")
    print(f"Recoveries via ReID gallery match (LOST_GALLERY -> ACTIVE): {reid_recoveries}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
