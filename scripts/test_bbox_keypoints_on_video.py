#!/usr/bin/env python3
"""Run bbox+keypoints detection on a video file and save an annotated copy.
No ROS2 required — exercises the same model as human_perception's
bbox_keypoints_node, with persistent per-person tracking IDs added via
Ultralytics' built-in tracker (not used by the ROS node itself, which has
no tracker enabled yet — see docs/context/bbox_keypoints.md).

Usage:
    source .venv/bin/activate
    python3 scripts/test_bbox_keypoints_on_video.py path/to/video.mp4
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="path to a video file")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="Ultralytics pose model path/name")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument(
        "--output-dir",
        default="test_output",
        help="folder to save the annotated video into (created if missing)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="stop after this many frames (default: whole video)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        raise SystemExit(f"video not found: {video_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_path.stem}_annotated.mp4"

    print(f"Loading model '{args.model}'...")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # avc1 (H.264) instead of mp4v: mp4v writes an MPEG-4 Part 2 stream that
    # most modern players/browsers can't decode (no bundled codec for it),
    # which is why earlier output files failed to open. H.264 is nearly
    # universally supported.
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    # persist=True keeps the same tracker (and ID assignment) alive across
    # the stream=True generator's frames, instead of restarting per frame.
    results_stream = model.track(
        source=str(video_path),
        conf=args.conf,
        persist=True,
        stream=True,
        verbose=False,
    )

    frame_count = 0
    unique_ids = set()
    for result in results_stream:
        annotated = result.plot()
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for i in range(len(boxes)):
                track_id = int(boxes.id[i])
                unique_ids.add(track_id)
                x1, _, _, y2 = boxes.xyxy[i].tolist()
                cv2.putText(
                    annotated,
                    f"id {track_id}",
                    (int(x1) + 4, int(y2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        writer.write(annotated)
        frame_count += 1
        if args.max_frames and frame_count >= args.max_frames:
            break

    writer.release()
    print(f"Processed {frame_count} frames, {len(unique_ids)} unique track id(s) -> {out_path}")


if __name__ == "__main__":
    main()
