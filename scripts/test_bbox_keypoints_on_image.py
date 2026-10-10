#!/usr/bin/env python3
"""Run bbox+keypoints detection on one or more static images and save annotated
results to an output folder. No ROS2 required — exercises the same model/logic
as human_perception's bbox_keypoints_node, just without the ROS wiring, so you
can quickly sanity-check detection quality on your own photos.

Usage:
    source .venv/bin/activate
    python3 scripts/test_bbox_keypoints_on_image.py path/to/image.jpg [more images...]
    python3 scripts/test_bbox_keypoints_on_image.py path/to/a/folder/
"""

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

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


def annotate(model: YOLO, image_path: Path, conf: float) -> tuple[object, int]:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"could not read image: {image_path}")

    results = model(frame, conf=conf, verbose=False)
    result = results[0]

    # result.plot() already draws boxes + keypoint skeletons; we add a
    # per-image sequential person ID on top of that (there's no persistent
    # tracking here, just "1st/2nd/... person found in this image").
    annotated = result.plot()

    boxes = result.boxes
    num_people = 0
    if boxes is not None:
        num_people = len(boxes)
        for i in range(num_people):
            x1, _, _, y2 = boxes.xyxy[i].tolist()
            # Bottom-left of the box, to avoid colliding with Ultralytics'
            # own "person <conf>" label drawn at the top-left.
            label_pos = (int(x1) + 4, int(y2) - 10)
            cv2.putText(
                annotated,
                f"id {i + 1}",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    return annotated, num_people


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="image file(s) or a folder of images")
    parser.add_argument("--model", default="yolo26n-pose.pt", help="Ultralytics pose model path/name")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument(
        "--output-dir",
        default="test_output",
        help="folder to save annotated images into (created if missing)",
    )
    args = parser.parse_args()

    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        print("no valid images found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model '{args.model}'...")
    model = YOLO(args.model)

    for image_path in image_paths:
        annotated, num_people = annotate(model, image_path, args.conf)
        out_path = output_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"{image_path.name}: {num_people} person(s) detected -> {out_path}")


if __name__ == "__main__":
    main()
