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
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "human_tracking"))
sys.path.insert(0, str(REPO_ROOT / "src" / "human_perception"))

import cv2
from ultralytics import YOLO

from _video_rotation import ensure_upright
from _yolo_tracking_adapter import yolo_result_to_detections
from human_tracking.reid_embedder import REID_MODELS, ReIDEmbedder
from human_tracking.track import ACTIVE, EDGE_SIDES_CHOICES, EDGE_SIDES_OFF, LOST_GALLERY, LOST_KALMAN
from human_tracking.tracker import Tracker

# Only ACTIVE tracks are ever drawn (see the main loop) - a lost track has
# no reliable current position to show - so there is just one draw color.
ACTIVE_COLOR = (0, 200, 0)  # green

# Standard COCO-17 keypoint order + skeleton edges (same pairs Ultralytics'
# own pose plotting uses), 0-indexed.
KEYPOINT_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]
MIN_KEYPOINT_CONF_TO_DRAW = 0.3


def draw_keypoints(frame, keypoints: list[tuple[float, float, float]], color) -> None:
    """Draw keypoint dots + skeleton lines for keypoints confident enough to trust."""
    confident = [conf >= MIN_KEYPOINT_CONF_TO_DRAW for _x, _y, conf in keypoints]

    for i, j in KEYPOINT_SKELETON:
        if i >= len(keypoints) or j >= len(keypoints):
            continue
        if not (confident[i] and confident[j]):
            continue
        xi, yi, _ = keypoints[i]
        xj, yj, _ = keypoints[j]
        cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), color, 2, cv2.LINE_AA)

    for (x, y, conf), is_confident in zip(keypoints, confident):
        if not is_confident:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)


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
    parser.add_argument(
        "--reid-model",
        default="osnet_x1_0",
        choices=REID_MODELS,
        help="ReID backbone (both MSMT17-trained) - see human_tracking/reid_embedder.py",
    )
    parser.add_argument(
        "--reid-normalize",
        action="store_true",
        help="apply torchreid's ImageNet mean/std input normalization before the ReID model",
    )
    parser.add_argument("--reid-weights", default=None, help="path to the --reid-model weights (default: ~/.cache)")
    parser.add_argument("--reid-device", default="cpu", help="e.g. cpu, cuda:0 - device for the OSNet embedder")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--reid-threshold", type=float, default=0.65)
    parser.add_argument("--reid-margin", type=float, default=0.08)
    parser.add_argument("--max-speed", type=float, default=3.0, help="bbox-heights/sec a lost track may have moved")
    parser.add_argument("--kalman-timeout", type=float, default=2.5, help="seconds")
    parser.add_argument("--gallery-timeout", type=float, default=45.0, help="seconds")
    parser.add_argument(
        "--reconcile-window",
        type=float,
        default=1.5,
        help="seconds after a track's creation during which it can still lose its detection "
        "to an older lost identity via ReID (see tracker.py's module docstring)",
    )
    parser.add_argument(
        "--density-threshold-step",
        type=float,
        default=0.05,
        help="extra reid_threshold required per additional simultaneous spatial contender "
        "(density gating - see tracker.py's _find_reid_matches docstring)",
    )
    parser.add_argument(
        "--density-margin-step",
        type=float,
        default=0.05,
        help="extra reid_margin required per additional simultaneous spatial contender",
    )
    parser.add_argument(
        "--orientation-relief",
        type=float,
        default=0.12,
        help="similarity bonus granted to a ReID comparison when the query and gallery sample have "
        "confidently-known, different keypoint-derived orientations (e.g. FRONTAL vs BACK) - see "
        "human_tracking/orientation.py",
    )
    parser.add_argument(
        "--containment-suppression-threshold",
        type=float,
        default=0.9,
        help="drop a raw detection whose bbox is at least this fraction covered by a larger "
        "detection's bbox - a duplicate/fragment detection of the same person (common when people "
        "sit very close together), not a second person - see tracker.py's suppress_contained_detections",
    )
    parser.add_argument(
        "--torso-rescue-weight",
        type=float,
        default=0.4,
        help="how much a higher torso-only similarity can lift a lower full-crop similarity, per "
        "gallery entry (0 = full-crop only, 1 = old behavior where torso alone can fully decide) - "
        "see tracker.py's Track.best_similarity",
    )
    parser.add_argument(
        "--reconcile-heading-rank-weight",
        type=float,
        default=0.15,
        help="during reconciliation, how much a lost track's Kalman-velocity heading can shift the "
        "ranking (not eligibility) when choosing between multiple already-qualified candidates - 0 "
        "disables it. See tracker.py's _find_reid_matches docstring",
    )
    parser.add_argument(
        "--reconcile-orientation-rank-weight",
        type=float,
        default=0.12,
        help="during reconciliation, how much keypoint-derived orientation consistency can shift the "
        "ranking (not eligibility) when choosing between multiple already-qualified candidates - 0 "
        "disables it",
    )
    parser.add_argument(
        "--cooccurrence-veto",
        type=int,
        default=1,
        help="1 = never reconcile a young track into a lost identity it was ever seen in the same frame "
        "as (two simultaneously-visible tracks are two different people); 0 disables. See tracker.py",
    )
    parser.add_argument(
        "--cooccurrence-max-overlap",
        type=float,
        default=0.5,
        help="two same-frame tracks only count as co-visible if their boxes overlap less than this "
        "(intersection over the smaller box) - guards against fragment/duplicate boxes of one person",
    )
    parser.add_argument(
        "--bystander-negative-check",
        type=int,
        default=0,
        help="1 = only revive a lost identity on a detection that resembles it more (by --bystander-margin) "
        "than any other track visible elsewhere in the same frame; 0 disables. See tracker.py",
    )
    parser.add_argument("--bystander-margin", type=float, default=0.0)
    parser.add_argument(
        "--assignment",
        choices=("greedy", "hungarian"),
        default="greedy",
        help="stage-1 motion/IoU assignment: greedy (highest IoU first) or hungarian (maximum total "
        "IoU) - see tracker.py's optimal_assignment",
    )
    parser.add_argument(
        "--assignment-stats",
        action="store_true",
        help="print how often greedy and optimal assignment would disagree per matching stage",
    )
    parser.add_argument(
        "--edge-embedding-sides",
        choices=EDGE_SIDES_CHOICES,
        default=EDGE_SIDES_OFF,
        help="withhold ReID embeddings (no gallery update, no ReID/reconciliation query) for boxes touching "
        "the image border: 'horizontal' = left/right only, 'all' = also top/bottom, 'off' = never. The box "
        "is still motion-tracked - see human_tracking/track.py's touches_image_edge",
    )
    parser.add_argument(
        "--edge-embedding-margin-px",
        type=float,
        default=2.0,
        help="a box within this many pixels of a checked image border counts as cut off",
    )
    parser.add_argument(
        "--no-edge-defers-reconcile-window",
        dest="edge_defers_reconcile_window",
        action="store_false",
        help="with the edge gate on, let a young track's reconciliation window run even while it has only "
        "seen edge-gated crops (by default the window starts once appearance first becomes available)",
    )
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
    parser.add_argument(
        "--dump-tracks",
        default=None,
        metavar="PATH.jsonl",
        help="write one JSON line per frame with every published track's id/state/bbox (plus the "
        "cumulative merged-id map), for diffing two runs frame by frame",
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
    embedder = ReIDEmbedder(
        weights_path=args.reid_weights,
        device=args.reid_device,
        model=args.reid_model,
        normalize=args.reid_normalize,
    )

    tracker = Tracker(
        reid_threshold=args.reid_threshold,
        reid_margin=args.reid_margin,
        max_speed_bbox_heights_per_sec=args.max_speed,
        kalman_timeout_s=args.kalman_timeout,
        gallery_timeout_s=args.gallery_timeout,
        reconcile_window_s=args.reconcile_window,
        density_threshold_step=args.density_threshold_step,
        density_margin_step=args.density_margin_step,
        orientation_relief=args.orientation_relief,
        containment_suppression_threshold=args.containment_suppression_threshold,
        torso_rescue_weight=args.torso_rescue_weight,
        reconcile_heading_rank_weight=args.reconcile_heading_rank_weight,
        reconcile_orientation_rank_weight=args.reconcile_orientation_rank_weight,
        cooccurrence_veto=bool(args.cooccurrence_veto),
        cooccurrence_max_overlap=args.cooccurrence_max_overlap,
        bystander_negative_check=bool(args.bystander_negative_check),
        bystander_margin=args.bystander_margin,
        assignment_method=args.assignment,
        collect_assignment_stats=args.assignment_stats,
        edge_embedding_sides=args.edge_embedding_sides,
        edge_embedding_margin_px=args.edge_embedding_margin_px,
        edge_defers_reconcile_window=args.edge_defers_reconcile_window,
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
    dump_file = open(args.dump_tracks, "w") if args.dump_tracks else None

    stream = detector(source=str(input_path), conf=args.conf, imgsz=args.imgsz, stream=True, verbose=False)

    t_start = time.time()
    frame_count = 0
    ever_active_ids = set()
    lost_kalman_recoveries = 0
    reid_recoveries = 0
    # Track ids currently in some lost state, to detect the frame a track
    # comes back to ACTIVE. Which *mechanism* recovered it (motion/IoU vs
    # ReID) is read from t.last_match_via_reid, not from which lost state it
    # was in - a LOST_KALMAN track can also be recovered via ReID fallback
    # (see tracker.py), so the lost state alone isn't a reliable proxy.
    seen_lost: set[int] = set()
    embed_stats: dict = {}
    all_vetoes: list[dict] = []
    all_merged_ids: dict[int, int] = {}

    for result in stream:
        frame = result.orig_img
        detections = yolo_result_to_detections(
            result,
            embedder,
            edge_embedding_sides=args.edge_embedding_sides,
            edge_embedding_margin_px=args.edge_embedding_margin_px,
            stats=embed_stats,
        )

        in_blackout = blackout_start is not None and blackout_start <= frame_count < blackout_end
        if in_blackout:
            detections = []

        published = tracker.update(
            detections, frame_area=frame_area, dt_seconds=dt_seconds, frame_size=(width, height)
        )

        annotated = frame.copy()
        for t in published:
            ever_active_ids.add(t.track_id)
            if t.state in (LOST_KALMAN, LOST_GALLERY):
                seen_lost.add(t.track_id)
            elif t.state == ACTIVE and t.track_id in seen_lost:
                seen_lost.discard(t.track_id)
                if t.last_match_via_reid:
                    reid_recoveries += 1
                else:
                    lost_kalman_recoveries += 1

            if t.state != ACTIVE:
                # Lost tracks have no reliable current position to show - a
                # stale/drifting box on screen is misleading, not informative.
                continue

            x1, y1, x2, y2 = (int(v) for v in t.kalman.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), ACTIVE_COLOR, 2)
            label = f"id {t.track_id} [{t.state}]"
            cv2.putText(
                annotated, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACTIVE_COLOR, 2, cv2.LINE_AA
            )
            draw_keypoints(annotated, t.last_keypoints, ACTIVE_COLOR)

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

        new_vetoes = tracker.veto_events
        all_vetoes.extend(new_vetoes)
        all_merged_ids.update(tracker.merged_ids)
        if dump_file is not None:
            record = {
                "vetoes": new_vetoes,
                "frame": frame_count,
                "tracks": [
                    {"id": t.track_id, "state": t.state, "bbox": [round(float(v), 1) for v in t.kalman.bbox]}
                    for t in published
                ],
                "merged": {str(k): v for k, v in all_merged_ids.items()},
            }
            dump_file.write(json.dumps(record) + "\n")

        writer.write(annotated)
        frame_count += 1

    writer.release()
    if dump_file is not None:
        dump_file.close()
    elapsed = time.time() - t_start
    print(f"Processed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    # Set difference, not a count subtraction: a merge can in principle land
    # on the very same frame a track is promoted (confirm_frames hits and a
    # reconciliation match both resolving that frame), in which case the
    # merged id was never actually published and was never in
    # ever_active_ids to begin with - subtracting counts would wrongly
    # dock it twice.
    net_ids = ever_active_ids - set(all_merged_ids.keys())
    print(f"Distinct track ids ever confirmed: {len(ever_active_ids)}")
    print(
        f"Of those, later reclaimed by reconciliation (same person, id merged back): {len(all_merged_ids)}"
    )
    print(f"Net distinct identities: {len(net_ids)}")
    print(f"Recoveries via motion/IoU re-match: {lost_kalman_recoveries}")
    print(f"Recoveries via ReID (appearance) match: {reid_recoveries}")
    print(
        f"Embeddings computed (full + torso crops): {embed_stats.get('crops_embedded', 0)}, "
        f"detections skipped at image edge: {embed_stats.get('crops_skipped_edge', 0)}"
    )
    if tracker.assignment_stats is not None:
        st = tracker.assignment_stats
        for stage in ("stage1", "reconcile", "stage2"):
            print(
                f"Assignment [{stage}]: frames with candidate pairs {st.get(f'{stage}_calls', 0)}, "
                f"with a conflict {st.get(f'{stage}_conflicts', 0)}, "
                f"greedy != optimal {st.get(f'{stage}_differ', 0)}"
            )
        print(f"Stage-1 solver time ({args.assignment}): {st.get('stage1_solve_s', 0.0) * 1e3:.1f} ms total")
    by_kind: dict[str, int] = {}
    for ev in all_vetoes:
        by_kind[ev["kind"]] = by_kind.get(ev["kind"], 0) + 1
    print(f"Bystander vetoes (pair-frames rejected that would otherwise have passed threshold): {by_kind}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
