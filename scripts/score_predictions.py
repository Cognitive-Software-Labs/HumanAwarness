#!/usr/bin/env python3
"""Scores exported predictions (scripts/test_tracking_head_pose_and_gaze_
on_video.py --export-jsonl) against manual ground truth
(scripts/label_review_tool.py's labels.json) - a gaze/awareness confusion
matrix (precision/recall/false-positive rate), ReID error counts (ID
switches, false merges), and the exact frame ranges of every disagreement
so you can jump straight to them in the review tool instead of re-scrubbing
a whole clip.

Usage:
    source .venv/bin/activate
    python3 scripts/score_predictions.py test_output/video1.jsonl test_output/video2.jsonl
    # labels default to <predictions>.labels.json next to each; override with
    # --labels one/path/per/predictions/file, same order
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _label_io import GAZE_LOOKING, GAZE_NOT_LOOKING, Labels, default_labels_path, load_predictions


def frame_ranges(frames: list) -> list:
    """Collapses a sorted list of frame numbers into [start, end] runs, for
    compact "here's where it disagreed" reporting."""
    if not frames:
        return []
    frames = sorted(frames)
    ranges = [[frames[0], frames[0]]]
    for f in frames[1:]:
        if f <= ranges[-1][1] + 1:
            ranges[-1][1] = f
        else:
            ranges.append([f, f])
    return ranges


def score_gaze(predictions_by_frame: dict, labels: Labels) -> dict:
    tp = fp = fn = tn = 0
    disagreement_frames = defaultdict(list)  # track_id -> [frame, ...]
    for frame_idx, rows in predictions_by_frame.items():
        for row in rows:
            gt = labels.gaze_label_for(row["track_id"], frame_idx)
            if gt not in (GAZE_LOOKING, GAZE_NOT_LOOKING):
                continue  # unlabeled or explicitly "unsure" - excluded from scoring
            predicted = row["is_looking"]
            gt_looking = gt == GAZE_LOOKING
            if predicted and gt_looking:
                tp += 1
            elif predicted and not gt_looking:
                fp += 1
                disagreement_frames[row["track_id"]].append(frame_idx)
            elif not predicted and gt_looking:
                fn += 1
                disagreement_frames[row["track_id"]].append(frame_idx)
            else:
                tn += 1
    total_labeled = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "total_labeled": total_labeled,
        "precision": precision, "recall": recall, "false_positive_rate": fpr,
        "disagreement_ranges": {tid: frame_ranges(fs) for tid, fs in disagreement_frames.items()},
    }


def score_identity(labels: Labels) -> dict:
    person_to_tracks = defaultdict(set)
    track_to_persons = defaultdict(set)
    for seg in labels.identity_segments:
        person_to_tracks[seg.person_id].add(seg.track_id)
        track_to_persons[seg.track_id].add(seg.person_id)

    id_switches = {pid: sorted(tids) for pid, tids in person_to_tracks.items() if len(tids) > 1}
    false_merges = {}
    for tid, pids in track_to_persons.items():
        if len(pids) > 1:
            segs = sorted((s for s in labels.identity_segments if s.track_id == tid), key=lambda s: s.start_frame)
            false_merges[tid] = [{"person_id": s.person_id, "start_frame": s.start_frame, "end_frame": s.end_frame}
                                  for s in segs]
    return {
        "num_tracks": len(track_to_persons),
        "num_distinct_people": len(person_to_tracks),
        "id_switches": id_switches,
        "false_merges": false_merges,
    }


def print_report(name: str, gaze: dict, identity: dict) -> None:
    print(f"\n=== {name} ===")
    print("-- Gaze / awareness --")
    if gaze["total_labeled"] == 0:
        print("  No labeled frames yet (use label_review_tool.py's l/k/u keys).")
    else:
        print(f"  labeled frames: {gaze['total_labeled']}  "
              f"(TP={gaze['tp']} FP={gaze['fp']} FN={gaze['fn']} TN={gaze['tn']})")
        p, r, fpr = gaze["precision"], gaze["recall"], gaze["false_positive_rate"]
        print(f"  precision={p:.3f} recall={r:.3f} false_positive_rate={fpr:.3f}"
              if p is not None and r is not None and fpr is not None else "  precision/recall: n/a")
        if gaze["disagreement_ranges"]:
            print("  disagreement frame ranges (predicted != ground truth), by track_id:")
            for tid, ranges in gaze["disagreement_ranges"].items():
                ranges_str = ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)
                print(f"    track {tid}: {ranges_str}")
    print("-- Identity / ReID --")
    print(f"  track_ids seen: {identity['num_tracks']}  distinct real people (by person_id): "
          f"{identity['num_distinct_people']}")
    if identity["id_switches"]:
        print("  ID switches (one real person split across multiple track_ids):")
        for pid, tids in identity["id_switches"].items():
            print(f"    {pid}: tracks {tids}")
    else:
        print("  no ID switches labeled")
    if identity["false_merges"]:
        print("  false merges (one track_id covering multiple real people):")
        for tid, segs in identity["false_merges"].items():
            seg_str = ", ".join(f"{s['person_id']}@[{s['start_frame']}-{s['end_frame']}]" for s in segs)
            print(f"    track {tid}: {seg_str}")
    else:
        print("  no false merges labeled")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", nargs="+", help="one or more --export-jsonl files to score")
    parser.add_argument("--labels", nargs="*", default=None,
                         help="labels JSON per predictions file, same order (default: <predictions>.labels.json)")
    parser.add_argument("--report-json", default=None, help="also write the full report as JSON to this path")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.predictions):
        parser.error("--labels must have one path per predictions file if given")

    agg_gaze = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    agg_switches = 0
    agg_merges = 0
    full_report = {}

    for i, pred_path_str in enumerate(args.predictions):
        pred_path = Path(pred_path_str)
        labels_path = Path(args.labels[i]) if args.labels else default_labels_path(pred_path)
        if not labels_path.exists():
            print(f"\n=== {pred_path.name} ===\n  No labels file at {labels_path} yet - skipping.")
            continue

        predictions_by_frame = load_predictions(pred_path)
        labels = Labels.load_or_init(labels_path, predictions_by_frame, video=str(pred_path))

        gaze = score_gaze(predictions_by_frame, labels)
        identity = score_identity(labels)
        print_report(pred_path.name, gaze, identity)
        full_report[pred_path.name] = {"gaze": gaze, "identity": identity}

        for k in ("tp", "fp", "fn", "tn"):
            agg_gaze[k] += gaze[k]
        agg_switches += len(identity["id_switches"])
        agg_merges += len(identity["false_merges"])

    if len(args.predictions) > 1:
        tp, fp, fn, tn = agg_gaze["tp"], agg_gaze["fp"], agg_gaze["fn"], agg_gaze["tn"]
        total = tp + fp + fn + tn
        print(f"\n=== TOTAL across {len(args.predictions)} videos ===")
        if total:
            precision = tp / (tp + fp) if (tp + fp) else None
            recall = tp / (tp + fn) if (tp + fn) else None
            print(f"  labeled frames: {total}  (TP={tp} FP={fp} FN={fn} TN={tn})")
            print(f"  precision={precision:.3f} recall={recall:.3f}"
                  if precision is not None and recall is not None else "  precision/recall: n/a")
        print(f"  ID switches: {agg_switches}  false merges: {agg_merges}")

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(full_report, indent=2))
        print(f"\nFull report -> {args.report_json}")


if __name__ == "__main__":
    main()
