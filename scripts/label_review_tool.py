#!/usr/bin/env python3
"""Frame-by-frame manual review tool: step through a video alongside the
per-track predictions scripts/test_tracking_head_pose_and_gaze_on_video.py
exported with --export-jsonl, and label ground truth for (a) whether each
person is actually looking at the robot and (b) whether the tracker's
track_ids actually correspond to real, persistent identities. Labels are
saved as frame intervals (not one entry per frame) next to the predictions
file - see _label_io.py for the on-disk format and why.

Usage:
    source .venv/bin/activate
    python3 scripts/label_review_tool.py path/to/video.mp4 \\
        --predictions test_output/video.jsonl

Keys:
    Right/d / Left/a   step 1 frame
    ]/[                jump 10 frames
    n / N              jump to next/previous frame where a labeled track's
                       gaze label disagrees with its prediction
    e / E              jump to next/previous "track event" frame (a track_id
                       entering or leaving the frame)
    click a box        select that track (required for the labels below)
    l / k / u          label selected track at this frame: looking /
                       not_looking / unsure (extends the open interval)
    i                  rename the selected track's person_id (merge with
                       another track_id by giving it the same person_id -
                       use this when the tracker gave the same real person
                       two different IDs, i.e. an ID switch)
    x                  split the selected track's identity segment at this
                       frame into a new person_id (use this when the
                       tracker fused two different real people under one
                       track_id, i.e. a false merge)
    s                  save labels now (also auto-saved on any exit: q/Esc,
                       closing the window, or the process being terminated)
    h                  toggle help overlay
    q/Esc              save and quit
"""

import argparse
import signal
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _label_io import GAZE_LOOKING, GAZE_NOT_LOOKING, GAZE_UNSURE, Labels, load_predictions
from _video_rotation import ensure_upright

WINDOW = "label_review_tool"
LOOKING_COLOR = (0, 220, 0)
NOT_LOOKING_COLOR = (0, 0, 230)
NO_PREDICTION_COLOR = (160, 160, 160)
SELECTED_COLOR = (0, 255, 255)
LABEL_MARK_COLOR = (255, 255, 255)
JUMP_SMALL = 1
JUMP_BIG = 10

GAZE_KEYS = {ord("l"): GAZE_LOOKING, ord("k"): GAZE_NOT_LOOKING, ord("u"): GAZE_UNSURE}


def track_ids_at(predictions_by_frame: dict, frame: int) -> set:
    return {row["track_id"] for row in predictions_by_frame.get(frame, [])}


def find_next_event_frame(predictions_by_frame: dict, total_frames: int, start: int, step: int) -> int | None:
    """A "track event" frame: the set of track_ids visible differs from the
    previous frame (someone appeared or disappeared) - lets you jump
    straight to track boundaries instead of scrubbing to find them."""
    frame = start
    prev_ids = track_ids_at(predictions_by_frame, frame)
    frame += step
    while 0 <= frame < total_frames:
        ids = track_ids_at(predictions_by_frame, frame)
        if ids != prev_ids:
            return frame
        prev_ids = ids
        frame += step
    return None


def find_next_disagreement_frame(predictions_by_frame: dict, labels: Labels, total_frames: int,
                                  start: int, step: int) -> int | None:
    frame = start + step
    while 0 <= frame < total_frames:
        for row in predictions_by_frame.get(frame, []):
            gt = labels.gaze_label_for(row["track_id"], frame)
            if gt in (GAZE_LOOKING, GAZE_NOT_LOOKING):
                predicted = GAZE_LOOKING if row["is_looking"] else GAZE_NOT_LOOKING
                if predicted != gt:
                    return frame
        frame += step
    return None


def prompt_text(base_frame, prompt: str) -> str | None:
    """Minimal inline text entry: no native text box in a bare OpenCV
    window, so this redraws `base_frame` with the prompt + typed-so-far
    text each keystroke until Enter/Esc. Good enough for the short
    person_id strings this tool ever asks for."""
    typed = ""
    while True:
        img = base_frame.copy()
        h = img.shape[0]
        cv2.rectangle(img, (0, h - 50), (img.shape[1], h), (30, 30, 30), -1)
        cv2.putText(img, f"{prompt}: {typed}_", (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    LABEL_MARK_COLOR, 2)
        cv2.imshow(WINDOW, img)
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):  # Enter
            return typed if typed else None
        if key == 27:  # Esc
            return None
        if key in (8, 127):  # Backspace
            typed = typed[:-1]
        elif 32 <= key < 127:
            typed += chr(key)


def draw_frame(frame, predictions_by_frame: dict, labels: Labels, frame_idx: int,
               selected_track_id: int | None, show_help: bool):
    annotated = frame.copy()
    for row in predictions_by_frame.get(frame_idx, []):
        x1, y1, x2, y2 = row["bbox"]
        predicted_looking = row["is_looking"]
        color = LOOKING_COLOR if predicted_looking else NOT_LOOKING_COLOR
        if not row["head_pose_valid"]:
            color = NO_PREDICTION_COLOR
        thickness = 4 if row["track_id"] == selected_track_id else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        if row["track_id"] == selected_track_id:
            cv2.rectangle(annotated, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), SELECTED_COLOR, 1)

        person_id = labels.person_id_for(row["track_id"], frame_idx)
        pred_text = "LOOKING" if predicted_looking else "not looking"
        gt_label = labels.gaze_label_for(row["track_id"], frame_idx)
        gt_text = f" | GT:{gt_label}" if gt_label else ""
        cv2.putText(annotated, f"id {row['track_id']} ({person_id}) pred:{pred_text}{gt_text}",
                    (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    total = frame.shape[0]
    bar_y = total - 28
    cv2.rectangle(annotated, (0, bar_y), (annotated.shape[1], total), (30, 30, 30), -1)
    sel_text = f"selected: track {selected_track_id}" if selected_track_id is not None else "selected: none"
    cv2.putText(annotated, f"frame {frame_idx} | {sel_text} | 'h' for help",
                (10, total - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, LABEL_MARK_COLOR, 1)

    if show_help:
        for i, line in enumerate(__doc__.split("Keys:")[1].strip("\n").splitlines()):
            cv2.putText(annotated, line, (10, 25 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        LABEL_MARK_COLOR, 1)
    return annotated


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="path to the video file")
    parser.add_argument("--predictions", required=True, help="path to the --export-jsonl output")
    parser.add_argument("--labels", default=None,
                         help="path to the labels JSON (default: <predictions>.labels.json)")
    parser.add_argument(
        "--max-display-width", type=int, default=1000,
        help="shrink the window to fit this width if the video is wider (e.g. a portrait phone "
        "clip's height commonly exceeds a small/remote screen) - clicks are mapped back to the "
        "video's real resolution, so selection stays accurate at any scale",
    )
    parser.add_argument("--max-display-height", type=int, default=680)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    labels_path = Path(args.labels) if args.labels else predictions_path.with_suffix(".labels.json")

    predictions_by_frame = load_predictions(predictions_path)
    labels = Labels.load_or_init(labels_path, predictions_by_frame, video=args.video)

    # Predictions were computed against the rotation-corrected video (every
    # export script runs footage through ensure_upright first - see
    # _video_rotation.py) - opening the raw path here would draw those same
    # bboxes onto still-rotated frames.
    cap = cv2.VideoCapture(str(ensure_upright(Path(args.video))))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Could not read frame count from video.")
        return
    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1
    # A portrait phone clip's height commonly exceeds a small/remote screen
    # (e.g. 1024px tall on a 725px-tall display) - never scale up, only
    # down, and always by the same factor on both axes.
    display_scale = min(1.0, args.max_display_width / frame_w, args.max_display_height / frame_h)

    # Start on the first frame that actually has a prediction, not frame 0 -
    # a clip's first couple of frames commonly have no detection yet, which
    # otherwise looks identical to "the tool is broken, no boxes anywhere".
    start_frame = min(predictions_by_frame.keys()) if predictions_by_frame else 0
    state = {"frame": start_frame, "selected": None}

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # (x, y) arrive in displayed-window pixels, which are smaller than
        # the video's real resolution whenever display_scale < 1 - map back
        # before testing against bboxes in the video's own coordinates.
        orig_x, orig_y = x / display_scale, y / display_scale
        candidates = []
        for row in predictions_by_frame.get(state["frame"], []):
            x1, y1, x2, y2 = row["bbox"]
            if x1 <= orig_x <= x2 and y1 <= orig_y <= y2:
                candidates.append(row)
        if candidates:
            # smallest box wins when boxes overlap - usually the more
            # specific/foreground pick in a crowd
            candidates.sort(key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]))
            state["selected"] = candidates[0]["track_id"]

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    # A plain SIGTERM/SIGHUP would otherwise kill the process without running
    # the finally block below, silently discarding everything since the last 's'.
    def exit_on_signal(_signum, _frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, exit_on_signal)
    signal.signal(signal.SIGHUP, exit_on_signal)

    show_help = False
    current_cap_frame = -1  # force an initial seek
    frame_img = None

    try:
        while True:
            frame_idx = max(0, min(state["frame"], total_frames - 1))
            state["frame"] = frame_idx
            if frame_idx != current_cap_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame_img = cap.read()
                if not ok:
                    break
                current_cap_frame = frame_idx

            annotated = draw_frame(frame_img, predictions_by_frame, labels, frame_idx,
                                    state["selected"], show_help)
            if display_scale < 1.0:
                display_frame = cv2.resize(annotated, (int(frame_w * display_scale), int(frame_h * display_scale)))
            else:
                display_frame = annotated
            cv2.imshow(WINDOW, display_frame)
            # A short timeout, not an indefinite wait: OpenCV only services
            # mouse-click callbacks while waitKey is running, but only
            # redraws on the NEXT loop iteration - an indefinite wait would
            # mean a click's effect (selection highlight) doesn't appear
            # until some unrelated key is pressed afterward.
            key = cv2.waitKeyEx(30)
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # closed with the window's X button
            if key == -1:
                continue
            key_low = key & 0xFF

            if key_low in (27, ord("q")):
                break
            elif key_low == ord("s"):
                labels.save(labels_path)
                print(f"Saved labels -> {labels_path}", flush=True)
            elif key_low == ord("h"):
                show_help = not show_help
            elif key in (65363, 2555904, 83) or key_low == ord("d"):  # right arrow (X11/Windows) or 'd'
                state["frame"] += JUMP_SMALL
            elif key in (65361, 2424832, 81) or key_low == ord("a"):  # left arrow or 'a'
                state["frame"] -= JUMP_SMALL
            elif key_low == ord("]"):
                state["frame"] += JUMP_BIG
            elif key_low == ord("["):
                state["frame"] -= JUMP_BIG
            elif key_low == ord("n"):
                nxt = find_next_disagreement_frame(predictions_by_frame, labels, total_frames, frame_idx, +1)
                if nxt is not None:
                    state["frame"] = nxt
            elif key_low == ord("N"):
                nxt = find_next_disagreement_frame(predictions_by_frame, labels, total_frames, frame_idx, -1)
                if nxt is not None:
                    state["frame"] = nxt
            elif key_low == ord("e"):
                nxt = find_next_event_frame(predictions_by_frame, total_frames, frame_idx, +1)
                if nxt is not None:
                    state["frame"] = nxt
            elif key_low == ord("E"):
                nxt = find_next_event_frame(predictions_by_frame, total_frames, frame_idx, -1)
                if nxt is not None:
                    state["frame"] = nxt
            elif key_low in GAZE_KEYS and state["selected"] is not None:
                labels.set_gaze_label(state["selected"], frame_idx, GAZE_KEYS[key_low])
            elif key_low == ord("i") and state["selected"] is not None:
                new_id = prompt_text(display_frame, "new person_id for this track")
                if new_id:
                    labels.set_person_id(state["selected"], frame_idx, new_id)
            elif key_low == ord("x") and state["selected"] is not None:
                new_id = prompt_text(display_frame, "new person_id from this frame onward")
                if new_id:
                    if not labels.split_segment(state["selected"], frame_idx, new_id):
                        print("Can't split here (frame is already a segment start).")
    finally:
        labels.save(labels_path)
        print(f"Saved labels -> {labels_path}", flush=True)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
