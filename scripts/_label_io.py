"""Shared data model for manual ground-truth labels, used by both
label_review_tool.py (writes labels interactively) and score_predictions.py
(reads them to score predictions). Kept separate from either script so the
on-disk format has exactly one place it's defined.

A labels file (<video_stem>.labels.json next to the predictions .jsonl) has
two independent label sets, both expressed as frame intervals rather than
one entry per frame - the whole point is that you mark *transitions*, not
every frame:

- `identity_segments`: ground truth for ReID. One entry per contiguous run
  of frames in which a given track_id correctly refers to one real person,
  tagged with a `person_id` you choose. Two track_ids sharing a person_id
  is a confirmed "same person" (the tracker split one real person into two
  IDs - an ID switch). A single track_id covering two person_ids in two
  different segments is a confirmed false merge (the tracker fused two real
  people into one ID) - you mark this by splitting the track's segment at
  the frame where the identity actually changed.
- `gaze_labels`: ground truth for "is this person looking at the robot",
  keyed by track_id, each a list of {start_frame, end_frame, label} runs
  (label is "looking" / "not_looking" / "unsure" - the last for frames
  where it genuinely can't be judged by eye, e.g. face not visible at all,
  so they're excluded from scoring rather than counted as a miss either way).
"""

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

GAZE_LOOKING = "looking"
GAZE_NOT_LOOKING = "not_looking"
GAZE_UNSURE = "unsure"


def resolve_predictions_path(jsonl_path) -> Path:
    """A recorded run's predictions.jsonl becomes predictions.jsonl.gz once
    it's compressed (human_run_recorder/compress.py): either name finds
    whichever exists."""
    path = Path(jsonl_path)
    if path.exists():
        return path
    for other in (path.with_name(path.name + ".gz"), path.with_suffix("") if path.suffix == ".gz" else None):
        if other is not None and other.exists():
            return other
    return path


def default_labels_path(predictions_path) -> Path:
    """<predictions>.labels.json, the same file whether the predictions are
    gzipped or not, so labels made before compression are still found."""
    path = Path(predictions_path)
    if path.suffix == ".gz":
        path = path.with_suffix("")
    return path.with_suffix(".labels.json")


def load_predictions(jsonl_path) -> dict:
    """Returns {frame_idx: [row, ...]} - every exported track-frame record,
    grouped by frame. Rows keep every field test_tracking_head_pose_and_
    gaze_on_video.py's --export-jsonl writes (bbox, track_id/state, head
    pose, fused gaze, geometric decision, is_looking). Reads .jsonl or
    .jsonl.gz."""
    by_frame: dict = {}
    path = resolve_predictions_path(jsonl_path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_frame.setdefault(row["frame"], []).append(row)
    return by_frame


@dataclass
class IdentitySegment:
    track_id: int
    start_frame: int
    end_frame: int  # inclusive
    person_id: str


@dataclass
class GazeInterval:
    start_frame: int
    end_frame: int  # inclusive
    label: str


@dataclass
class Labels:
    video: str
    identity_segments: list = field(default_factory=list)  # list[IdentitySegment]
    gaze_labels: dict = field(default_factory=dict)  # track_id (str) -> list[GazeInterval]

    @classmethod
    def load_or_init(cls, path: Path, predictions_by_frame: dict, video: str) -> "Labels":
        if path.exists():
            raw = json.loads(path.read_text())
            labels = cls(
                video=raw["video"],
                identity_segments=[IdentitySegment(**s) for s in raw.get("identity_segments", [])],
                gaze_labels={
                    tid: [GazeInterval(**iv) for iv in intervals]
                    for tid, intervals in raw.get("gaze_labels", {}).items()
                },
            )
            labels._fill_missing_identity_segments(predictions_by_frame)
            return labels
        labels = cls(video=video)
        labels._fill_missing_identity_segments(predictions_by_frame)
        return labels

    def _fill_missing_identity_segments(self, predictions_by_frame: dict) -> None:
        """Auto-seed one identity_segment per track_id spanning its full
        observed lifetime, defaulting person_id to f"P{track_id}" - the
        "tracker's own IDs are right until proven otherwise" starting
        point. Only adds tracks with no segment at all yet, so it never
        overwrites a split/merge you've already made."""
        first_seen: dict = {}
        last_seen: dict = {}
        for frame_idx, rows in predictions_by_frame.items():
            for row in rows:
                tid = row["track_id"]
                first_seen[tid] = min(frame_idx, first_seen.get(tid, frame_idx))
                last_seen[tid] = max(frame_idx, last_seen.get(tid, frame_idx))
        covered_tracks = {s.track_id for s in self.identity_segments}
        for tid in sorted(first_seen):
            if tid in covered_tracks:
                continue
            self.identity_segments.append(
                IdentitySegment(track_id=tid, start_frame=first_seen[tid], end_frame=last_seen[tid],
                                 person_id=f"P{tid}")
            )

    def segment_for(self, track_id: int, frame: int) -> IdentitySegment | None:
        for seg in self.identity_segments:
            if seg.track_id == track_id and seg.start_frame <= frame <= seg.end_frame:
                return seg
        return None

    def person_id_for(self, track_id: int, frame: int) -> str:
        seg = self.segment_for(track_id, frame)
        return seg.person_id if seg is not None else f"P{track_id}"

    def split_segment(self, track_id: int, split_frame: int, new_person_id: str) -> bool:
        """Splits whichever segment currently covers (track_id, split_frame)
        into [old_start, split_frame-1] (keeps its person_id) and
        [split_frame, old_end] (gets new_person_id). Returns False if
        split_frame isn't strictly inside an existing segment (nothing to
        split, e.g. split_frame is already a segment's start)."""
        seg = self.segment_for(track_id, split_frame)
        if seg is None or split_frame <= seg.start_frame:
            return False
        old_end = seg.end_frame
        seg.end_frame = split_frame - 1
        self.identity_segments.append(
            IdentitySegment(track_id=track_id, start_frame=split_frame, end_frame=old_end,
                             person_id=new_person_id)
        )
        return True

    def set_person_id(self, track_id: int, frame: int, new_person_id: str) -> None:
        """Renames the person_id of whichever segment covers (track_id,
        frame) - the "these two track_ids are actually the same person"
        correction (set both to the same person_id)."""
        seg = self.segment_for(track_id, frame)
        if seg is not None:
            seg.person_id = new_person_id

    def set_gaze_label(self, track_id: int, frame: int, label: str) -> None:
        """Extends the open interval for this track if its label already
        matches and it's contiguous with `frame`; otherwise closes the
        previous interval (if any) and opens a new one. This is what lets
        you label a whole look/away run with one keypress per transition
        instead of one per frame."""
        key = str(track_id)
        intervals = self.gaze_labels.setdefault(key, [])
        intervals.sort(key=lambda iv: iv.start_frame)
        for iv in intervals:
            if iv.start_frame <= frame <= iv.end_frame and iv.label == label:
                return  # already labeled this, nothing to do
        for iv in intervals:
            if iv.start_frame <= frame <= iv.end_frame:
                # Re-labeling a frame inside an existing, differently-labeled
                # interval: shrink/split it rather than silently overlapping.
                if frame == iv.start_frame == iv.end_frame:
                    intervals.remove(iv)
                elif frame == iv.start_frame:
                    iv.start_frame = frame + 1
                elif frame == iv.end_frame:
                    iv.end_frame = frame - 1
                else:
                    tail_start = frame + 1
                    tail_end = iv.end_frame
                    iv.end_frame = frame - 1
                    intervals.append(GazeInterval(tail_start, tail_end, iv.label))
                break
        for iv in intervals:
            if iv.label == label and (iv.end_frame == frame - 1 or iv.start_frame == frame + 1):
                iv.start_frame = min(iv.start_frame, frame)
                iv.end_frame = max(iv.end_frame, frame)
                self._merge_adjacent(key)
                return
        intervals.append(GazeInterval(frame, frame, label))
        self._merge_adjacent(key)

    def _merge_adjacent(self, key: str) -> None:
        intervals = sorted(self.gaze_labels[key], key=lambda iv: iv.start_frame)
        merged: list = []
        for iv in intervals:
            if merged and merged[-1].label == iv.label and merged[-1].end_frame + 1 >= iv.start_frame:
                merged[-1].end_frame = max(merged[-1].end_frame, iv.end_frame)
            else:
                merged.append(iv)
        self.gaze_labels[key] = merged

    def gaze_label_for(self, track_id: int, frame: int) -> str | None:
        for iv in self.gaze_labels.get(str(track_id), []):
            if iv.start_frame <= frame <= iv.end_frame:
                return iv.label
        return None

    def save(self, path: Path) -> None:
        payload = {
            "video": self.video,
            "identity_segments": [asdict(s) for s in self.identity_segments],
            "gaze_labels": {
                tid: [asdict(iv) for iv in intervals] for tid, intervals in self.gaze_labels.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2))
