"""Writes one recorded run to its folder (docs/context/run_recorder.md). No
ROS imports: detections are read by attribute, so tests use SimpleNamespace.

A run folder holds:
- camera.mkv: the camera frames the pipeline actually produced output for,
  in order. Frame N here is "frame" N in predictions.jsonl, so the review
  tool and the video scripts can be pointed straight at it.
- annotated.mkv: the same frames with the overlay and per-person debug
  numbers drawn on.
- predictions.jsonl: one row per person per frame, the video script's
  --export-jsonl format plus the extra fields only the live run has.
- markers.jsonl: problem markers dropped live (`ros2 run human_run_recorder mark`).
- rosout.log: every node's log lines from /rosout.
- params.yaml: every running node's parameters, read a few seconds in.
- meta.json: topics, git commit, frame count, fps, start/end time.
- .recording: the recorder's pid while it is writing (removed on close), so
  compress.py never touches a run that is still being recorded.

Videos are Matroska (.mkv), not mp4: an mp4 killed before it's closed has no
index and can't be opened at all, while a killed .mkv loses only the last
frames still in the encoder.
"""

import json
import math
import os
import subprocess
from pathlib import Path

import cv2

from human_run_recorder.compress import LOCK_FILE
from human_visualization.adapters import LOST_GALLERY, is_looking

_NO_DECISION = ("", "none")
_FOURCC = "mp4v"
_MIN_FPS, _MAX_FPS = 1.0, 120.0
_DEBUG_COLOR = (255, 255, 255)
_DEBUG_BG_COLOR = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _r(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def detection_to_row(det, frame: int, timestamp_s: float, stamp_ns: int, aware_threshold_deg: float) -> dict | None:
    """One predictions.jsonl row. Same keys and None conventions as the
    video script's export (so label_review_tool / score_predictions read it
    unchanged) plus the live-only diagnostics. None for a LOST_GALLERY
    track: it isn't in the frame, its bbox is stale."""
    if det.state == LOST_GALLERY:
        return None
    pose = bool(det.head_pose_valid)
    decided = pose and det.distance_method not in _NO_DECISION
    x1, y1 = det.bbox_x, det.bbox_y
    return {
        "frame": frame,
        "timestamp_s": _r(timestamp_s),
        "stamp_ns": stamp_ns,
        "track_id": int(det.track_id),
        "state": det.state,
        "confidence": _r(det.confidence),
        "bbox": [round(x1), round(y1), round(x1 + det.bbox_width), round(y1 + det.bbox_height)],
        "head_pose_valid": pose,
        "head_yaw_deg": _r(det.head_yaw_deg, 2) if pose else None,
        "head_pitch_deg": _r(det.head_pitch_deg, 2) if pose else None,
        "head_roll_deg": _r(det.head_roll_deg, 2) if pose else None,
        "bearing_deg": _r(det.bearing_deg, 2) if pose else None,
        "gaze_robot_deg": _r(det.gaze_robot_deg, 2) if pose else None,
        "gaze_source": det.gaze_source or None,
        "gaze_gate_reason": det.gaze_gate_reason or None,
        "iris_valid": bool(det.iris_valid),
        "iris_fixating": bool(det.iris_fixating),
        "iris_yaw_deg": _r(det.iris_yaw_deg, 2) if det.iris_fixating else None,
        "iris_offset": _r(det.iris_offset) if det.iris_offset >= 0.0 else None,
        "eyes_averted": bool(det.eyes_averted),
        "distance_m": _r(det.distance_m, 3) if decided else None,
        "distance_method": det.distance_method or None,
        "theta_to_robot_deg": _r(det.theta_to_robot_deg, 2) if decided else None,
        "sees_robot": bool(det.sees_robot),
        "direct_contact": bool(det.direct_contact),
        "is_looking": is_looking(det, aware_threshold_deg),
        "awareness_value": _r(det.awareness_value) if det.track_id >= 0 else None,
        "awareness_confidence": _r(det.awareness_confidence) if det.track_id >= 0 else None,
    }


def debug_lines(row: dict) -> list[str]:
    """The per-person numbers drawn under the bbox on annotated.mkv."""
    if not row["head_pose_valid"]:
        return []
    lines = [f"yaw {row['head_yaw_deg']:.0f} pitch {row['head_pitch_deg']:.0f} src {row['gaze_source']}"]
    if row["theta_to_robot_deg"] is not None:
        lines.append(f"theta {row['theta_to_robot_deg']:.1f} dist {row['distance_m']:.2f}m {row['distance_method']}")
    iris = f"iris off {row['iris_offset']:.2f}" if row["iris_offset"] is not None else "iris -"
    if row["eyes_averted"]:
        iris += " AVERTED"
    if row["gaze_gate_reason"]:
        iris += f" ({row['gaze_gate_reason']})"
    lines.append(iris)
    return lines


def draw_debug_lines(image, bbox: list[int], lines: list[str]) -> None:
    """Small text block just below the bbox (inside the frame)."""
    x1, _y1, _x2, y2 = bbox
    h, w = image.shape[:2]
    y = min(max(y2 + 18, 18), h - 4 - 18 * max(len(lines) - 1, 0))
    for line in lines:
        (text_w, text_h), baseline = cv2.getTextSize(line, _FONT, 0.5, 1)
        x = max(0, min(int(x1), w - text_w - 4))
        cv2.rectangle(image, (x, y - text_h - 3), (x + text_w + 4, y + baseline), _DEBUG_BG_COLOR, -1)
        cv2.putText(image, line, (x + 2, y), _FONT, 0.5, _DEBUG_COLOR, 1)
        y += 18


def estimate_fps(stamps_s: list[float], fallback_fps: float) -> float:
    """Mean rate over the stamps, clamped to a sane range; `fallback_fps`
    if there are fewer than two or they don't advance (unstamped source)."""
    if len(stamps_s) < 2 or stamps_s[-1] <= stamps_s[0]:
        return fallback_fps
    fps = (len(stamps_s) - 1) / (stamps_s[-1] - stamps_s[0])
    return min(max(fps, _MIN_FPS), _MAX_FPS)


class VideoSink:
    """An .mkv writer whose fps is only known after `probe_frames` frames:
    the pipeline outputs fewer frames than the camera sends, at a rate that
    depends on the machine. Frames are held until then, then written."""

    def __init__(self, path: Path, probe_frames: int, fallback_fps: float):
        self.path = path
        self.probe_frames = max(probe_frames, 2)
        self.fallback_fps = fallback_fps
        self.fps: float | None = None
        self._pending: list[tuple[object, float]] = []
        self._writer = None

    def write(self, frame, stamp_s: float) -> None:
        if self._writer is not None:
            self._writer.write(frame)
            return
        self._pending.append((frame, stamp_s))
        if len(self._pending) >= self.probe_frames:
            self._open_and_flush()

    def _open_and_flush(self) -> None:
        self.fps = estimate_fps([s for _, s in self._pending], self.fallback_fps)
        h, w = self._pending[0][0].shape[:2]
        self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*_FOURCC), self.fps, (w, h))
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video writer for {self.path}")
        for frame, _ in self._pending:
            self._writer.write(frame)
        self._pending.clear()

    def close(self) -> None:
        if self._writer is None and self._pending:
            self._open_and_flush()
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def git_info(start: Path) -> dict:
    """Commit and dirty-file count of the repo `start` lives in (the
    workspace, for an installed package), so a run says which code made it.
    Empty if there's no repo or no git."""
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            try:
                commit = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
                ).stdout.strip()
                dirty = subprocess.run(
                    ["git", "-C", str(parent), "status", "--porcelain"], capture_output=True, text=True, timeout=5
                ).stdout.splitlines()
                branch = subprocess.run(
                    ["git", "-C", str(parent), "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return {}
            return {"repo": str(parent), "commit": commit, "branch": branch, "uncommitted_files": len(dirty)}
    return {}


class RunWriter:
    def __init__(
        self,
        run_dir: Path,
        record_annotated: bool = True,
        probe_frames: int = 15,
        fallback_fps: float = 15.0,
        aware_threshold_deg: float = 45.0,
    ):
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / LOCK_FILE).write_text(f"{os.getpid()}\n")
        self.aware_threshold_deg = aware_threshold_deg
        self.camera = VideoSink(run_dir / "camera.mkv", probe_frames, fallback_fps)
        self.annotated = VideoSink(run_dir / "annotated.mkv", probe_frames, fallback_fps) if record_annotated else None
        # Line-buffered: a crash keeps everything up to the last full line.
        self._predictions = open(run_dir / "predictions.jsonl", "w", buffering=1)
        self._markers = open(run_dir / "markers.jsonl", "w", buffering=1)
        self._log = open(run_dir / "rosout.log", "w", buffering=1)
        self.frames_written = 0
        self.markers_written = 0
        self._first_stamp_ns: int | None = None
        self.last_stamp_ns: int | None = None

    def add_frame(self, camera_bgr, detections, stamp_ns: int, draw_overlay=None) -> tuple[int, list[dict]]:
        """Writes one processed frame and its people. `draw_overlay(image,
        detections)` draws the normal overlay onto the annotated copy (passed
        in so this module needs no drawing choices of its own). Returns the
        frame index and the rows written."""
        frame = self.frames_written
        if self._first_stamp_ns is None:
            self._first_stamp_ns = stamp_ns
        self.last_stamp_ns = stamp_ns
        timestamp_s = (stamp_ns - self._first_stamp_ns) * 1e-9
        rows = []
        for det in detections:
            row = detection_to_row(det, frame, timestamp_s, stamp_ns, self.aware_threshold_deg)
            if row is not None:
                rows.append(row)
                self._predictions.write(json.dumps(row) + "\n")
        self.camera.write(camera_bgr, stamp_ns * 1e-9)
        if self.annotated is not None:
            annotated = camera_bgr.copy()
            if draw_overlay is not None:
                draw_overlay(annotated, detections)
            for row in rows:
                draw_debug_lines(annotated, row["bbox"], debug_lines(row))
            self.annotated.write(annotated, stamp_ns * 1e-9)
        self.frames_written += 1
        return frame, rows

    def add_marker(self, note: str, wall_time_iso: str) -> dict:
        """Marks the newest frame written: what was on screen (give or take
        the pipeline delay) when the problem was seen."""
        marker = {
            "frame": max(self.frames_written - 1, 0),
            "stamp_ns": self.last_stamp_ns,
            "wall_time": wall_time_iso,
            "note": note,
        }
        self._markers.write(json.dumps(marker) + "\n")
        self.markers_written += 1
        return marker

    def add_log_line(self, line: str) -> None:
        self._log.write(line.rstrip("\n") + "\n")

    def write_json(self, name: str, data: dict) -> None:
        (self.run_dir / name).write_text(json.dumps(data, indent=2) + "\n")

    def summary(self) -> dict:
        duration_s = (
            (self.last_stamp_ns - self._first_stamp_ns) * 1e-9 if self._first_stamp_ns is not None else 0.0
        )
        return {
            "frames": self.frames_written,
            "markers": self.markers_written,
            "duration_s": round(duration_s, 3),
            "video_fps": self.camera.fps,
            "mean_processed_fps": (
                round((self.frames_written - 1) / duration_s, 2)
                if duration_s > 0 and math.isfinite(duration_s) else None
            ),
        }

    def close(self) -> None:
        self.camera.close()
        if self.annotated is not None:
            self.annotated.close()
        for f in (self._predictions, self._markers, self._log):
            f.close()
        (self.run_dir / LOCK_FILE).unlink(missing_ok=True)
