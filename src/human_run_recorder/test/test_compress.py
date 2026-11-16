import gzip
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from human_run_recorder import compress
from human_run_recorder.compress import (
    LOCK_FILE,
    compress_run,
    count_video_frames,
    ffmpeg_available,
    runs_to_compress,
)
from human_run_recorder.run_files import RunWriter

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg/ffprobe not installed")


def det():
    return SimpleNamespace(
        track_id=1, state="ACTIVE", confidence=0.9, bbox_x=10.0, bbox_y=10.0, bbox_width=20.0, bbox_height=40.0,
        head_pose_valid=True, head_yaw_deg=5.0, head_pitch_deg=0.0, head_roll_deg=0.0, bearing_deg=0.0,
        gaze_robot_deg=5.0, gaze_source="head", gaze_gate_reason="eyes too small", iris_valid=False,
        iris_fixating=False, iris_yaw_deg=0.0, iris_offset=0.05, eyes_averted=False, distance_m=1.5,
        distance_method="shoulder_width", theta_to_robot_deg=12.0, sees_robot=True, direct_contact=False,
        awareness_value=0.7, awareness_confidence=0.4,
    )


def make_run(runs_dir, name, frames=20):
    run = RunWriter(runs_dir / name, probe_frames=5)
    rng = np.random.default_rng(0)
    for i in range(frames):
        img = (rng.integers(0, 255, (96, 128, 3)) // 64 * 64).astype(np.uint8)
        run.add_frame(img, [det()], 1_000_000_000 + i * 66_666_667)
    run.add_log_line("1.0 frame=3 WARN [gaze_node] something")
    run.close()
    run.write_json("meta.json", {"run_name": name, **run.summary()})
    return runs_dir / name


@needs_ffmpeg
def test_compressed_run_keeps_every_frame_and_row(tmp_path):
    run_dir = make_run(tmp_path, "a")
    rows_before = (run_dir / "predictions.jsonl").read_text()
    log_before = (run_dir / "rosout.log").read_text()

    state = compress_run(run_dir, threads=1, low_priority=False)

    assert state["complete"] and state["video_crf"] == 23
    for name in ("camera.mkv", "annotated.mkv"):
        assert count_video_frames(run_dir / name) == 20
        cap = cv2.VideoCapture(str(run_dir / name))
        assert cap.get(cv2.CAP_PROP_FOURCC) and cap.read()[0]
    assert not (run_dir / "predictions.jsonl").exists()
    with gzip.open(run_dir / "predictions.jsonl.gz", "rt") as f:
        assert f.read() == rows_before
    with gzip.open(run_dir / "rosout.log.gz", "rt") as f:
        assert f.read() == log_before
    assert (run_dir / "markers.jsonl").exists() and (run_dir / "meta.json").exists()
    assert not list(run_dir.glob("*compressing*"))
    assert runs_to_compress(tmp_path) == []  # never compressed twice


def test_run_being_recorded_is_left_alone(tmp_path):
    live = RunWriter(tmp_path / "live")  # its lock holds this (living) process's pid
    (tmp_path / "live" / "meta.json").write_text("{}")
    assert runs_to_compress(tmp_path) == []
    live.close()
    assert runs_to_compress(tmp_path) == [tmp_path / "live"]


def test_crashed_run_is_compressed(tmp_path):
    run_dir = make_run(tmp_path, "crashed")
    (run_dir / LOCK_FILE).write_text("999999999\n")  # no such process: the recorder died
    assert runs_to_compress(tmp_path) == [run_dir]


def test_exclude_skips_the_new_run(tmp_path):
    make_run(tmp_path, "old")
    new = make_run(tmp_path, "new")
    assert runs_to_compress(tmp_path, exclude=new) == [tmp_path / "old"]


@needs_ffmpeg
def test_interrupted_compression_resumes_without_redoing_steps(tmp_path, monkeypatch):
    run_dir = make_run(tmp_path, "a")
    calls = []
    real = compress.compress_video
    monkeypatch.setattr(compress, "compress_video", lambda path, **kw: (calls.append(path.name), real(path, **kw)))

    stop_after_first = iter([False, True, True, True, True])
    state = compress_run(run_dir, threads=1, low_priority=False, should_stop=lambda: next(stop_after_first))
    assert not state.get("complete") and state["done"] == ["camera.mkv"]
    assert runs_to_compress(tmp_path) == [run_dir]

    state = compress_run(run_dir, threads=1, low_priority=False)
    assert state["complete"]
    assert calls == ["camera.mkv", "annotated.mkv"]  # camera.mkv wasn't re-encoded a second time


@needs_ffmpeg
def test_failed_video_keeps_the_original(tmp_path):
    run_dir = make_run(tmp_path, "a")
    (run_dir / "camera.mkv").write_bytes(b"not a video")
    state = compress_run(run_dir, threads=1, low_priority=False)
    assert (run_dir / "camera.mkv").read_bytes() == b"not a video"
    assert "camera.mkv" in state["failed"] and "annotated.mkv" in state["done"]
    assert state["complete"] and (run_dir / "predictions.jsonl.gz").exists()  # the rest still compressed
    assert not list(run_dir.glob("*compressing*"))
    assert json.loads((run_dir / "meta.json").read_text())["compressed"]["complete"]
    assert runs_to_compress(tmp_path) == []  # not retried on every later run


def test_without_ffmpeg_only_text_is_compressed(tmp_path, monkeypatch):
    run_dir = make_run(tmp_path, "a")
    size = (run_dir / "camera.mkv").stat().st_size
    monkeypatch.setattr(compress, "ffmpeg_available", lambda: False)
    state = compress_run(run_dir)
    assert state["complete"] and "video_skipped" in state
    assert (run_dir / "camera.mkv").stat().st_size == size
    assert (run_dir / "predictions.jsonl.gz").exists()
