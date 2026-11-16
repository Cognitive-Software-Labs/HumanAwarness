"""Shrinks finished run folders without deleting anything
(docs/context/run_recorder.md, "Compressing old runs"). No ROS imports.

- camera.mkv / annotated.mkv are re-encoded to H.264 (CRF 23 by default),
  keeping their names, so every tool opens them exactly as before.
  On the labeled phone clip: 2.6x smaller, and re-running the pipeline on
  the compressed video gave the same is_looking as on the original video on
  96.0% of frames, against 96.3% for the uncompressed recording itself.
- predictions.jsonl and rosout.log are gzipped (predictions: 7.5x smaller).
  The review and scoring tools read the .gz directly; `zless` reads the log.
- markers.jsonl, meta.json, params.yaml and labels stay as plain text.

Every step writes to a temporary file, checks it (same number of video
frames; byte-identical text after decompressing), and only then replaces
the original. Finished steps are recorded in meta.json, so a step is never
done twice (a second H.264 pass would lose quality again) and an
interrupted run carries on where it stopped. A video that fails (e.g.
damaged by a crash) is kept as it is, the reason goes in meta.json, and it
isn't retried on every later run.
"""

import datetime
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

LOCK_FILE = ".recording"  # present while a recorder is writing the folder; holds its pid
VIDEO_FILES = ("camera.mkv", "annotated.mkv")
TEXT_FILES = ("predictions.jsonl", "rosout.log")
_TMP_TAG = ".compressing"


def _read_meta(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "meta.json").read_text())
    except (OSError, ValueError):
        return {}


def _write_meta(run_dir: Path, meta: dict) -> None:
    tmp = run_dir / f"meta.json{_TMP_TAG}"
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp, run_dir / "meta.json")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_being_recorded(run_dir: Path) -> bool:
    lock = run_dir / LOCK_FILE
    if not lock.exists():
        return False
    try:
        return _pid_alive(int(lock.read_text().strip()))
    except (OSError, ValueError):
        return False  # unreadable lock: a crashed recorder's leftover


def runs_to_compress(runs_dir: Path, exclude: Path | None = None) -> list[Path]:
    """Finished run folders (a meta.json, no live recorder) not fully
    compressed yet, oldest first. A crashed run counts as finished."""
    if not runs_dir.is_dir():
        return []
    found = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if exclude is not None and run_dir.resolve() == exclude.resolve():
            continue
        if not (run_dir / "meta.json").exists() or is_being_recorded(run_dir):
            continue
        if not _read_meta(run_dir).get("compressed", {}).get("complete"):
            found.append(run_dir)
    return found


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def count_video_frames(path: Path) -> int | None:
    """Frames in the file, counted from its packets (no decoding). Works on
    an .mkv whose recorder crashed before closing it."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=600,
        ).stdout.strip()
        return int(out)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def compress_video(path: Path, crf: int = 23, threads: int = 2, low_priority: bool = True, on_process=None) -> None:
    """Re-encodes `path` to H.264 in place. `threads` and `low_priority`
    keep it from slowing a live pipeline running at the same time.
    `on_process(popen)` gets the ffmpeg process, so a caller can stop it."""
    tmp = path.with_name(path.stem + _TMP_TAG + path.suffix)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
           "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast", "-pix_fmt", "yuv420p",
           "-threads", str(threads), str(tmp)]
    if low_priority:
        cmd = ["nice", "-n", "19", *cmd]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if on_process is not None:
        on_process(proc)
    _, err = proc.communicate()
    try:
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {path.name}: {err.strip()[-300:]}")
        before, after = count_video_frames(path), count_video_frames(tmp)
        if before is None or before != after:
            raise RuntimeError(f"{path.name}: frame count changed ({before} -> {after}), original kept")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _sha256(chunks) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


def _file_chunks(f, size=1 << 20):
    while chunk := f.read(size):
        yield chunk


def gzip_file(path: Path) -> Path:
    """path -> path.gz, checked byte for byte before the original is removed."""
    gz = path.with_name(path.name + ".gz")
    tmp = path.with_name(path.name + ".gz" + _TMP_TAG)
    try:
        with open(path, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        with open(path, "rb") as a, gzip.open(tmp, "rb") as b:
            if _sha256(_file_chunks(a)) != _sha256(_file_chunks(b)):
                raise RuntimeError(f"{path.name}: gzip check failed, original kept")
        os.replace(tmp, gz)
    finally:
        tmp.unlink(missing_ok=True)
    path.unlink()
    return gz


def _folder_bytes(run_dir: Path) -> int:
    return sum(p.stat().st_size for p in run_dir.iterdir() if p.is_file())


def compress_run(run_dir: Path, crf: int = 23, threads: int = 2, low_priority: bool = True,
                 on_process=None, should_stop=lambda: False) -> dict:
    """Compresses one finished run folder; returns its meta.json "compressed"
    entry. Steps already done (listed there) are skipped. Without ffmpeg the
    videos are left as they are and the text files are still gzipped."""
    for leftover in run_dir.glob(f"*{_TMP_TAG}*"):  # from an interrupted earlier attempt
        leftover.unlink()
    meta = _read_meta(run_dir)
    state = meta.get("compressed", {})
    done = set(state.get("done", []))
    state.setdefault("bytes_before", _folder_bytes(run_dir))
    have_ffmpeg = ffmpeg_available()

    for name in VIDEO_FILES:
        if should_stop():
            break
        path = run_dir / name
        if name in done or not path.exists() or path.stat().st_size == 0:
            continue
        if not have_ffmpeg:
            state["video_skipped"] = "ffmpeg/ffprobe not installed"
            continue
        if name in state.get("failed", {}):
            continue
        try:
            compress_video(path, crf=crf, threads=threads, low_priority=low_priority, on_process=on_process)
        except RuntimeError as e:
            if should_stop():  # ffmpeg was stopped on purpose: just try again next time
                break
            state.setdefault("failed", {})[name] = str(e)
        else:
            done.add(name)
            state["video_crf"] = crf
        state["done"] = sorted(done)
        _write_meta(run_dir, {**meta, "compressed": state})

    for name in TEXT_FILES:
        if should_stop():
            break
        path = run_dir / name
        if name in done or not path.exists():
            continue
        gzip_file(path)
        done.add(name)
        state.update(done=sorted(done))
        _write_meta(run_dir, {**meta, "compressed": state})

    failed = state.get("failed", {})
    videos_left = [
        n for n in VIDEO_FILES
        if have_ffmpeg and (run_dir / n).exists() and (run_dir / n).stat().st_size > 0
        and n not in done and n not in failed
    ]
    texts_left = [n for n in TEXT_FILES if (run_dir / n).exists()]
    if not should_stop() and not videos_left and not texts_left:
        state.update(
            complete=True,
            bytes_after=_folder_bytes(run_dir),
            at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        _write_meta(run_dir, {**meta, "compressed": state})
    return state


def main(argv=None) -> int:
    """`ros2 run human_run_recorder compress_runs [runs_dir] [--crf N]`:
    compress every finished run now (e.g. after a session, so the last run
    doesn't wait for the next one to start)."""
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("runs_dir", nargs="?", default="~/human_awareness_runs")
    parser.add_argument("--crf", type=int, default=23, help="H.264 quality: lower = better and bigger (default 23)")
    parser.add_argument("--threads", type=int, default=0, help="ffmpeg threads (0 = all cores)")
    args = parser.parse_args(argv)

    runs = runs_to_compress(Path(args.runs_dir).expanduser())
    if not runs:
        print("Nothing to compress.")
        return 0
    if not ffmpeg_available():
        print("ffmpeg/ffprobe not found: only the text files will be compressed (sudo apt install ffmpeg).")
    for run_dir in runs:
        print(f"{run_dir.name}: compressing...", flush=True)
        state = compress_run(run_dir, crf=args.crf, threads=args.threads, low_priority=False)
        for name, reason in state.get("failed", {}).items():
            print(f"  {name} left as it was: {reason}")
        if state.get("complete"):
            print(f"  {state['bytes_before'] / 1e6:.1f} MB -> {state['bytes_after'] / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
