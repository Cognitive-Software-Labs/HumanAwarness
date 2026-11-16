# Run recorder (debug logs from live runs)

## What it is

The first live run on the Unitree R1 (2026-11-16) showed two problems
(eyes-only glances called "looking", awareness fading too fast) that could
only be described from memory afterwards: nothing from the run was kept.
`run_recorder_node` (`human_run_recorder` package) saves a live run to its
own folder, so a problem seen on the robot can be replayed, measured and
labeled later with the same tools used on the phone clips.

```bash
ros2 launch human_awareness_bringup full_pipeline.launch.py record:=true viewer:=true
ros2 run human_run_recorder mark        # second terminal: Enter = "problem here"
```

Code: `src/human_run_recorder/human_run_recorder/`:
- `run_files.py`: the folder and everything in it (`RunWriter`, row format,
  debug text, video writing). No ROS imports, unit-tested.
- `run_recorder_node.py`: the ROS wrapper (pairing, /rosout, params, markers).
- `mark.py`: the `mark` command.
- `compress.py`: shrinks finished runs (background, and the
  `compress_runs` command).

## The run folder

`~/human_awareness_runs/<YYYY-mm-dd_HH-MM-SS>/` (`runs_dir:=`, `run_name:=`):

| file | what's in it |
|---|---|
| `camera.mkv` | the camera frames the pipeline produced output for, in order. Frame N is `"frame": N` in `predictions.jsonl` |
| `annotated.mkv` | the same frames with the normal overlay plus, under each box: head yaw/pitch and gaze source; theta to the robot, distance and its method; iris offset, `AVERTED`, and why the iris path did or didn't run |
| `predictions.jsonl` | one row per person per frame: the video script's `--export-jsonl` fields plus `stamp_ns`, `gaze_gate_reason`, `iris_offset`, `eyes_averted`, `iris_valid`, `iris_yaw_deg`, `theta_to_robot_deg`, `distance_method`, `awareness_value`/`_confidence` |
| `markers.jsonl` | problem markers: `frame`, `stamp_ns`, wall time, note |
| `rosout.log` | every node's log lines (INFO and up), each with the frame count at the time |
| `params.yaml` | every running node's parameters, read `params_dump_delay_s` (5 s) after start, in the config files' own format |
| `meta.json` | topics, git commit/branch/uncommitted-file count, start/end time, frames, markers, fps; after compression, a `compressed` entry |
| `.recording` | the recorder's pid while it writes the folder (removed on close) |

## Using a run afterwards

- **Watch**: open `annotated.mkv` in any player.
- **Find the problems**: `scripts/label_review_tool.py` picks up
  `markers.jsonl` next to the predictions. `m`/`M` jump between markers, the
  marker's note is shown in a red banner, and the selected person shows the
  live-only numbers (source, theta, iris offset, AVERTED, awareness):
  ```bash
  python3 scripts/label_review_tool.py ~/human_awareness_runs/<run>/camera.mkv \
      --predictions ~/human_awareness_runs/<run>/predictions.jsonl
  ```
- **Score it**: label in the same tool, then
  `python3 scripts/score_predictions.py ~/human_awareness_runs/<run>/predictions.jsonl`.
- **Re-run with other parameters**:
  `scripts/test_tracking_head_pose_and_gaze_on_video.py` on `camera.mkv`
  with the changed flags (and the R1 rig flags, see README; the script
  doesn't read `robot_rig.yaml`).

## Compressing old runs

Nothing is ever deleted. When a new run starts, `run_recorder_node` shrinks
every earlier finished run in a background thread (`compress_previous_runs`,
on by default):

| file | becomes | on the labeled phone clip |
|---|---|---|
| `camera.mkv`, `annotated.mkv` | H.264, CRF 23, same file name | 9.7 MB -> 3.7 MB (2.6x) |
| `predictions.jsonl` | `predictions.jsonl.gz` | 1.18 MB -> 0.16 MB (7.5x) |
| `rosout.log` | `rosout.log.gz` (read with `zless`) | |
| markers, meta, params, labels | unchanged (tiny) | |

To compress the last run too (it otherwise waits for the next run to
start): `ros2 run human_run_recorder compress_runs` (`--crf N`, or another
runs folder as the first argument).

Every tool still works on a compressed run without any change to the
commands: the videos keep their names, `_label_io.load_predictions` reads
the `.gz` (and finds it when given the old `predictions.jsonl` path), and
the labels file keeps its name (`predictions.labels.json`), so labels made
before compression are still found.

Decisions:
- **CRF 23, measured on what matters: re-running the pipeline.** The full
  chain (deterministic: two runs on the same video gave identical output,
  all 2003 rows) was run on the original phone clip, on its recording as
  the recorder writes it, and on H.264 versions of that recording. Share of
  frames with the same `is_looking` as on the original video:

  | video | size | same `is_looking` as original | score vs labels (TP / FP / FN) |
  |---|---|---|---|
  | as recorded (mp4v) | 9.7 MB | 96.3% | 498 / 112 / 94 |
  | H.264 CRF 20 | 5.4 MB | 96.5% | 507 / 110 / 85 |
  | **H.264 CRF 23** | **3.7 MB** | **96.0%** | 505 / 124 / 87 |
  | H.264 CRF 28 | 1.9 MB | 93.6% | 492 / 124 / 100 |

  The recording itself already changes ~4% of decisions (it's lossy too),
  and CRF 23 adds almost nothing to that; CRF 28 adds about 3 points. Head
  yaw differs by under 1° on the median frame and the iris offset by 0.01.
  So a re-run never exactly matches what happened live; `predictions.jsonl`
  is the exact record of the live decisions, and gzip keeps it bit for bit.
- **H.264 seeks exactly**: 0 of 62 random seeks landed on a wrong frame,
  and the frame count is right, as the review tool needs.
- **Only after a check, never half-done.** Each file is written to a
  temporary file, checked (same frame count via `ffprobe -count_packets`;
  text byte-identical after decompressing), then swapped in. Steps done are
  listed in `meta.json`, so a stopped compression resumes without encoding a
  video twice (a second lossy pass would lose quality again). Leftover
  temporary files are removed on the next attempt.
- **A video that can't be re-encoded is kept as it is** and the reason goes
  in `meta.json` (`compressed.failed`), so it isn't retried on every run.
  A video cut short by a crash still compresses (tested: 58 of 58 frames).
- **Never a run that's still being recorded**: skipped while its
  `.recording` lock names a live process. A crashed run's stale lock
  doesn't block it.
- **Low priority while the robot runs**: `nice 19` and 2 ffmpeg threads
  (`compress_threads`). Stopping the recorder stops ffmpeg; the run is
  finished next time.
- **No ffmpeg**: the text is still gzipped, the videos stay as they are.

## Design decisions

- **Only the frames the pipeline processed are recorded, not every camera
  frame.** Each output message is paired with the image of the same stamp
  (the same `ImageBuffer` `visualization_node` uses), so frame N of the
  video is exactly the image the pipeline's row N was computed on. That is
  what the review tool and scorer need, and it is also what to re-run to
  reproduce the run. The camera's extra frames had no output and can't be
  scored. Cost: the video plays at the pipeline's rate, which is estimated
  from the first `video_probe_frames` (15) stamps. `timestamp_s` in the rows
  is the real time since the first frame, so time-based behaviour (fixation,
  awareness) can still be checked exactly.
- **Not a rosbag.** A raw 1280x1080 bgr8 stream at 30 Hz is ~125 MB/s. A bag
  would also need a separate exporter before the review tool could use it.
  `ros2 bag record` still works alongside for anyone who wants one.
- **Matroska (`.mkv`), not `.mp4`.** Tested with `kill -9` partway through
  writing: the `.mp4` couldn't be opened at all (its index is written on
  close), and the `.mkv` lost only the last 2 of 60 frames. A debug log
  matters most when something crashed. Seeking by frame number in the
  `.mkv` files was checked to land on the exact frame (0 misses in 44
  seeks against a sequential decode), and OpenCV reports the right frame
  count, which the review tool relies on.
- **Rows and markers are written line by line** (line-buffered files), so a
  crash keeps everything up to the last full line.
- **A marker marks the newest recorded frame when it arrives**: the frame
  that was on the viewer at that moment, give or take the pipeline delay.
  Clock-free on purpose: no comparison between the camera stamps and the
  laptop's clock, which differ on a robot. Step back with `[`/`a` from the
  marker to the moment it started.
- **`/rosout` is read with transient-local QoS, depth 1000**, matching the
  publishers, so the log also has what nodes said before the recorder
  started (model loading, device).
- **Parameters are read from the running nodes**, not copied from the yaml
  files, so command-line overrides and launch arguments are included.
- **The gaze stage's diagnostics are now on the message**
  (`PersonDetection.msg`: `gaze_gate_reason`, `iris_offset` (-1 = no face
  landmarks), `eyes_averted`), set by `gaze_node`. Without them a live run
  couldn't show why the veto did or didn't fire, which was the first R1
  problem. `visualization_node` now also shows "NOT LOOKING (eyes away)"
  live, as the video script already did.
- **Records nothing unless asked** (`record:=false` by default), and never
  overwrites a folder: a second run with the same `run_name` fails at
  start instead.

## Parameters (`config/human_run_recorder.yaml`)

| param | default | meaning |
|---|---|---|
| `image_topic` | `/camera/image_raw` | camera topic (the bringup's `image_topic` remap applies) |
| `detections_topic` | `human_awareness/awareness` | the pipeline's final output |
| `mark_topic` | `human_run_recorder/mark` | `std_msgs/String` markers, the note as data |
| `runs_dir` | `~/human_awareness_runs` | one subfolder per run |
| `run_name` | `""` | folder name; empty = start time |
| `image_buffer_size` | 60 | camera frames kept for pairing; must cover the pipeline delay |
| `aware_threshold_deg` | 45.0 | flat fallback for `is_looking`, same as the script and viz node |
| `record_annotated` | true | also write `annotated.mkv` |
| `video_probe_frames` | 15 | frames used to estimate the video fps |
| `fallback_fps` | 15.0 | fps if the stamps can't give one (unstamped source) |
| `params_dump_delay_s` | 5.0 | when to read every node's parameters |
| `log_min_level` | 20 | lowest `/rosout` level kept (10 debug, 20 info, 30 warn, 40 error) |
| `compress_previous_runs` | true | at start, shrink earlier finished runs in the background |
| `compress_crf` | 23 | H.264 quality for that (lower = better, bigger) |
| `compress_threads` | 2 | ffmpeg threads, at lowest CPU priority |

## Verified (2026-11-16)

- **Compression** (`test_compress.py`, 7 cases: all frames and rows kept
  and byte-identical text, a live run left alone, a crashed run picked up,
  the new run excluded, an interrupted run resumed without re-encoding,
  a broken video kept while the rest is compressed, no-ffmpeg mode). Over
  ROS: run A recorded, then run B started; A was compressed in the
  background in 5 s while B still recorded all 300 of 300 frames at 15 Hz.
  `compress_runs` then compressed B and, run again, found nothing to do.
  The scorer read A's `predictions.jsonl.gz` through the old path. (Those
  two runs used random-noise frames, the worst case for video: 2.7x and
  1.9x smaller. Real footage: see the table above.)
- **Unit tests** (`src/human_run_recorder/test/test_run_files.py`, 12):
  row fields and `None` rules, LOST_GALLERY skipped, debug text, fps
  estimate, a full folder round trip (both videos have exactly as many
  frames as `predictions.jsonl` has frame numbers, markers land on the
  right frame), a run shorter than the probe still writes its video, and an
  existing folder is never overwritten.
- **Throughput, recorder alone**: 1280x1080 frames at 15 Hz with detections
  0.3 s behind them (simulated pipeline delay), 20 s: 300 of 300 frames
  recorded, both videos 300 frames at 15.2 fps, no warnings.
- **Full launch with `record:=true`** (isolated `ROS_DOMAIN_ID`, the labeled
  WhatsApp clip replayed at 10 Hz on `/test/cam`, every stage on CPU): the
  folder had all seven files; `params.yaml` held all six pipeline nodes;
  `rosout.log` started with the nodes' model loading; three markers (two
  from `mark` piped from stdin, one from `ros2 topic pub`) were all saved;
  rows carried `gaze_gate_reason`/`iris_offset`/`eyes_averted` from
  `gaze_node`. A frame of `annotated.mkv` and the review tool's drawing of
  the run (marker banner, marker count) were checked by eye.
- That run only produced 26 frames in 73 s: the machine is shared (load
  ~19) and the live pipeline already running there had its detector at
  200% CPU, while the test chain also ran all models on CPU. The
  throughput test above shows the recorder itself isn't the limit.

## Not done yet

- **No total size cap.** Runs are compressed, never deleted. Very roughly,
  per minute at 1280x1080 and 15 fps: ~40 MB per video while recording and
  ~15 MB after compression, plus ~1 MB of predictions (~0.15 MB gzipped).
  `record_annotated:=false` halves the video part; `annotated.mkv` can be
  regenerated from `camera.mkv` + `predictions.jsonl`, though no command for
  that exists yet.
- **Not run on the R1 yet.** The R1's camera topic, rate and QoS are still
  unknown (see `gaze.md`, "Robot rig").
- **The marker lands at "now", not at the start of the problem.** People
  press it a second or two late; that's left to stepping back in the review
  tool rather than guessing an offset.
- **The video script doesn't read `params.yaml`**: re-running a recorded
  run with its exact parameters still means passing them as flags.
