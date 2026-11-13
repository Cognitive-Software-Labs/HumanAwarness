# Awareness over time (stage D, first dimension)

## Design reference

`docs/context/design_doc.md`: Part II "D · Per-ID aggregation" (temporal
fusion per ID, awareness × availability × willingness, each a value plus a
confidence with decay), its "D · Aggregation" parameter table
(`tau_awareness_s`, `cue_max_age_s`, `init_value`, `fusion_weights`), and
Part III's "All cues → fusion" adapter ("last value" cache; missing =
unknown, not 0).

Only the **awareness** dimension is built. Availability, willingness, the
trajectory cues, the combined cooperation score and stage C (environment)
are not: they need robot odometry or labeled environment data the project
doesn't have yet.

## What it is

Every stage before this one (A, B1, B2) produces per-frame cues with no
memory of the person. Awareness is the first per-person state: for each
`track_id`, a **value** ("knows the robot is there", 0..1) and a
**confidence** (how much recent observation backs the value, 0..1), driven
by `human_gaze`'s per-frame `sees_robot` decision and decaying over
`tau_awareness_s` = 8 s.

Awareness is not "looking right now". Someone who looked at the robot three
seconds ago and is now talking to someone else still knows it's there, so
the value climbs within a second of a real look and fades over about 8 s
afterwards. It does not drop as soon as the person looks away.

Code: `src/human_awareness/human_awareness/awareness.py`
(`AwarenessAggregator`, pure Python, no ROS imports, unit-tested in
`src/human_awareness/test/`), `adapters.py` (PersonDetection → cue) and
`awareness_node.py` (the ROS wrapper).

## The model

Per track, over each stretch of time, one of three cases holds. Each update
is the exact solution for constant evidence over the interval:

| cue holding | value | confidence |
|---|---|---|
| sees the robot (weight w) | rises toward 1, time constant `rise_time_s / w` | rises toward 1, time constant `confidence_time_s / w` |
| in view, doesn't see it (weight w) | relaxes toward `(1 - w) × init_value` with `tau_awareness_s` | rises toward 1, time constant `confidence_time_s / w` |
| no usable cue | relaxes toward `init_value` (0.5) with `tau_awareness_s` | decays toward 0 with `tau_awareness_s` |

With the defaults (head-only w = 0.5, iris w = 1): 1 s of head-only "sees
the robot" takes a new person from 0.5 to 0.82; a single head-only frame at
30 fps closes 3.3% of the remaining gap to 1 (+0.016 from 0.5). Watched not
looking, a head-only person settles at 0.25 and an iris-confirmed one at 0.
Not seen at all, the value settles back at 0.5 with confidence 0.

## Design decisions

- **Fast rise, slow fade.** One clear look is enough to know the robot is
  there; looking away doesn't undo it. The rise has its own time constant
  (`rise_time_s` = 0.5 s at weight 1, so 1 s for head pose), which damps
  one-frame flickers. Everything else decays with the design doc's
  `tau_awareness_s`.
- **"Not looking" is weak evidence, "not seen" is none.** A valid
  observation of someone not looking pulls the value down over
  `tau_awareness_s`. A head-only reading only pulls it halfway down (to
  0.25), because head pose can't see the eyes. A frame with no usable cue
  pulls toward 0.5 ("don't know"), never toward 0, per the design doc's
  "missing = unknown, not 0".
- **Confidence grows with any valid observation, not only with
  sightings.** Watching someone for 20 s who never looks at the robot is
  real evidence that they haven't noticed it. If only sightings raised
  confidence, that person would stay "don't know" forever, and the future
  combination step (which drops dimensions under `min_confidence` = 0.3)
  would ignore exactly the people who haven't noticed the robot.
- **Iris counts double head pose** (`weight_iris` = 1.0, `weight_head` =
  0.5), standing in for the design doc's `fusion_weights` file. The weight
  scales the rise rate, the not-looking floor and the confidence build-up
  together. This is the design doc's premise (iris is the finer signal),
  not a measured ratio. The labeled clip can't measure it: see "Empirical
  validation".
- **A cue only comes from a fresh observation:** an ACTIVE track, a valid
  head pose, and a phase-2 decision that actually ran (`distance_method`
  not `""`/`"none"`). Anything else is "no cue":
  - `LOST_KALMAN`/`LOST_GALLERY` tracks are still published, but with the
    keypoints of their last match. `head_pose_node` and `gaze_node` skip
    them (as the video script does), so they arrive with
    `head_pose_valid=false`. The adapter also checks the state itself, so
    a stream recorded before those nodes skipped them still gives no cue.
  - A frame where phase 2 couldn't run (no confident nose/eyes, no
    distance) has `sees_robot=False` without having tested anything.
  - The script's flat phase-1 fallback (`--aware-threshold-deg`) stays a
    display-only label and is not used as a cue.
- **`sees_robot`, not `direct_contact`.** Peripheral sight of the robot is
  enough to know it's there (the design doc's own reason for the robot
  cone). Direct contact will matter more for willingness.
- **The iris-position veto needs nothing special.** Over ROS it arrives as
  `sees_robot=False` (`eyes_averted` isn't on the message), and the script
  feeds the same vetoed `sees_robot`, so ROS and script agree.
- **`cue_max_age_s` is a hold, not a filter on arrival.** A cue holds from
  its frame until a newer one replaces it or it is 0.5 s old (the design
  doc's "last value" cache). A one- or two-frame detection gap doesn't
  break a run of evidence. A long gap is "no cue" after 0.5 s. A frame with
  no usable cue doesn't cancel the held one.
- **Time comes from header stamps, not the node clock.** The result is
  independent of frame rate and of how often `observe` is called (a unit
  test checks 10/30/60 fps give the same values), and a recorded stream
  replays identically at any speed. Three exceptions, decided by
  `adapters.classify_stamp`:
  - A message up to 1 s older than the newest one was delivered out of
    order. It is dropped with a warning, since the newer state already
    covers that moment.
  - A jump back of more than 1 s means the source restarted (looping bag,
    reset simulation). All state is reset, with a warning.
  - An unset stamp (0) is timed by when the message arrived, with a
    one-time warning. The upstream nodes keep working on such a stream
    (they fall back to a nominal frame step), and awareness would
    otherwise never move.
- **State lives as long as the tracker publishes the ID.**
  `human_tracking` publishes every live track, `LOST_*` included, in every
  message. The state of a lost person keeps decaying toward "don't know"
  and carries on when ReID brings the same `track_id` back. An ID missing
  from a message has been forgotten (gallery timeout) or merged, so its
  state is dropped.
- **Extends `PersonDetection` again rather than adding a message type**,
  as every stage so far has: `awareness_value` / `awareness_confidence`,
  with message defaults 0.5 / 0 so a detection that never went through
  `awareness_node` reads as "don't know", not "not aware". The node
  republishes the whole `PersonDetectionArray`, one message out per message
  in, with the same header.
- **Published per input message, not at `output_rate_hz`.** The design
  doc's fixed 10 Hz tick is for the combined score, which will merge
  streams at different rates. For one dimension fed by one ~10-30 Hz
  stream, publishing per message keeps it aligned with the frames.

## Interfaces

`awareness_node` (`ros2 launch human_awareness human_awareness.launch.py`):

- subscribes `gazes_topic` (default `human_gaze/gazes`,
  `human_awareness_msgs/PersonDetectionArray`) from `gaze_node`;
- publishes `awareness_topic` (default `human_awareness/awareness`), the
  same message with, for every detection with `track_id >= 0`:
  - `awareness_value` (float32, 0..1): rises while the person sees the
    robot, fades over `tau_awareness_s` otherwise; 0.5 = don't know;
  - `awareness_confidence` (float32, 0..1): 0 = no observation behind the
    value. A consumer should ignore the value below some confidence (the
    design doc's `min_confidence` = 0.3).
  - Untracked detections (`track_id = -1`) pass through at 0.5 / 0.

`scripts/test_tracking_head_pose_and_gaze_on_video.py` draws each visible
person's value, confidence and a bar ("aware 0.82 conf 0.61"), and adds
`awareness_value`/`awareness_confidence` to every `--export-jsonl` row. It
builds the same cue the adapter does, so script and node agree (checked,
see below).

## Parameters (`AwarenessAggregator` / `awareness_node`, `config/human_awareness.yaml`)

| param | default | meaning |
|---|---|---|
| `tau_awareness_s` | 8.0 | design doc: how long a sighting keeps someone aware; also how fast confidence fades without observation |
| `cue_max_age_s` | 0.5 | design doc: a cue holds for at most this long after its frame |
| `init_value` | 0.5 | design doc: where a new `track_id` starts (confidence always starts at 0); also the "don't know" level unseen people relax to |
| `rise_time_s` | 0.5 | new: time constant of the rise while seeing the robot, at weight 1 (head pose: 1 s) |
| `confidence_time_s` | 2.0 | new: time constant of the confidence build-up while a cue holds, at weight 1 (head pose: 4 s) |
| `weight_iris` | 1.0 | design doc's `fusion_weights`, iris source |
| `weight_head` | 0.5 | design doc's `fusion_weights`, head-pose source |

The two new time constants are picked, not fitted. `rise_time_s` caps
what one false-positive frame can do (at 30 fps, +0.016 from 0.5, +0.025
from the head-only floor of 0.25) while a 0.3 s look, the design doc's
minimum fixation, already moves a new person from 0.5 to 0.63 (head) or
0.73 (iris). `confidence_time_s` makes about 1.4 s of
head-only observation clear the design doc's `min_confidence` = 0.3.

## Empirical validation

- **Unit tests** (`src/human_awareness/test/test_awareness.py`, 33 cases):
  start at 0.5/0; missing cues never move it off 0.5; rise and fade match
  their time constants; iris counts more than head; a one-frame flicker
  moves it < 0.03; a cue counts for at most `cue_max_age_s`; identical
  results at 10/30/60 fps; state survives LOST and resumes; pruning;
  out-of-order stamps; the adapter's ACTIVE / head pose / phase-2 rules;
  in-order / reordered / restart stamp classification.
- **The labeled WhatsApp clip** (`data/ground_truth/whatsapp_1017_headpose.*`,
  full script, `--reid-device cuda:0 --head-pose-device cuda:0
  --gaze-device cuda:0 --head-only-extra-deg 35`). The labels mean "looking
  right now", not "aware", so this is a qualitative check, not a score:
  - Rises during looking runs: track 1, frames 15-63: 0.63 → 0.92; track 2,
    247-294: 0.76 → 0.95; track 4, 127-160: 0.45 → 0.77, and 407-484 →
    0.95.
  - Fades over ~8 s when a not-looking stretch follows: track 4 after frame
    484 reads 0.95 → 0.89 / 0.81 / 0.70 at +1 / +2 / +4 s (the model alone,
    head-only, predicts 0.87 / 0.80 / 0.67; the difference is a few
    false-positive frames). Track 2 across its 5.7 s not-looking run
    (frames 379-549): 0.93 → 0.70 at +4 s.
  - No jumps: the largest frame-to-frame change on any track is 0.018;
    the 7 isolated one-frame `sees_robot` flickers of the labeled people
    moved it by ≤ 0.012.
  - It is not a "looking now" signal, as intended: mean value on labeled
    looking vs not-looking frames is 0.88 vs 0.89 (track 1), 0.83 vs 0.80
    (track 2) and 0.77 vs 0.69 (track 4). Tracks 1 and 2 glance at the
    robot every few seconds and stay aware throughout.
  - ReID returns carry the state on: track 4 is gone for 3.0 s (frames
    320-409) and comes back at 0.555 / 0.387 (left at 0.581 / 0.566: both
    decayed by exactly e^(-3/8)); track 1 is gone 0.9 s and comes back at
    0.768 / 0.619.
  - The gaze stage's output is untouched: with awareness added, the export
    is identical row for row to the script before this change (2001 rows),
    and scoring still gives precision 0.83 / recall 0.86 / 105 false
    positives.
- **Iris weighting is untested on real footage.** The clip has only 11
  iris-sourced frames (tracks 1 and 4). All 10 that are labeled are labeled
  "looking", yet all 11 came out `sees_robot=False`: yaw alone puts them
  10-23° from the camera, and with pitch the 3D angle misses the iris
  branch's allowance (robot cone + `peripheral_buffer_deg`, 15-19° at their
  1.2-1.7 m), which doesn't get the head branch's 35° widening. Here they
  pull awareness toward 0 instead of 0.25, but the largest effect anywhere
  is -0.004 against treating them as head-only frames. Too few frames to
  tune on, but iris `sees_robot` looks strict on this footage, which
  matters to the gaze stage more than to this one.
- **Under ROS**: the node, started with
  `ros2 launch human_awareness human_awareness.launch.py` (yaml loaded,
  checked with `ros2 param get`), was fed the clip's export as
  `PersonDetectionArray` messages on `human_gaze/gazes`. Each track was
  published from its first to its last frame, `LOST_GALLERY` while not
  visible (118 such entries). Over all 2001 (frame, track) pairs the node's
  output matches the script's in-process values to 3e-8 (float32). A
  22.6 s jump back reset everything to 0.5 / 0 with the warning; an
  untracked detection passed through at 0.5 / 0. A second run checked the
  stamp handling:
  - a message 33 ms older than the newest got no reply, and the next
    in-order one carried on without a reset;
  - an 8 s jump back reset the state;
  - 31 unstamped messages, 1.05 s of `sees_robot`, took a new person from
    0.5 to 0.82, the same as stamped frames over that time.

## Not done yet

- **Availability, willingness, the trajectory/torso/gesture cues, the
  combined cooperation score (geometric mean, `min_confidence`,
  `score_on`/`score_off` hysteresis, `output_rate_hz`) and stage C's
  context modulation**: they need robot odometry or labeled environment
  data first.
- **No awareness ground truth.** The only labels are "looking right now"
  on one clip, so the shape of the curve (time constants, weights, the
  0.25 head-only floor) is reasoned and checked qualitatively, not fitted.
  A label for "has this person noticed the robot yet" (e.g. one marked
  moment per person) would allow an actual score.
- **Iris vs head weight** has no real-footage support either way (see
  "Empirical validation").
- **Reconciliation merges lose up to `reconcile_window_s` (1.5 s) of
  evidence.** When `human_tracking` folds a just-created track into an
  older lost identity, the young ID disappears and its state is pruned;
  the old ID carries on with its own decayed state. The tracker records
  merges (`Tracker.merged_ids`) but doesn't publish them, so the node
  can't transfer the young ID's evidence.
- **Not run against a live camera or the full ROS chain** (perception →
  tracking → head pose → gaze → awareness), the same gap every other
  package notes. The node is verified on replayed real-clip cues; the
  upstream nodes on synthetic input or the video scripts.
