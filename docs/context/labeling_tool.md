# Manual labeling / review tool

## What it is

A frame-by-frame review workflow for checking the pipeline's subjective
"is this person looking at the robot" output against human judgment on real
footage. (ReID got the same treatment only briefly - a dedicated toggle-based
tool was built and then deliberately dropped; see "ReID: tried, dropped"
below for why.) Three pieces:

- `scripts/test_tracking_head_pose_and_gaze_on_video.py --export-jsonl PATH`
  (full chain: tracking + head pose + iris/gaze fusion) and
  `scripts/test_tracking_and_head_pose_on_video.py --export-jsonl PATH`
  (head pose only, no iris/MediaPipe models to load - faster to iterate on
  when iris isn't what you're checking) - both write one JSON record per
  visible track per frame in the same schema (bbox, track_id/state, head
  pose, `gaze_robot_deg`, `is_looking`). The head-pose-only script's
  `gaze_source` is always `"head"` (no iris fusion without MediaPipe), but
  it still runs the phase-2 geometric decision straight off head pose
  alone - the same code path `human_gaze.gaze_pipeline.GazePipeline`
  already uses for its own head-only fallback frames, which needs no iris
  input at all - so `sees_robot`/`direct_contact`/`distance_m` are real
  there too, not placeholders, and both exports stay interchangeable as
  far as the tools below are concerned. Empirically, the geometric
  decision's default angular tolerance (`head_only_extra_deg` etc.) turned
  out too tight for real footage - see "Validating against real footage"
  below for the numbers.
  This is the "prediction" side of the comparison.
- `scripts/label_review_tool.py` — an OpenCV window that steps through the
  video frame-by-frame, overlays the exported predictions, and lets you
  record ground truth by clicking a track and pressing a key (see its own
  docstring for the full key list).
- `scripts/score_predictions.py` — diffs predictions against the labels
  file and reports a gaze confusion matrix (precision/recall/false-positive
  rate) plus ReID error counts (ID switches, false merges, both zero/empty
  unless you've used `label_review_tool.py`'s `i`/`x` keys), with the exact
  frame ranges of every disagreement so you can jump straight back into the
  review tool instead of re-scrubbing a whole clip looking for them.

## ReID: tried, dropped

A dedicated `label_reid_tool.py` was built first (toggle the selected
track against its nearest non-overlapping candidate: `y`/`n` for same/
different, no typing) - per the project owner's own initial request to
validate identity before gaze. Deleted once tried against real footage:
ReID errors (an ID switch, a false merge) are visually obvious just
watching the tracked/annotated output video - a dedicated structured
labeling pass wasn't worth the overhead for this project's footage.
`label_review_tool.py`'s `i`/`x` keys (rename/split a track's person_id)
are what's left of that identity-correction machinery - still functional,
just reached by typing rather than a candidate-gallery toggle, since nothing
else in the tool depends on the toggle-specific affordances anymore.

## Why intervals, not per-frame labels

Labeling every single frame independently doesn't scale past a few seconds
of footage, and isn't how a human actually judges gaze anyway - you notice
"they started looking away around here" and "they're still looking away
here", not a fresh independent judgment every 33ms. `_label_io.py`'s
`Labels.set_gaze_label` records one key-press as an edit to an interval
(extending, splitting, or opening one), so you only press a key at an
actual transition. The same reasoning applies to identity: a correction is
"this track_id is the same person as that one" (merge - fixes an ID
switch) or "this track_id stops being this person at this frame" (split -
fixes a false merge), both expressed as edits to `identity_segments`
rather than a per-frame person_id.

## Why OpenCV, not a web app

Every other test/debug entry point in this project (`test_*_on_video.py`)
is already a plain Python + OpenCV script with no extra service to run -
the labeling tool follows the same shape instead of introducing a second
stack (a local web server + browser UI) for one feature. The cost is a
hand-rolled text-entry prompt for person_id input (`prompt_text` in
`label_review_tool.py`) since a bare OpenCV window has no native widgets -
acceptable for the short strings this tool ever asks for.

## On-disk format

`_label_io.py` defines both: predictions are read as JSONL (one `json.loads`
per line, grouped by frame), labels are a single JSON file next to the
predictions file (`<predictions>.labels.json` by default) holding
`identity_segments` (per-track_id frame ranges tagged with a chosen
`person_id`) and `gaze_labels` (per-track_id list of labeled frame
intervals). `Labels.load_or_init` auto-seeds one identity segment per
track_id spanning its whole observed lifetime the first time a video is
opened (defaulting to trusting the tracker's own IDs), so there's nothing
to set up before you can start correcting it.

**Keep labeled data out of `test_output/`** - that directory is
`.gitignore`d (disposable annotated-video previews), but a `.labels.json`
file represents real manual labeling work. Point `--export-jsonl` and
`label_review_tool.py --predictions` at a tracked directory instead (e.g.
`data/ground_truth/<video_stem>.jsonl`) if the labels are worth keeping
across machines/commits.

## Validating against real footage

First real use of this tool (a multi-person WhatsApp clip, person id 1
labeled by hand across 583 frames) immediately surfaced a real bug, not
just a scoring exercise: the original head-only pipeline had **37%
recall** on "looking" (154 missed out of 244 real "looking" frames,
`human_head_pose.head_crop`'s `min_keypoint_conf` gate on *both* shoulders
rejected 151 of those outright - confirmed by inspecting the actual frame:
a turned/three-quarter stance with one shoulder foreshortened below
confidence while the face stayed clearly visible). `neck_point`'s
single-shoulder fallback (see its own docstring) fixed that directly -
recall rose to 98.4% - at a real precision cost (0.874 -> 0.702, since the
less-centered crop produces noisier angle estimates on some frames).

Turning on the phase-2 geometric decision (this file's "head-pose-only"
export, described above) swung the opposite way at its *default* angular
tolerance: precision jumped to 0.921 (6 false positives, confirming the
3D cone test genuinely resolves cases the flat 2D threshold can't - e.g. a
person turned toward another person in the room, not the camera, which
the flat `gaze_robot_deg - bearing` approximation can't tell apart from
"off-center but looking at the camera") but recall collapsed to 0.287.
Root cause: `head_only_extra_deg`'s default (10°) plus
`peripheral_buffer_deg` (5°) plus the robot's physical cone
(`atan(robot_radius_m / distance)`, a few more degrees at typical
distances) totals roughly 25-30° of tolerance - tighter than this
pipeline's own real noise floor, evidenced directly by two genuinely-
"looking" frames that measured 48° and 53° under the flat-threshold run.
Raising `head_only_extra_deg` to 35° (no other parameter changed) landed
at precision 0.779 / recall 0.939 - the best F1 of every variant tried,
confirming `head_only_extra_deg` really is the right knob for this gap
(as `docs/context/gaze.md`'s own design-decisions section already expected
it to be), just defaulted too conservatively for real handheld footage.
**Promoted to `test_tracking_and_head_pose_on_video.py`'s own
`--head-only-extra-deg` default** (was 10.0, matching the full gaze/iris
pipeline's separately-tuned value) - the two scripts cover different
scenarios (head-only vs. iris-available) and don't need the same default.
The full gaze/iris pipeline's `head_only_extra_deg=10.0`
(`human_gaze/gaze_node.py`, `config/human_gaze.yaml`) is untouched - it was
tuned against different evidence (see `docs/context/gaze.md`'s "Empirical
validation") and this finding doesn't bear on it.

None of this is a finished calibration - one person, one clip, one
`camera_hfov_deg`/`camera_height_m` guess. It's the first real evidence
that the gap exists and roughly how big the fix needs to be, which is
exactly what labeling against real footage was supposed to surface.

Labeling the other two people in the same clip (tracks 2 and 4, plus a
brief track 5) exposed a different failure: overall precision 0.60, and
0.38 for the man (track 2), whose 271 false positives were nearly all head
toward the camera with eyes clearly turned to the side - invisible to head
pose. The full iris pipeline didn't help (MobileGaze barely engaged: eyes
too small, then too jittery to fixate). Recording iris landmarks on every
labeled frame showed that where the iris sits between the eye corners
separates the two cases well even at ~11px eye width, which became the
iris-position veto (`docs/context/gaze.md`, "Iris-position veto"): end to
end, precision 0.60 -> 0.83, recall 0.88 -> 0.86. Fitted on this clip
only - a second labeled clip is the real test.

## Not done yet

- No playback (autoplay at video fps) - frame-by-frame + jump-to-event
  navigation only, on the assumption that precise transition-finding
  matters more here than watching at real speed.
- No cross-video aggregation of `person_id`s - the same real person
  appearing in two different video files gets labeled independently in
  each; nothing currently ties those together into one global identity.
- `score_predictions.py`'s ReID metrics only count what you explicitly
  labeled (a merge/split you made) - it has no way to flag a suspected ID
  switch or false merge on its own; finding those to label in the first
  place is still a manual review job (`e`/`E` in the review tool jumps
  between track-appearance/disappearance frames to make that search
  faster, but doesn't do it for you).
