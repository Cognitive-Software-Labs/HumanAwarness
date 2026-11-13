"""Per-track fixation detection: only a gaze direction the person actually
held for a moment counts as a real "looking at X," not a single noisy
frame mid-saccade - matching the design doc's framing (docs/context/gaze.md):
a fixation needs `fixation_min_ms` of staying within `fixation_max_dev_deg`,
and a short gap (a blink, or one frame gating failed) under
`blink_gap_max_ms` doesn't reset it - "Clipitul nu intrerupe o fixatie"
(blinking doesn't interrupt a fixation). A longer gap ends it, whether its
frames were fed as None or not fed at all.

One `FixationTracker` instance per tracked person (owned by the caller,
keyed by track_id) - fixation state is inherently per-identity, the same
way `human_tracking`'s own per-ID state is.
"""

from dataclasses import dataclass


@dataclass
class FixationState:
    is_fixating: bool = False
    angle_deg: float | None = None  # mean angle over the current fixation, once confirmed
    start_ts: float | None = None  # when the confirmed fixation began (identifies it)


class FixationTracker:
    def __init__(
        self,
        fixation_min_ms: float = 300.0,
        fixation_max_dev_deg: float = 4.0,
        blink_gap_max_ms: float = 300.0,
    ):
        self.fixation_min_s = fixation_min_ms / 1000.0
        self.fixation_max_dev_deg = fixation_max_dev_deg
        self.blink_gap_s = blink_gap_max_ms / 1000.0

        self._start_ts: float | None = None
        self._values: list[float] = []
        self._last_valid_ts: float | None = None

    def update(self, angle_deg: float | None, timestamp_s: float) -> FixationState:
        """Feed one frame's gaze angle (or None if unavailable this frame -
        a blink, a failed gate, an occlusion) and get back whether a
        fixation is currently confirmed. Frames the caller skips entirely
        (e.g. while the track is LOST) count as a gap too."""
        if angle_deg is None:
            if self._start_ts is not None and self._gap_exceeds_blink(timestamp_s):
                self._reset()
            return self._current_state()

        if self._start_ts is None or self._gap_exceeds_blink(timestamp_s):
            # Checked here as well as on None updates: after unfed frames, the
            # old fixation would otherwise absorb this reading and count as
            # held across the whole gap.
            self._start_fixation(angle_deg, timestamp_s)
        else:
            mean_angle = sum(self._values) / len(self._values)
            if abs(angle_deg - mean_angle) > self.fixation_max_dev_deg:
                # Moved too far from the running fixation to still be the
                # same one - this sample starts a new candidate instead.
                self._start_fixation(angle_deg, timestamp_s)
            else:
                self._values.append(angle_deg)
                self._last_valid_ts = timestamp_s

        return self._current_state()

    def _gap_exceeds_blink(self, timestamp_s: float) -> bool:
        if self._last_valid_ts is None:
            return False
        return (timestamp_s - self._last_valid_ts) > self.blink_gap_s

    def _start_fixation(self, angle_deg: float, timestamp_s: float) -> None:
        self._start_ts = timestamp_s
        self._values = [angle_deg]
        self._last_valid_ts = timestamp_s

    def _reset(self) -> None:
        self._start_ts = None
        self._values = []
        self._last_valid_ts = None

    def _current_state(self) -> FixationState:
        if self._start_ts is None or self._last_valid_ts is None:
            return FixationState(is_fixating=False, angle_deg=None)
        duration_s = self._last_valid_ts - self._start_ts
        if duration_s < self.fixation_min_s:
            return FixationState(is_fixating=False, angle_deg=None)
        mean_angle = sum(self._values) / len(self._values)
        return FixationState(is_fixating=True, angle_deg=mean_angle, start_ts=self._start_ts)
