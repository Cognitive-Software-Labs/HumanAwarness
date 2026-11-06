"""Per-person awareness over time ("knows the robot is there"): the first
dimension of the design doc's stage D, per-ID aggregation
(docs/context/awareness.md). Pure Python, no ROS: `awareness_node` feeds it
from human_gaze's PersonDetectionArray, the video test script straight from
`GazeTrackResult`.

Each track_id has a value and a confidence, both in [0, 1], moved by
whichever cue holds over each stretch of time:

- sees the robot (`sees_robot`, weight w from the gaze source): value
  rises toward 1 with time constant `rise_time_s / w`;
- in view but doesn't see it: value relaxes with `tau_s` toward
  `(1 - w) * init_value`, which is 0 for an iris reading and only part of
  the way down for head pose, since head pose can't see the eyes;
- no usable cue (not detected, LOST_*, no head pose, or the last cue is
  older than `cue_max_age_s`): value relaxes with `tau_s` back toward
  `init_value` ("don't know"), never toward 0;
- confidence rises toward 1 with time constant `confidence_time_s / w`
  while a cue holds, and decays with `tau_s` while none does.

A cue holds from its stamp until a newer one replaces it or it is
`cue_max_age_s` old, the design doc's "last value" cache: a detection gap of
a frame or two doesn't break a run of evidence, a long one doesn't stretch
it. Every step is the exact solution for constant evidence over its
interval, so the result doesn't depend on the frame rate or on how often
`observe` is called.
"""

import math
from dataclasses import dataclass

# human_gaze's `gaze_source` values (PersonDetection.msg): which signal produced `sees_robot`.
SOURCE_IRIS = "iris"
SOURCE_HEAD = "head"


@dataclass(frozen=True)
class AwarenessCue:
    """One frame's gaze decision for one person."""

    sees_robot: bool
    source: str  # SOURCE_IRIS or SOURCE_HEAD


@dataclass(frozen=True)
class AwarenessEstimate:
    value: float  # 0..1, "knows the robot is there"; init_value (0.5) = don't know
    confidence: float  # 0..1, how much recent observation backs `value`; 0 = none


@dataclass
class _TrackState:
    value: float
    confidence: float
    stamp_s: float
    cue: AwarenessCue | None = None
    cue_stamp_s: float = 0.0


class AwarenessAggregator:
    def __init__(
        self,
        tau_s: float = 8.0,
        cue_max_age_s: float = 0.5,
        rise_time_s: float = 0.5,
        confidence_time_s: float = 2.0,
        weight_iris: float = 1.0,
        weight_head: float = 0.5,
        init_value: float = 0.5,
    ):
        """`tau_s`, `cue_max_age_s` and `init_value` are the design doc's
        `tau_awareness_s`, `cue_max_age_s` and `init_value`; `weight_*` stand
        in for its `fusion_weights` file. `rise_time_s` and
        `confidence_time_s` are new here (docs/context/awareness.md)."""
        for name, v in (
            ("tau_s", tau_s),
            ("cue_max_age_s", cue_max_age_s),
            ("rise_time_s", rise_time_s),
            ("confidence_time_s", confidence_time_s),
        ):
            if v <= 0.0:
                raise ValueError(f"{name} must be > 0, got {v}")
        for name, w in (("weight_iris", weight_iris), ("weight_head", weight_head)):
            if not 0.0 < w <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {w}")
        if not 0.0 <= init_value <= 1.0:
            raise ValueError(f"init_value must be in [0, 1], got {init_value}")
        self.tau_s = tau_s
        self.cue_max_age_s = cue_max_age_s
        self.rise_time_s = rise_time_s
        self.confidence_time_s = confidence_time_s
        self.init_value = init_value
        self._weights = {SOURCE_IRIS: weight_iris, SOURCE_HEAD: weight_head}
        self._states: dict[int, _TrackState] = {}

    def observe(self, track_id: int, stamp_s: float, cue: AwarenessCue | None) -> AwarenessEstimate:
        """Advances `track_id` to `stamp_s`, then records `cue`, which holds
        from `stamp_s` on. `None` means no usable cue this frame: unknown,
        not "doesn't see the robot". A new track_id starts at `init_value`
        with confidence 0. A stamp older than the track's current one is
        out of order: nothing changes."""
        if cue is not None and cue.source not in self._weights:
            raise ValueError(f"unknown gaze source {cue.source!r}, expected one of {sorted(self._weights)}")
        state = self._states.get(track_id)
        if state is None:
            state = _TrackState(self.init_value, 0.0, stamp_s)
            self._states[track_id] = state
        elif stamp_s < state.stamp_s:
            return AwarenessEstimate(state.value, state.confidence)
        else:
            self._advance(state, stamp_s)
        if cue is not None:
            state.cue = cue
            state.cue_stamp_s = stamp_s
        return AwarenessEstimate(state.value, state.confidence)

    def get(self, track_id: int) -> AwarenessEstimate | None:
        state = self._states.get(track_id)
        return AwarenessEstimate(state.value, state.confidence) if state is not None else None

    def prune(self, track_ids) -> None:
        """Drops every track not in `track_ids`. Pass every track the tracker
        still publishes, LOST_KALMAN/LOST_GALLERY included, so a person who
        comes back under the same track_id keeps their state."""
        keep = set(track_ids)
        for track_id in [t for t in self._states if t not in keep]:
            del self._states[track_id]

    def reset(self) -> None:
        self._states.clear()

    def _advance(self, state: _TrackState, stamp_s: float) -> None:
        t = state.stamp_s
        if state.cue is not None:
            held_until = state.cue_stamp_s + self.cue_max_age_s
            cue_end = min(stamp_s, held_until)
            if cue_end > t:
                self._apply_cue(state, state.cue, cue_end - t)
                t = cue_end
            if stamp_s >= held_until:
                state.cue = None
        if stamp_s > t:
            self._apply_no_cue(state, stamp_s - t)
        state.stamp_s = stamp_s

    def _apply_cue(self, state: _TrackState, cue: AwarenessCue, dt: float) -> None:
        w = self._weights[cue.source]
        if cue.sees_robot:
            state.value = 1.0 - (1.0 - state.value) * math.exp(-w * dt / self.rise_time_s)
        else:
            target = (1.0 - w) * self.init_value
            state.value = target + (state.value - target) * math.exp(-dt / self.tau_s)
        state.confidence = 1.0 - (1.0 - state.confidence) * math.exp(-w * dt / self.confidence_time_s)

    def _apply_no_cue(self, state: _TrackState, dt: float) -> None:
        decay = math.exp(-dt / self.tau_s)
        state.value = self.init_value + (state.value - self.init_value) * decay
        state.confidence *= decay
