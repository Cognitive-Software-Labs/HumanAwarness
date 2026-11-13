"""Orchestrates one tracked person's gaze estimate for one frame: re-derive
the same head crop B1 (human_head_pose) uses -> gate with MediaPipe ->
MobileGaze -> plausibility check against head pose -> fixation -> One Euro
-> fuse with head pose into a final angle -> phase-2 geometric decision ->
iris-position veto on head-only decisions (docs/context/gaze.md). One
`GazePipeline` instance is shared across all tracked people; per-track state
(fixation, smoothing) lives in dicts keyed by `track_id`, pruned when a
track is no longer published at all (not just momentarily lost - see
`prune`).

Why the head crop is re-derived here rather than reused from B1 directly:
human_head_pose's own output message doesn't carry the crop image, only the
resulting angles - the same `extract_head_crop` adapter is called again
with the same geometry, the same way `scripts/test_tracking_and_head_pose_
on_video.py` already re-derives a track's head region for a second
consumer. The only difference from B1's own crop is the minimum-size floor
(`mp_min_head_px`, see docs/context/gaze.md): MediaPipe needs a
meaningfully larger crop than 6DRepNet360 does to resolve eye/iris detail.

The phase-2 geometric decision (distance -> robot cone -> real 3D angle,
replacing the flat `gaze_robot_deg` threshold used by phase-1 demos) is
computed independently of iris gating, straight off keypoints + whichever
of (iris yaw/pitch, head yaw/pitch) the fusion above already selected - so
it still produces a `sees_robot`/`direct_contact` call even on frames where
iris gating failed outright, same as the flat angle already does.
"""

from dataclasses import dataclass, replace

import numpy as np

from human_gaze.adapters import crop_relative_to_frame_xyxy, extract_face_crop
from human_gaze.camera_model import (
    CameraRobotOffset,
    intrinsics_from_hfov,
    person_position_camera_frame,
    robot_center_camera_frame,
)
from human_gaze.distance_estimator import estimate_distance_m
from human_gaze.fixation import FixationTracker
from human_gaze.fusion import SOURCE_HEAD, fuse
from human_gaze.gaze_estimator import GazeEstimator, gaze_yaw_pitch_to_vector
from human_gaze.geometric_decision import GeometricGazeResult, evaluate as evaluate_geometric
from human_gaze.mediapipe_gate import MediaPipeGate
from human_gaze.one_euro_filter import OneEuroFilter
from human_head_pose.adapters import extract_head_crop
from human_head_pose.geometry import bearing_deg

_NOSE = 0
_LEFT_EYE, _RIGHT_EYE = 1, 2


@dataclass
class GazeTrackResult:
    gaze_robot_deg: float  # final, fused angle - always populated once head pose is valid
    source: str  # "iris" or "head" - which signal actually produced gaze_robot_deg
    iris_valid: bool  # whether an iris reading was available at all this frame (gated+estimated, pre-fixation)
    iris_fixating: bool
    gate_reason: str  # diagnostic: why iris gating passed/failed this frame
    iris_yaw_deg: float = 0.0  # fixation-confirmed, filtered iris yaw - raw, NOT bearing-corrected; 0 if not fixating
    face_bbox_xyxy: tuple[float, float, float, float] | None = None  # frame-absolute, for display
    geometric: GeometricGazeResult | None = None  # phase-2 decision; None if no reference pixel was available at all
    iris_offset: float | None = None  # eye widths between the iris and where it'd sit looking at the camera; None without face landmarks
    eyes_averted: bool = False  # head-only frame with iris_offset over the veto limit; geometric sees_robot/direct_contact forced False


class GazePipeline:
    def __init__(
        self,
        gaze_weights_path: str | None = None,
        mediapipe_model_path: str | None = None,
        device: str = "cpu",
        mp_min_head_px: float = 80.0,
        min_eye_px: float = 12.0,
        max_yaw_gaze_deg: float = 45.0,
        max_eye_head_deg: float = 30.0,
        ear_min: float = 0.2,
        blur_min: float = 0.0,
        fixation_min_ms: float = 300.0,
        fixation_max_dev_deg: float = 4.0,
        blink_gap_max_ms: float = 300.0,
        one_euro_min_cutoff: float = 1.0,
        one_euro_beta: float = 0.3,
        camera_height_m: float = 1.2,
        camera_pitch_deg: float = 0.0,
        robot_radius_m: float = 0.3,
        theta_contact_deg: float = 12.0,
        peripheral_buffer_deg: float = 5.0,
        head_only_extra_deg: float = 10.0,
        shoulder_width_prior_m: float = 0.40,
        person_height_prior_m: float = 1.70,
        camera_offset_right_m: float = 0.0,
        camera_offset_down_m: float = 0.0,
        camera_offset_forward_m: float = 0.0,
        iris_veto_max_offset: float = 0.12,
        iris_shift_per_yaw_deg: float = 0.0027,
    ):
        """`camera_height_m`/`camera_pitch_deg` are placeholders, not a
        real calibration - this project has only ever run against recorded
        phone video with no known camera mount, so the phase-2 geometric
        decision (docs/context/gaze.md) is physically meaningful only once
        these (and `robot_radius_m`) are set to the real deployment rig's
        values. `camera_offset_*` default to zero - the robot's center
        exactly at the camera - see `camera_model.CameraRobotOffset` and
        `config/robot_rig.yaml` for how to set these for a camera that
        isn't mounted at the robot's own center."""
        self.estimator = GazeEstimator(weights_path=gaze_weights_path, device=device)
        self.gate = MediaPipeGate(
            model_path=mediapipe_model_path, min_eye_px=min_eye_px, ear_min=ear_min, blur_min=blur_min
        )
        self.mp_min_head_px = mp_min_head_px
        self.max_yaw_gaze_deg = max_yaw_gaze_deg
        self.max_eye_head_deg = max_eye_head_deg
        self.fixation_min_ms = fixation_min_ms
        self.fixation_max_dev_deg = fixation_max_dev_deg
        self.blink_gap_max_ms = blink_gap_max_ms
        self.one_euro_min_cutoff = one_euro_min_cutoff
        self.one_euro_beta = one_euro_beta
        self.camera_height_m = camera_height_m
        self.camera_pitch_deg = camera_pitch_deg
        self.robot_radius_m = robot_radius_m
        self.theta_contact_deg = theta_contact_deg
        self.peripheral_buffer_deg = peripheral_buffer_deg
        self.head_only_extra_deg = head_only_extra_deg
        self.shoulder_width_prior_m = shoulder_width_prior_m
        self.person_height_prior_m = person_height_prior_m
        self.iris_veto_max_offset = iris_veto_max_offset
        self.iris_shift_per_yaw_deg = iris_shift_per_yaw_deg
        self._robot_center_camera_frame = robot_center_camera_frame(
            CameraRobotOffset(camera_offset_right_m, camera_offset_down_m, camera_offset_forward_m)
        )

        self._fixation_trackers: dict[int, FixationTracker] = {}
        self._smoothers: dict[int, OneEuroFilter] = {}
        self._pitch_smoothers: dict[int, OneEuroFilter] = {}
        # Per track: start time of the fixation its two smoothers are following.
        self._smoothed_fixation_start: dict[int, float] = {}

    def _pitch_smoother_for(self, track_id: int) -> OneEuroFilter:
        if track_id not in self._pitch_smoothers:
            self._pitch_smoothers[track_id] = OneEuroFilter(min_cutoff=self.one_euro_min_cutoff, beta=self.one_euro_beta)
        return self._pitch_smoothers[track_id]

    def _tracker_for(self, track_id: int) -> FixationTracker:
        if track_id not in self._fixation_trackers:
            self._fixation_trackers[track_id] = FixationTracker(
                self.fixation_min_ms, self.fixation_max_dev_deg, self.blink_gap_max_ms
            )
        return self._fixation_trackers[track_id]

    def _smoother_for(self, track_id: int) -> OneEuroFilter:
        if track_id not in self._smoothers:
            self._smoothers[track_id] = OneEuroFilter(min_cutoff=self.one_euro_min_cutoff, beta=self.one_euro_beta)
        return self._smoothers[track_id]

    def prune(self, active_track_ids: set[int]) -> None:
        """Drop per-track state for any track the caller no longer
        publishes at all - a track that's merely LOST_KALMAN/LOST_GALLERY
        (still alive in human_tracking, just without a fresh crop this
        frame) should still be in `active_track_ids` and keep its state."""
        for state_dict in (
            self._fixation_trackers, self._smoothers, self._pitch_smoothers, self._smoothed_fixation_start
        ):
            for track_id in list(state_dict.keys()):
                if track_id not in active_track_ids:
                    del state_dict[track_id]

    def _evaluate_geometric(
        self,
        keypoints: list[tuple[float, float, float]],
        bbox_xyxy: tuple[float, float, float, float] | None,
        image_width: float,
        image_height: float | None,
        camera_hfov_deg: float,
        yaw_deg: float,
        pitch_deg: float,
        source: str,
        min_keypoint_conf: float,
    ) -> GeometricGazeResult | None:
        """`None` only when there isn't even enough to build a reference
        pixel/intrinsics (no `image_height`, or neither the nose nor both
        eyes are confident) - every other failure (no distance estimate)
        still returns a real `GeometricGazeResult` with `distance_m=None`,
        so callers can tell "couldn't try" apart from "tried, no distance"."""
        if image_height is None:
            return None
        intrinsics = intrinsics_from_hfov(image_width, image_height, camera_hfov_deg)

        ref_point = None
        if len(keypoints) > _NOSE and keypoints[_NOSE][2] >= min_keypoint_conf:
            ref_point = (keypoints[_NOSE][0], keypoints[_NOSE][1])
        elif (
            len(keypoints) > _RIGHT_EYE
            and keypoints[_LEFT_EYE][2] >= min_keypoint_conf
            and keypoints[_RIGHT_EYE][2] >= min_keypoint_conf
        ):
            ref_point = (
                (keypoints[_LEFT_EYE][0] + keypoints[_RIGHT_EYE][0]) / 2.0,
                (keypoints[_LEFT_EYE][1] + keypoints[_RIGHT_EYE][1]) / 2.0,
            )
        if ref_point is None:
            return None

        bbox_height_px = (bbox_xyxy[3] - bbox_xyxy[1]) if bbox_xyxy is not None else None
        distance_m, distance_method = estimate_distance_m(
            keypoints,
            intrinsics,
            self.camera_height_m,
            self.camera_pitch_deg,
            bbox_height_px=bbox_height_px,
            shoulder_width_prior_m=self.shoulder_width_prior_m,
            person_height_prior_m=self.person_height_prior_m,
            min_keypoint_conf=min_keypoint_conf,
        )
        person_position = (
            person_position_camera_frame(ref_point[0], ref_point[1], distance_m, distance_method, intrinsics)
            if distance_m is not None
            else None
        )
        gaze_vector = gaze_yaw_pitch_to_vector(yaw_deg, pitch_deg)

        return evaluate_geometric(
            gaze_vector,
            person_position,
            self._robot_center_camera_frame,
            distance_method,
            source,
            robot_radius_m=self.robot_radius_m,
            theta_contact_deg=self.theta_contact_deg,
            peripheral_buffer_deg=self.peripheral_buffer_deg,
            head_only_extra_deg=self.head_only_extra_deg,
        )

    def _apply_iris_veto(
        self, result: GazeTrackResult, iris_horiz_pos: float | None, head_yaw_deg: float
    ) -> GazeTrackResult:
        if iris_horiz_pos is None:
            return result
        # Looking at the camera with the head turned shifts the iris off-centre by this much.
        expected_pos = 0.5 + self.iris_shift_per_yaw_deg * head_yaw_deg
        result.iris_offset = abs(iris_horiz_pos - expected_pos)
        # An iris-sourced decision already reflects the eyes; only head-only ones are blind to them.
        if result.source != SOURCE_HEAD or result.iris_offset <= self.iris_veto_max_offset:
            return result
        result.eyes_averted = True
        if result.geometric is not None:
            result.geometric = replace(result.geometric, sees_robot=False, direct_contact=False)
        return result

    def process_track(
        self,
        track_id: int,
        frame: np.ndarray,
        keypoints: list[tuple[float, float, float]],
        head_yaw_deg: float,
        bbox_center_x: float,
        image_width: float,
        camera_hfov_deg: float,
        timestamp_s: float,
        dt_seconds: float,
        min_keypoint_conf: float = 0.3,
        head_k_h: float = 0.8,
        head_scale_alpha: float = 1.7,
        crop_margin: float = 0.2,
        head_pitch_deg: float = 0.0,
        bbox_xyxy: tuple[float, float, float, float] | None = None,
        image_height: float | None = None,
    ) -> GazeTrackResult:
        """`head_yaw_deg` must already be a valid B1 reading - callers
        should only invoke this when `head_pose_valid` is true (see
        human_gaze/gaze_node.py); there is no meaningful fallback angle to
        compute here otherwise. `head_pitch_deg`, `bbox_xyxy`, and
        `image_height` are only needed for the phase-2 geometric decision
        (`GazeTrackResult.geometric`) - omitting them still produces a
        valid flat `gaze_robot_deg`, just no geometric result."""
        bearing = bearing_deg(bbox_center_x, image_width, camera_hfov_deg)
        fixation_tracker = self._tracker_for(track_id)

        def geometric(yaw_deg: float, pitch_deg: float, source: str) -> GeometricGazeResult | None:
            return self._evaluate_geometric(
                keypoints, bbox_xyxy, image_width, image_height, camera_hfov_deg,
                yaw_deg, pitch_deg, source, min_keypoint_conf,
            )

        if abs(head_yaw_deg) >= self.max_yaw_gaze_deg:
            fixation_tracker.update(None, timestamp_s)
            fused = fuse(head_yaw_deg, bearing, None, False, self.max_eye_head_deg)
            return GazeTrackResult(
                fused.gaze_robot_deg, fused.source, False, False, "head turned away from gaze range",
                geometric=geometric(head_yaw_deg, head_pitch_deg, fused.source),
            )

        head_crop, head_crop_bbox = extract_head_crop(
            frame, keypoints, min_keypoint_conf, head_k_h, head_scale_alpha, crop_margin, self.mp_min_head_px
        )
        if head_crop is None:
            fixation_tracker.update(None, timestamp_s)
            fused = fuse(head_yaw_deg, bearing, None, False, self.max_eye_head_deg)
            return GazeTrackResult(
                fused.gaze_robot_deg, fused.source, False, False, "head crop too small for MediaPipe",
                geometric=geometric(head_yaw_deg, head_pitch_deg, fused.source),
            )

        gate_result = self.gate.process(head_crop)
        if not gate_result.passed:
            fixation_tracker.update(None, timestamp_s)
            fused = fuse(head_yaw_deg, bearing, None, False, self.max_eye_head_deg)
            return self._apply_iris_veto(
                GazeTrackResult(
                    fused.gaze_robot_deg, fused.source, False, False, gate_result.reason,
                    geometric=geometric(head_yaw_deg, head_pitch_deg, fused.source),
                ),
                gate_result.iris_horiz_pos, head_yaw_deg,
            )

        face_bbox_frame = crop_relative_to_frame_xyxy(gate_result.face_bbox_xyxy, head_crop_bbox)
        face_crop = extract_face_crop(frame, face_bbox_frame)
        gaze_result = self.estimator.estimate([face_crop])[0]
        if gaze_result is None:
            fixation_tracker.update(None, timestamp_s)
            fused = fuse(head_yaw_deg, bearing, None, False, self.max_eye_head_deg)
            return self._apply_iris_veto(
                GazeTrackResult(
                    fused.gaze_robot_deg, fused.source, False, False, "empty face crop",
                    geometric=geometric(head_yaw_deg, head_pitch_deg, fused.source),
                ),
                gate_result.iris_horiz_pos, head_yaw_deg,
            )

        iris_yaw_deg, iris_pitch_deg = gaze_result
        # Raw (pre-smoothing) angle feeds fixation detection, so a real
        # saccade is still seen as real movement rather than pre-blunted by
        # the One Euro filter - see module docstring.
        fixation_state = fixation_tracker.update(iris_yaw_deg, timestamp_s)

        filtered_iris_yaw_deg = None
        filtered_iris_pitch_deg = 0.0
        if fixation_state.is_fixating:
            if self._smoothed_fixation_start.get(track_id) != fixation_state.start_ts:
                # A new fixation (after a saccade or a gap) gets fresh filters
                # and starts from its own angle. The old ones would pull it
                # toward the previous fixation's - most of all after a gap,
                # since dt_seconds is the frame interval, not the time since
                # they last ran.
                self._smoothers.pop(track_id, None)
                self._pitch_smoothers.pop(track_id, None)
                self._smoothed_fixation_start[track_id] = fixation_state.start_ts
            filtered_iris_yaw_deg = self._smoother_for(track_id).filter(fixation_state.angle_deg, dt_seconds)
            # Pitch isn't part of the fixation decision itself (that's
            # yaw-stability only, matching the design doc) - it rides along,
            # smoothed the same way, whenever a fixation is already confirmed.
            filtered_iris_pitch_deg = self._pitch_smoother_for(track_id).filter(iris_pitch_deg, dt_seconds)

        fused = fuse(head_yaw_deg, bearing, filtered_iris_yaw_deg, fixation_state.is_fixating, self.max_eye_head_deg)
        reason = "ok" if fused.source != SOURCE_HEAD else "iris not yet fixating or implausible vs head"
        geometric_yaw, geometric_pitch = (
            (filtered_iris_yaw_deg, filtered_iris_pitch_deg) if fused.source != SOURCE_HEAD
            else (head_yaw_deg, head_pitch_deg)
        )
        return self._apply_iris_veto(
            GazeTrackResult(
                fused.gaze_robot_deg,
                fused.source,
                True,
                fixation_state.is_fixating,
                reason,
                filtered_iris_yaw_deg if filtered_iris_yaw_deg is not None else 0.0,
                face_bbox_frame,
                geometric=geometric(geometric_yaw, geometric_pitch, fused.source),
            ),
            gate_result.iris_horiz_pos, head_yaw_deg,
        )
