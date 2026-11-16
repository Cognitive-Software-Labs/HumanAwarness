"""PersonDetection -> PersonOverlay, and the image buffer that pairs each
detections message with the exact camera frame it was computed on. No ROS
imports: messages are only read by attribute, so tests use SimpleNamespace.
"""

from collections import deque

from human_visualization.overlay import PersonOverlay

ACTIVE = "ACTIVE"
LOST_KALMAN = "LOST_KALMAN"
LOST_GALLERY = "LOST_GALLERY"

# `distance_method` when the phase-2 decision didn't run ("") or found no
# distance ("none") - same rule as human_awareness.adapters.
_NO_DECISION = ("", "none")


def is_looking(det, aware_threshold_deg: float) -> bool:
    """The same label the full-chain video script draws: the phase-2
    `sees_robot` (iris veto already applied by gaze_node) when it ran, else
    the flat phase-1 `|gaze_robot_deg| <= aware_threshold_deg` fallback."""
    if not det.head_pose_valid:
        return False
    if det.distance_method not in _NO_DECISION:
        return bool(det.sees_robot)
    return abs(det.gaze_robot_deg) <= aware_threshold_deg


def person_detection_to_overlay(
    det, aware_threshold_deg: float = 45.0, show_gaze: bool = True, show_awareness: bool = True
) -> PersonOverlay | None:
    """`None` for a LOST_GALLERY track: it isn't in the frame, it's only kept
    for re-identification, so there's nothing to draw."""
    if det.state == LOST_GALLERY:
        return None
    x1, y1 = det.bbox_x, det.bbox_y
    tracked = det.track_id >= 0
    return PersonOverlay(
        bbox_xyxy=(x1, y1, x1 + det.bbox_width, y1 + det.bbox_height),
        track_id=det.track_id,
        predicted=det.state == LOST_KALMAN,
        head_pose_valid=bool(det.head_pose_valid) if show_gaze and tracked else None,
        is_looking=is_looking(det, aware_threshold_deg),
        awareness=(det.awareness_value, det.awareness_confidence) if show_awareness and tracked else None,
    )


def stamp_to_ns(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class ImageBuffer:
    """The last `maxlen` image messages, looked up by exact header stamp.

    Every node in the chain copies the image's header onto its output, so a
    detections message has the same stamp as its frame, but arrives later by
    the whole pipeline's processing time. The buffer has to cover that delay
    (at 30 fps, 60 frames = 2 s).
    """

    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self._items: deque[tuple[int, object]] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, stamp_ns: int, msg) -> None:
        self._items.append((stamp_ns, msg))

    def oldest_ns(self) -> int | None:
        return self._items[0][0] if self._items else None

    def newest(self):
        return self._items[-1][1] if self._items else None

    def pop_match(self, stamp_ns: int):
        """The image with exactly this stamp, or None. Images older than the
        match are dropped too: detections arrive in order, so they can't be
        asked for again."""
        for i, (s, msg) in enumerate(self._items):
            if s == stamp_ns:
                for _ in range(i + 1):
                    self._items.popleft()
                return msg
            if s > stamp_ns:
                break
        return None
