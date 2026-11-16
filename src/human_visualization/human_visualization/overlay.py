"""Draws the per-person overlay (bbox, track id, head/face crops, looking
label, awareness bar) onto a BGR frame. Pure OpenCV, no ROS imports, so the
offline video script and the live visualization node draw the same thing.
"""

from dataclasses import dataclass

import cv2

ACTIVE_COLOR = (0, 200, 0)
PREDICTED_COLOR = (0, 165, 255)
UNTRACKED_COLOR = (200, 200, 0)
HEAD_CROP_COLOR = (255, 165, 0)
FACE_CROP_COLOR = (255, 0, 255)
LOOKING_COLOR = (0, 220, 0)
NOT_LOOKING_COLOR = (0, 0, 230)
NO_HEAD_POSE_COLOR = (160, 160, 160)
AWARENESS_BAR_BG_COLOR = (90, 90, 90)
INFO_COLOR = (255, 255, 255)
INFO_BG_COLOR = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Text rows above the bbox, bottom to top.
_ID_ROW_PX = 10
_HEAD_POSE_ROW_PX = 34
_LOOKING_ROW_PX = 58
_AWARENESS_ROW_PX = 82
_MIN_TEXT_Y = 15


@dataclass
class PersonOverlay:
    """What to draw for one person. `head_pose_valid=None` means the gaze
    stage didn't run for this input, so no looking label is drawn at all;
    `awareness=None` likewise skips the awareness bar."""

    bbox_xyxy: tuple[float, float, float, float]
    track_id: int = -1  # -1: untracked detection
    predicted: bool = False  # LOST_KALMAN: box is the Kalman prediction, not a detection
    head_crop_bbox: tuple[float, float, float, float] | None = None
    face_bbox: tuple[float, float, float, float] | None = None
    head_pose_valid: bool | None = None
    is_looking: bool = False
    eyes_averted: bool = False
    awareness: tuple[float, float] | None = None  # (value, confidence)


def awareness_color(value: float) -> tuple[int, int, int]:
    """Red at 0 through green at 1 (BGR)."""
    return 0, int(220 * value), int(230 * (1.0 - value))


def draw_awareness(image, value: float, confidence: float, x: int, y: int) -> None:
    text = f"aware {value:.2f} conf {confidence:.2f}"
    color = awareness_color(value)
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    bar_w = 80
    x = max(0, min(x, image.shape[1] - (text_w + 8 + bar_w)))  # keep the whole label inside the frame
    cv2.putText(image, text, (x, y), FONT, 0.6, color, 2)
    bar_x = x + text_w + 8
    cv2.rectangle(image, (bar_x, y - text_h), (bar_x + bar_w, y), AWARENESS_BAR_BG_COLOR, -1)
    cv2.rectangle(image, (bar_x, y - text_h), (bar_x + int(bar_w * value), y), color, -1)


def looking_label(is_looking: bool, eyes_averted: bool) -> tuple[str, tuple[int, int, int]]:
    if is_looking:
        return "LOOKING AT ROBOT", LOOKING_COLOR
    return ("NOT LOOKING (eyes away)" if eyes_averted else "NOT LOOKING"), NOT_LOOKING_COLOR


def _row_y(y1: int, offset_px: int) -> int:
    return max(y1 - offset_px, _MIN_TEXT_Y)


def _put_label(image, text: str, x: int, y: int, scale: float, color) -> None:
    """putText with x clamped so the whole label stays inside the frame -
    a person at the right edge would otherwise lose their id."""
    (text_w, _), _ = cv2.getTextSize(text, FONT, scale, 2)
    x = max(0, min(x, image.shape[1] - text_w))
    cv2.putText(image, text, (x, y), FONT, scale, color, 2)


def _rect(image, bbox, color) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)


def draw_person(image, person: PersonOverlay) -> None:
    x1, y1, _x2, _y2 = (int(v) for v in person.bbox_xyxy)
    if person.track_id < 0:
        box_color, id_label = UNTRACKED_COLOR, "untracked"
    elif person.predicted:
        box_color, id_label = PREDICTED_COLOR, f"id {person.track_id} (predicted)"
    else:
        box_color, id_label = ACTIVE_COLOR, f"id {person.track_id}"
    _rect(image, person.bbox_xyxy, box_color)
    _put_label(image, id_label, x1, _row_y(y1, _ID_ROW_PX), 0.7, box_color)

    if person.head_crop_bbox is not None:
        _rect(image, person.head_crop_bbox, HEAD_CROP_COLOR)
    if person.face_bbox is not None:
        _rect(image, person.face_bbox, FACE_CROP_COLOR)

    if person.head_pose_valid is False:
        _put_label(image, "no head pose", x1, _row_y(y1, _HEAD_POSE_ROW_PX), 0.6, NO_HEAD_POSE_COLOR)
    elif person.head_pose_valid:
        label, color = looking_label(person.is_looking, person.eyes_averted)
        _put_label(image, label, x1, _row_y(y1, _LOOKING_ROW_PX), 0.7, color)

    if person.awareness is not None:
        value, confidence = person.awareness
        draw_awareness(image, value, confidence, x1, _row_y(y1, _AWARENESS_ROW_PX))


def draw_info(image, lines: list[str]) -> None:
    """Small status block in the top-left corner (one text line per entry)."""
    y = 22
    for line in lines:
        (text_w, text_h), baseline = cv2.getTextSize(line, FONT, 0.6, 1)
        cv2.rectangle(image, (4, y - text_h - 4), (12 + text_w, y + baseline), INFO_BG_COLOR, -1)
        cv2.putText(image, line, (8, y), FONT, 0.6, INFO_COLOR, 1)
        y += text_h + baseline + 8
