# One detected (and optionally tracked) person: bounding box + pose
# keypoints, in image pixel coordinates.

int32 track_id                      # -1 if not tracked
string state                        # "" if not tracked, else the human_tracking
                                     # lifecycle state (ACTIVE/LOST_KALMAN/LOST_GALLERY)
float32 confidence                  # overall detection confidence [0.0, 1.0]

# Bounding box, top-left origin, pixel coordinates.
float32 bbox_x
float32 bbox_y
float32 bbox_width
float32 bbox_height

human_awareness_msgs/Keypoint2D[] keypoints   # COCO-17 order

# Head pose, from human_head_pose (all zero/false if head_pose_valid is false -
# no head crop could be extracted this frame, e.g. shoulders not confident, or
# a LOST_KALMAN/LOST_GALLERY track, whose keypoints are from its last match).
bool head_pose_valid
float32 head_yaw_deg     # 6DRepNet360 output, degrees, full -180..180 range
float32 head_pitch_deg
float32 head_roll_deg
float32 bearing_deg      # this person's azimuth from the camera's optical axis
float32 gaze_robot_deg   # fused "facing the robot" angle, -180..180 (~0 = facing
                          # the robot, ~180 = facing directly away) - head_yaw_deg
                          # minus bearing_deg when no iris reading is trusted this
                          # frame, or the iris-derived angle when gaze_source is
                          # "iris" (see human_gaze/fusion.py); always populated
                          # whenever head_pose_valid is true, same as before
                          # human_gaze existed - gaze_source distinguishes which

# Iris/gaze, from human_gaze (all zero/false, gaze_source "head", if
# iris_valid is false - gating failed this frame, or head is turned too far
# for eyes to be usable; see docs/context/gaze.md). Left at the defaults
# (gaze_source "") when head_pose_valid is false: gaze isn't attempted then.
bool iris_valid           # an iris reading was gated through and estimated this frame
bool iris_fixating        # iris_valid *and* held steady long enough to trust (see fixation.py)
string gaze_source        # "head" or "iris" - which signal produced gaze_robot_deg above
float32 iris_yaw_deg      # fixation-confirmed, filtered iris gaze yaw, degrees - 0 if not iris_fixating

# Phase-2 geometric decision (human_gaze/geometric_decision.py, docs/context/gaze.md):
# replaces gaze_robot_deg's flat threshold with a real cone test around the
# robot's physical size at the person's actual distance - accounting for
# the camera not necessarily being mounted exactly at the robot's own
# center (config/robot_rig.yaml's camera_offset_*). All zero/false if no
# reference pixel (nose or both eyes) was confident enough to even try -
# check iris_valid/head_pose_valid first. camera_height_m/robot_radius_m/
# camera_offset_* etc. are placeholder values until calibrated on the real
# deployment rig - see docs/context/gaze.md before trusting these on real
# footage.
float32 distance_m        # estimated distance from the PERSON to the ROBOT (not the camera), meters; 0 if no estimate was possible
string distance_method    # which tier produced the underlying camera-to-person measurement: "ground_plane" | "shoulder_width" | "height" | "none"
float32 theta_to_robot_deg  # angle between the person's gaze vector and the true direction to the robot
bool sees_robot            # theta_to_robot_deg within the robot's cone + peripheral_buffer_deg (+ head_only_extra_deg if gaze_source is "head")
bool direct_contact        # theta_to_robot_deg within theta_contact_deg - distance-independent, stricter

# Awareness, from human_awareness ("knows the robot is there"): per-track_id
# state accumulated over time from sees_robot above, not a property of this
# frame alone - see docs/context/awareness.md. Filled in for ACTIVE and LOST_*
# tracks alike. The defaults (0.5, 0) are "don't know": where a new track_id
# starts, and what a detection that never went through awareness_node carries.
float32 awareness_value 0.5       # 0..1; rises while the person sees the robot, fades over tau_awareness_s otherwise
float32 awareness_confidence 0.0  # 0..1; how much recent observation backs awareness_value - 0 = none
