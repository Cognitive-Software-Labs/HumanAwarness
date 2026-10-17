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
# no head crop could be extracted this frame, e.g. shoulders not confident).
bool head_pose_valid
float32 head_yaw_deg     # 6DRepNet360 output, degrees, full -180..180 range
float32 head_pitch_deg
float32 head_roll_deg
float32 bearing_deg      # this person's azimuth from the camera's optical axis
float32 gaze_robot_deg   # head_yaw_deg - bearing_deg, wrapped to -180..180:
                          # ~0 means the head is pointed back at the robot,
                          # ~180 means facing directly away from it
