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
