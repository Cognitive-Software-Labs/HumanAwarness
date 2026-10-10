# One detected person: bounding box + pose keypoints, in image pixel coordinates.

int32 track_id                      # -1 if not tracked
float32 confidence                  # overall detection confidence [0.0, 1.0]

# Bounding box, top-left origin, pixel coordinates.
float32 bbox_x
float32 bbox_y
float32 bbox_width
float32 bbox_height

human_awareness_msgs/Keypoint2D[] keypoints   # COCO-17 order
