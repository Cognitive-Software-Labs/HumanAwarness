# A single 2D body keypoint in image pixel coordinates.

string name          # e.g. "nose", "left_shoulder" (COCO-17 naming)
float32 x
float32 y
float32 confidence   # [0.0, 1.0], per-keypoint detector confidence
