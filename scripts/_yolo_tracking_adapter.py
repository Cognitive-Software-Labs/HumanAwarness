"""Shared helper for scripts that run human_tracking directly on a YOLO pose
stream, no ROS involved - extracted because test_human_tracking_on_video.py
and test_tracking_and_head_pose_on_video.py both needed the exact same
per-frame bundling (YOLO result -> crops/torso-crops -> ReID embeddings ->
Detection list), previously duplicated between them.
"""

from human_tracking.reid_embedder import ReIDEmbedder
from human_tracking.track import torso_crop_bbox
from human_tracking.tracker import Detection


def yolo_result_to_detections(result, embedder: ReIDEmbedder) -> list[Detection]:
    """Adapts one Ultralytics YOLO pose result into `human_tracking.Detection`
    objects (bbox + keypoints + full-crop/torso-crop ReID embeddings), ready
    for `Tracker.update()`."""
    frame = result.orig_img
    boxes, keypoints = result.boxes, result.keypoints

    crops, torso_crops, dets_meta = [], [], []
    if boxes is not None:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            crops.append(frame[max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)])
            if keypoints is not None:
                kp_xy = keypoints.xy[i].tolist()
                kp_conf = keypoints.conf[i].tolist() if keypoints.conf is not None else [1.0] * len(kp_xy)
                kp = [(x, y, c) for (x, y), c in zip(kp_xy, kp_conf)]
            else:
                kp = []
            dets_meta.append(((x1, y1, x2, y2), float(boxes.conf[i]), kp))

            torso_bbox = torso_crop_bbox(kp, (x1, y1, x2, y2)) if kp else None
            if torso_bbox is not None:
                tx1, ty1, tx2, ty2 = torso_bbox
                torso_crops.append(frame[max(0, int(ty1)) : int(ty2), max(0, int(tx1)) : int(tx2)])
            else:
                torso_crops.append(None)

    # Embed full + torso crops in one batched call (embed() already treats a
    # None entry as "no embedding for this one").
    n = len(crops)
    all_embeddings = embedder.embed(crops + torso_crops) if crops else []
    embeddings, torso_embeddings = all_embeddings[:n], all_embeddings[n:]

    return [
        Detection(bbox_xyxy=bbox, confidence=conf, keypoints=kp, embedding=emb, torso_embedding=temb)
        for (bbox, conf, kp), emb, temb in zip(dets_meta, embeddings, torso_embeddings)
    ]
