"""Shared helper for scripts that run human_tracking directly on a YOLO pose
stream, no ROS involved - extracted because test_human_tracking_on_video.py
and test_tracking_and_head_pose_on_video.py both needed the exact same
per-frame bundling (YOLO result -> crops/torso-crops -> ReID embeddings ->
Detection list), previously duplicated between them.
"""

from human_tracking.reid_embedder import ReIDEmbedder
from human_tracking.track import EDGE_SIDES_OFF, torso_crop_bbox, touches_image_edge
from human_tracking.tracker import Detection


def yolo_result_to_detections(
    result,
    embedder: ReIDEmbedder,
    edge_embedding_sides: str = EDGE_SIDES_OFF,
    edge_embedding_margin_px: float = 2.0,
    stats: dict | None = None,
) -> list[Detection]:
    """Adapts one Ultralytics YOLO pose result into `human_tracking.Detection`
    objects (bbox + keypoints + full-crop/torso-crop ReID embeddings), ready
    for `Tracker.update()`.

    A box the tracker's edge-crop gate would discard anyway (same
    `edge_embedding_sides`/`edge_embedding_margin_px` as `Tracker`) is not
    embedded at all - saves the OSNet forward pass. If `stats` is given,
    `stats["crops_embedded"]`/`stats["crops_skipped_edge"]` are incremented."""
    frame = result.orig_img
    frame_size = (frame.shape[1], frame.shape[0])
    boxes, keypoints = result.boxes, result.keypoints

    crops, torso_crops, dets_meta = [], [], []
    if boxes is not None:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            skip = touches_image_edge((x1, y1, x2, y2), frame_size, edge_embedding_margin_px, edge_embedding_sides)
            crops.append(None if skip else frame[max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)])
            if keypoints is not None:
                kp_xy = keypoints.xy[i].tolist()
                kp_conf = keypoints.conf[i].tolist() if keypoints.conf is not None else [1.0] * len(kp_xy)
                kp = [(x, y, c) for (x, y), c in zip(kp_xy, kp_conf)]
            else:
                kp = []
            dets_meta.append(((x1, y1, x2, y2), float(boxes.conf[i]), kp))

            torso_bbox = torso_crop_bbox(kp, (x1, y1, x2, y2)) if (kp and not skip) else None
            if torso_bbox is not None:
                tx1, ty1, tx2, ty2 = torso_bbox
                torso_crops.append(frame[max(0, int(ty1)) : int(ty2), max(0, int(tx1)) : int(tx2)])
            else:
                torso_crops.append(None)

    # Embed full + torso crops in one batched call (embed() already treats a
    # None entry as "no embedding for this one").
    n = len(crops)
    if stats is not None:
        embedded = sum(1 for c in crops + torso_crops if c is not None)
        stats["crops_embedded"] = stats.get("crops_embedded", 0) + embedded
        stats["crops_skipped_edge"] = stats.get("crops_skipped_edge", 0) + sum(1 for c in crops if c is None)
    all_embeddings = embedder.embed(crops + torso_crops) if crops else []
    embeddings, torso_embeddings = all_embeddings[:n], all_embeddings[n:]

    return [
        Detection(bbox_xyxy=bbox, confidence=conf, keypoints=kp, embedding=emb, torso_embedding=temb)
        for (bbox, conf, kp), emb, temb in zip(dets_meta, embeddings, torso_embeddings)
    ]
