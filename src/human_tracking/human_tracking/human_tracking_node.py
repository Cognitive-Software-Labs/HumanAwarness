"""ROS2 node: persistent person identity across occlusion.

Subscribes to a raw camera image and human_perception's per-frame
PersonDetectionArray (time-synchronized, since ReID needs actual pixel
crops that the detections message alone doesn't carry), runs them through
the same Tracker/ReIDEmbedder used by scripts/test_human_tracking_on_video.py,
and republishes a PersonDetectionArray - reusing the existing message type
rather than introducing a parallel one, since PersonDetection already has
`track_id` and a `state` field is all tracking adds - with track_id and
state populated for every currently-confirmed identity (ACTIVE or a still-
recoverable LOST_KALMAN/LOST_GALLERY; see docs/context/human_tracking.md
for why NEW/FORGOTTEN are never published).
"""

import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from human_awareness_msgs.msg import Keypoint2D, PersonDetection, PersonDetectionArray
from human_tracking.reid_embedder import ReIDEmbedder
from human_tracking.track import torso_crop_bbox
from human_tracking.tracker import Detection, Tracker

# Same list, same order, as human_perception's bbox_keypoints_node.py - this
# is the COCO-17 wire-format contract both packages agree on (how `keypoints`
# is indexed matters here, e.g. for torso_crop_bbox), not an implementation
# detail private to either package, so it's kept in sync by convention
# rather than imported across packages.
COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _stamp_to_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class HumanTrackingNode(Node):
    def __init__(self):
        super().__init__("human_tracking_node")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "human_perception/detections")
        self.declare_parameter("tracked_topic", "human_tracking/tracked_detections")
        self.declare_parameter("sync_slop_s", 0.05)
        self.declare_parameter("reid_weights_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("gallery_size", 8)
        self.declare_parameter("iou_threshold", 0.3)
        self.declare_parameter("reid_threshold", 0.65)
        self.declare_parameter("reid_margin", 0.08)
        self.declare_parameter("max_speed_bbox_heights_per_sec", 3.0)
        self.declare_parameter("confirm_frames", 3)
        self.declare_parameter("kalman_timeout_s", 2.5)
        self.declare_parameter("gallery_timeout_s", 45.0)
        self.declare_parameter("reconcile_window_s", 1.5)

        image_topic = self.get_parameter("image_topic").value
        detections_topic = self.get_parameter("detections_topic").value
        tracked_topic = self.get_parameter("tracked_topic").value
        sync_slop_s = self.get_parameter("sync_slop_s").value
        reid_weights_path = self.get_parameter("reid_weights_path").value or None
        device = self.get_parameter("device").value

        self.get_logger().info(f"Loading OSNet ReID embedder on device '{device}'")
        self.embedder = ReIDEmbedder(weights_path=reid_weights_path, device=device)

        self.tracker = Tracker(
            gallery_size=self.get_parameter("gallery_size").value,
            iou_threshold=self.get_parameter("iou_threshold").value,
            reid_threshold=self.get_parameter("reid_threshold").value,
            reid_margin=self.get_parameter("reid_margin").value,
            max_speed_bbox_heights_per_sec=self.get_parameter("max_speed_bbox_heights_per_sec").value,
            confirm_frames=self.get_parameter("confirm_frames").value,
            kalman_timeout_s=self.get_parameter("kalman_timeout_s").value,
            gallery_timeout_s=self.get_parameter("gallery_timeout_s").value,
            reconcile_window_s=self.get_parameter("reconcile_window_s").value,
        )

        self.bridge = CvBridge()
        self._last_stamp_s: float | None = None

        self.tracked_pub = self.create_publisher(PersonDetectionArray, tracked_topic, 10)

        image_sub = message_filters.Subscriber(self, Image, image_topic)
        detections_sub = message_filters.Subscriber(self, PersonDetectionArray, detections_topic)
        # Approximate, not exact: a detections message's header stamp is
        # copied from the image it was computed from (see bbox_keypoints_node),
        # so timestamps should match closely, but allow a small slop for
        # clock/serialization jitter between the two separate topics rather
        # than silently dropping every pair that isn't bit-for-bit aligned.
        # Keeping this as self._sync (not a local variable) matters: it's
        # the only thing keeping the two message_filters.Subscriber objects
        # (and the real rclpy subscriptions they wrap) alive - letting it
        # get garbage-collected silently drops both subscriptions.
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, detections_sub], queue_size=10, slop=sync_slop_s
        )
        self._sync.registerCallback(self._on_synced_pair)

        self.get_logger().info(
            f"Subscribed to '{image_topic}' + '{detections_topic}', publishing tracked identities on '{tracked_topic}'"
        )

    def _on_synced_pair(self, image_msg: Image, detections_msg: PersonDetectionArray) -> None:
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        frame_area = frame.shape[0] * frame.shape[1]

        stamp_s = _stamp_to_seconds(image_msg.header.stamp)
        dt_seconds = (stamp_s - self._last_stamp_s) if self._last_stamp_s is not None else (1.0 / 30.0)
        self._last_stamp_s = stamp_s
        if dt_seconds <= 0.0:
            # Out-of-order or duplicate stamp - still advance the tracker
            # with a small nominal step rather than feeding it a zero/
            # negative dt, which would break the spatial-gate and timeout
            # math elsewhere (both assume time moves forward).
            dt_seconds = 1.0 / 30.0

        crops, torso_crops, dets_meta = [], [], []
        for det in detections_msg.detections:
            x1, y1 = det.bbox_x, det.bbox_y
            x2, y2 = x1 + det.bbox_width, y1 + det.bbox_height
            bbox = (x1, y1, x2, y2)
            keypoints = [(kp.x, kp.y, kp.confidence) for kp in det.keypoints]
            dets_meta.append((bbox, det.confidence, keypoints))

            crops.append(frame[max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)])
            torso_bbox = torso_crop_bbox(keypoints, bbox) if keypoints else None
            if torso_bbox is not None:
                tx1, ty1, tx2, ty2 = torso_bbox
                torso_crops.append(frame[max(0, int(ty1)) : int(ty2), max(0, int(tx1)) : int(tx2)])
            else:
                torso_crops.append(None)

        n = len(crops)
        all_embeddings = self.embedder.embed(crops + torso_crops) if crops else []
        embeddings, torso_embeddings = all_embeddings[:n], all_embeddings[n:]

        detections = [
            Detection(bbox_xyxy=bbox, confidence=conf, keypoints=kp, embedding=emb, torso_embedding=temb)
            for (bbox, conf, kp), emb, temb in zip(dets_meta, embeddings, torso_embeddings)
        ]

        published = self.tracker.update(detections, frame_area=frame_area, dt_seconds=dt_seconds)

        out = PersonDetectionArray()
        out.header = image_msg.header
        for t in published:
            detection = PersonDetection()
            detection.track_id = t.track_id
            detection.state = t.state
            detection.confidence = t.last_confidence

            x1, y1, x2, y2 = t.kalman.bbox
            detection.bbox_x = float(x1)
            detection.bbox_y = float(y1)
            detection.bbox_width = float(x2 - x1)
            detection.bbox_height = float(y2 - y1)

            for i, (kx, ky, kconf) in enumerate(t.last_keypoints):
                kp = Keypoint2D()
                kp.name = COCO_KEYPOINT_NAMES[i] if i < len(COCO_KEYPOINT_NAMES) else f"kp_{i}"
                kp.x = float(kx)
                kp.y = float(ky)
                kp.confidence = float(kconf)
                detection.keypoints.append(kp)

            out.detections.append(detection)

        self.tracked_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HumanTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
