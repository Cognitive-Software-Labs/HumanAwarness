"""ROS2 node: head pose relative to the robot, per tracked person.

Subscribes to a raw camera image and human_tracking's (or human_perception's)
PersonDetectionArray, time-synchronized since head pose needs actual pixel
crops the detections message alone doesn't carry - the same pattern
human_tracking_node already uses for the same reason. Republishes the same
PersonDetectionArray with its head-pose fields filled in (see
human_awareness_msgs/PersonDetection.msg) - reusing the existing message
again rather than introducing a parallel type, consistent with how
human_tracking added track_id/state onto it instead of a new message.

Only ACTIVE tracks and untracked detections get a head pose. LOST_KALMAN/
LOST_GALLERY tracks carry their last match's keypoints, not this frame's, so
they pass through unchanged with head_pose_valid=False.
"""

import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from human_awareness_msgs.msg import PersonDetectionArray
from human_head_pose.adapters import extract_head_crop, has_fresh_keypoints, person_detection_to_bbox_keypoints
from human_head_pose.geometry import bearing_deg, gaze_robot_deg
from human_head_pose.head_pose_estimator import HeadPoseEstimator


class HeadPoseNode(Node):
    def __init__(self):
        super().__init__("head_pose_node")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "human_tracking/tracked_detections")
        self.declare_parameter("head_poses_topic", "human_head_pose/head_poses")
        self.declare_parameter("sync_slop_s", 0.05)
        self.declare_parameter("weights_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("min_keypoint_conf", 0.3)
        self.declare_parameter("camera_hfov_deg", 70.0)
        self.declare_parameter("head_k_h", 0.8)
        self.declare_parameter("head_scale_alpha", 1.7)
        self.declare_parameter("crop_margin", 0.2)
        self.declare_parameter("min_head_px", 35.0)

        image_topic = self.get_parameter("image_topic").value
        detections_topic = self.get_parameter("detections_topic").value
        head_poses_topic = self.get_parameter("head_poses_topic").value
        sync_slop_s = self.get_parameter("sync_slop_s").value
        weights_path = self.get_parameter("weights_path").value or None
        device = self.get_parameter("device").value
        self.min_keypoint_conf = self.get_parameter("min_keypoint_conf").value
        self.camera_hfov_deg = self.get_parameter("camera_hfov_deg").value
        self.head_k_h = self.get_parameter("head_k_h").value
        self.head_scale_alpha = self.get_parameter("head_scale_alpha").value
        self.crop_margin = self.get_parameter("crop_margin").value
        self.min_head_px = self.get_parameter("min_head_px").value

        self.get_logger().info(f"Loading 6DRepNet360 head pose estimator on device '{device}'")
        self.estimator = HeadPoseEstimator(weights_path=weights_path, device=device)

        self.bridge = CvBridge()
        self.head_poses_pub = self.create_publisher(PersonDetectionArray, head_poses_topic, 10)

        image_sub = message_filters.Subscriber(self, Image, image_topic)
        detections_sub = message_filters.Subscriber(self, PersonDetectionArray, detections_topic)
        # Kept as self._sync, not a local variable, for the same reason
        # human_tracking_node does: letting it get garbage-collected silently
        # drops both subscriptions.
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, detections_sub], queue_size=10, slop=sync_slop_s
        )
        self._sync.registerCallback(self._on_synced_pair)

        self.get_logger().info(
            f"Subscribed to '{image_topic}' + '{detections_topic}', publishing head poses on '{head_poses_topic}'"
        )

    def _on_synced_pair(self, image_msg: Image, detections_msg: PersonDetectionArray) -> None:
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        image_width = frame.shape[1]

        bbox_keypoints = [person_detection_to_bbox_keypoints(det) for det in detections_msg.detections]
        # No crop for a LOST_* track, so no pose and head_pose_valid=False
        # below: its keypoints aren't from this frame (see has_fresh_keypoints).
        head_crops = [
            extract_head_crop(
                frame,
                keypoints,
                self.min_keypoint_conf,
                self.head_k_h,
                self.head_scale_alpha,
                self.crop_margin,
                self.min_head_px,
            )[0]
            if has_fresh_keypoints(det)
            else None
            for det, (_bbox_xyxy, keypoints) in zip(detections_msg.detections, bbox_keypoints)
        ]
        poses = self.estimator.estimate(head_crops)

        out = PersonDetectionArray()
        out.header = image_msg.header
        for det, (bbox_xyxy, _keypoints), pose in zip(detections_msg.detections, bbox_keypoints, poses):
            det.head_pose_valid = pose is not None
            if pose is not None:
                pitch, yaw, roll = pose
                x1, _y1, x2, _y2 = bbox_xyxy
                bearing = bearing_deg((x1 + x2) / 2.0, image_width, self.camera_hfov_deg)
                det.head_pitch_deg = pitch
                det.head_yaw_deg = yaw
                det.head_roll_deg = roll
                det.bearing_deg = bearing
                det.gaze_robot_deg = gaze_robot_deg(yaw, bearing)
            out.detections.append(det)

        self.head_poses_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HeadPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
