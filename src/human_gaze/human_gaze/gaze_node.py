"""ROS2 node: iris/gaze estimation per tracked person, fused with B1's head
pose into a single "facing the robot" angle - "MediaPipe pregateste, L2CS-
Net estimeaza" (docs/context/gaze.md).

Subscribes to a raw camera image and human_head_pose's PersonDetectionArray
(time-synchronized, same pattern as human_head_pose_node and
human_tracking_node before it - head pose must already be valid for a
detection before gaze is attempted at all, and like head_pose_node it only
looks at ACTIVE tracks and untracked detections: a LOST_KALMAN/LOST_GALLERY
track passes through with head_pose_valid=False and its head-pose fields
zeroed, whatever head pose arrived with it). Republishes the same message
type again: `gaze_robot_deg` is overwritten with the fused angle whenever
an iris reading is trusted this frame, and the new iris_* fields record
which source actually won - see human_awareness_msgs/PersonDetection.msg.

Per-track fixation/smoothing state lives inside `GazePipeline`, keyed by
`track_id` - `dt_seconds` comes from consecutive image header stamps (not
an assumed fixed frame rate), same reasoning as human_tracking_node: the
One Euro filter's cutoff math assumes real elapsed time, which can vary
with real message-passing overhead.
"""

import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from human_awareness_msgs.msg import PersonDetectionArray
from human_gaze.gaze_pipeline import GazePipeline
from human_head_pose.adapters import has_fresh_keypoints, person_detection_to_bbox_keypoints


def _stamp_to_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class GazeNode(Node):
    def __init__(self):
        super().__init__("gaze_node")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "human_head_pose/head_poses")
        self.declare_parameter("gaze_topic", "human_gaze/gazes")
        self.declare_parameter("sync_slop_s", 0.05)
        self.declare_parameter("gaze_weights_path", "")
        self.declare_parameter("mediapipe_model_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("min_keypoint_conf", 0.3)
        self.declare_parameter("camera_hfov_deg", 70.0)
        self.declare_parameter("head_k_h", 0.8)
        self.declare_parameter("head_scale_alpha", 1.7)
        self.declare_parameter("crop_margin", 0.2)
        self.declare_parameter("mp_min_head_px", 80.0)
        self.declare_parameter("min_eye_px", 12.0)
        self.declare_parameter("max_yaw_gaze_deg", 45.0)
        self.declare_parameter("max_eye_head_deg", 30.0)
        self.declare_parameter("ear_min", 0.2)
        self.declare_parameter("blur_min", 0.0)
        self.declare_parameter("fixation_min_ms", 300.0)
        self.declare_parameter("fixation_max_dev_deg", 4.0)
        self.declare_parameter("blink_gap_max_ms", 300.0)
        self.declare_parameter("one_euro_min_cutoff", 1.0)
        self.declare_parameter("one_euro_beta", 0.3)
        self.declare_parameter("camera_height_m", 1.2)
        self.declare_parameter("camera_pitch_deg", 0.0)
        self.declare_parameter("robot_radius_m", 0.3)
        self.declare_parameter("theta_contact_deg", 12.0)
        self.declare_parameter("peripheral_buffer_deg", 5.0)
        self.declare_parameter("head_only_extra_deg", 10.0)
        self.declare_parameter("shoulder_width_prior_m", 0.40)
        self.declare_parameter("person_height_prior_m", 1.70)
        self.declare_parameter("camera_offset_right_m", 0.0)
        self.declare_parameter("camera_offset_down_m", 0.0)
        self.declare_parameter("camera_offset_forward_m", 0.0)
        self.declare_parameter("iris_veto_max_offset", 0.12)
        self.declare_parameter("iris_shift_per_yaw_deg", 0.0027)

        image_topic = self.get_parameter("image_topic").value
        detections_topic = self.get_parameter("detections_topic").value
        gaze_topic = self.get_parameter("gaze_topic").value
        sync_slop_s = self.get_parameter("sync_slop_s").value
        device = self.get_parameter("device").value
        self.min_keypoint_conf = self.get_parameter("min_keypoint_conf").value
        self.camera_hfov_deg = self.get_parameter("camera_hfov_deg").value
        self.head_k_h = self.get_parameter("head_k_h").value
        self.head_scale_alpha = self.get_parameter("head_scale_alpha").value
        self.crop_margin = self.get_parameter("crop_margin").value

        self.get_logger().info(f"Loading MediaPipe Face Landmarker + MobileGaze on device '{device}'")
        self.pipeline = GazePipeline(
            gaze_weights_path=self.get_parameter("gaze_weights_path").value or None,
            mediapipe_model_path=self.get_parameter("mediapipe_model_path").value or None,
            device=device,
            mp_min_head_px=self.get_parameter("mp_min_head_px").value,
            min_eye_px=self.get_parameter("min_eye_px").value,
            max_yaw_gaze_deg=self.get_parameter("max_yaw_gaze_deg").value,
            max_eye_head_deg=self.get_parameter("max_eye_head_deg").value,
            ear_min=self.get_parameter("ear_min").value,
            blur_min=self.get_parameter("blur_min").value,
            fixation_min_ms=self.get_parameter("fixation_min_ms").value,
            fixation_max_dev_deg=self.get_parameter("fixation_max_dev_deg").value,
            blink_gap_max_ms=self.get_parameter("blink_gap_max_ms").value,
            one_euro_min_cutoff=self.get_parameter("one_euro_min_cutoff").value,
            one_euro_beta=self.get_parameter("one_euro_beta").value,
            camera_height_m=self.get_parameter("camera_height_m").value,
            camera_pitch_deg=self.get_parameter("camera_pitch_deg").value,
            robot_radius_m=self.get_parameter("robot_radius_m").value,
            theta_contact_deg=self.get_parameter("theta_contact_deg").value,
            peripheral_buffer_deg=self.get_parameter("peripheral_buffer_deg").value,
            head_only_extra_deg=self.get_parameter("head_only_extra_deg").value,
            shoulder_width_prior_m=self.get_parameter("shoulder_width_prior_m").value,
            person_height_prior_m=self.get_parameter("person_height_prior_m").value,
            camera_offset_right_m=self.get_parameter("camera_offset_right_m").value,
            camera_offset_down_m=self.get_parameter("camera_offset_down_m").value,
            camera_offset_forward_m=self.get_parameter("camera_offset_forward_m").value,
            iris_veto_max_offset=self.get_parameter("iris_veto_max_offset").value,
            iris_shift_per_yaw_deg=self.get_parameter("iris_shift_per_yaw_deg").value,
        )

        self.bridge = CvBridge()
        self._last_stamp_s: float | None = None
        self.gaze_pub = self.create_publisher(PersonDetectionArray, gaze_topic, 10)

        image_sub = message_filters.Subscriber(self, Image, image_topic)
        detections_sub = message_filters.Subscriber(self, PersonDetectionArray, detections_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, detections_sub], queue_size=10, slop=sync_slop_s
        )
        self._sync.registerCallback(self._on_synced_pair)

        self.get_logger().info(
            f"Subscribed to '{image_topic}' + '{detections_topic}', publishing gaze on '{gaze_topic}'"
        )

    def _on_synced_pair(self, image_msg: Image, detections_msg: PersonDetectionArray) -> None:
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        image_height, image_width = frame.shape[:2]

        stamp_s = _stamp_to_seconds(image_msg.header.stamp)
        dt_seconds = (stamp_s - self._last_stamp_s) if self._last_stamp_s is not None else (1.0 / 30.0)
        self._last_stamp_s = stamp_s
        if dt_seconds <= 0.0:
            dt_seconds = 1.0 / 30.0  # out-of-order/duplicate stamp - see human_tracking_node for the same guard

        out = PersonDetectionArray()
        out.header = image_msg.header
        active_track_ids = set()

        for det in detections_msg.detections:
            # LOST tracks too, so they keep their fixation/smoothing state
            # (see GazePipeline.prune), even though they're skipped below.
            if det.track_id >= 0:
                active_track_ids.add(det.track_id)

            # A LOST_* track's keypoints aren't from this frame (see
            # has_fresh_keypoints): no gaze from them, and no head pose
            # either, whatever the input carries (it may be a bag, or come
            # from something other than head_pose_node). Cleared to zero, as
            # PersonDetection.msg defines an invalid head pose.
            if not has_fresh_keypoints(det):
                det.head_pose_valid = False
                det.head_yaw_deg = det.head_pitch_deg = det.head_roll_deg = 0.0
                det.bearing_deg = det.gaze_robot_deg = 0.0
            if not det.head_pose_valid:
                out.detections.append(det)
                continue

            bbox_xyxy, keypoints = person_detection_to_bbox_keypoints(det)
            x1, _y1, x2, _y2 = bbox_xyxy
            result = self.pipeline.process_track(
                track_id=det.track_id,
                frame=frame,
                keypoints=keypoints,
                head_yaw_deg=det.head_yaw_deg,
                bbox_center_x=(x1 + x2) / 2.0,
                image_width=image_width,
                camera_hfov_deg=self.camera_hfov_deg,
                timestamp_s=stamp_s,
                dt_seconds=dt_seconds,
                min_keypoint_conf=self.min_keypoint_conf,
                head_k_h=self.head_k_h,
                head_scale_alpha=self.head_scale_alpha,
                crop_margin=self.crop_margin,
                head_pitch_deg=det.head_pitch_deg,
                bbox_xyxy=bbox_xyxy,
                image_height=image_height,
            )
            det.gaze_robot_deg = result.gaze_robot_deg
            det.gaze_source = result.source
            det.iris_valid = result.iris_valid
            det.iris_fixating = result.iris_fixating
            det.iris_yaw_deg = result.iris_yaw_deg
            det.gaze_gate_reason = result.gate_reason
            det.iris_offset = result.iris_offset if result.iris_offset is not None else -1.0
            det.eyes_averted = result.eyes_averted
            if result.geometric is not None:
                g = result.geometric
                det.distance_m = g.distance_m if g.distance_m is not None else 0.0
                det.distance_method = g.distance_method
                det.sees_robot = g.sees_robot
                det.direct_contact = g.direct_contact
                det.theta_to_robot_deg = g.theta_to_robot_deg if g.theta_to_robot_deg is not None else 0.0
            out.detections.append(det)

        self.pipeline.prune(active_track_ids)
        self.gaze_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = GazeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
