"""ROS2 node: monocular RGB human detection (bbox + COCO-17 keypoints) via YOLO26-pose."""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from human_awareness_msgs.msg import Keypoint2D, PersonDetection, PersonDetectionArray

# Keypoint order produced by YOLO-pose models, trained on the COCO-17 skeleton.
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


class BboxKeypointsNode(Node):
    def __init__(self):
        super().__init__("bbox_keypoints_node")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "human_perception/detections")
        self.declare_parameter("model_path", "yolo26n-pose.pt")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("publish_annotated", False)
        self.declare_parameter("annotated_image_topic", "human_perception/annotated_image")

        image_topic = self.get_parameter("image_topic").value
        detections_topic = self.get_parameter("detections_topic").value
        model_path = self.get_parameter("model_path").value
        self.device = self.get_parameter("device").value
        self.confidence_threshold = self.get_parameter("confidence_threshold").value
        self.imgsz = self.get_parameter("imgsz").value
        self.publish_annotated = self.get_parameter("publish_annotated").value
        annotated_image_topic = self.get_parameter("annotated_image_topic").value

        self.get_logger().info(f"Loading model '{model_path}' on device '{self.device}'")
        self.model = YOLO(model_path)

        self.bridge = CvBridge()

        self.detections_pub = self.create_publisher(PersonDetectionArray, detections_topic, 10)
        self.annotated_pub = None
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(Image, annotated_image_topic, 10)

        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)

        self.get_logger().info(
            f"Subscribed to '{image_topic}', publishing detections on '{detections_topic}'"
        )

    def image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        results = self.model(
            frame,
            device=self.device,
            conf=self.confidence_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        result = results[0]

        detection_array = PersonDetectionArray()
        detection_array.header = msg.header

        boxes = result.boxes
        keypoints = result.keypoints

        if boxes is not None:
            for i in range(len(boxes)):
                detection = PersonDetection()

                xyxy = boxes.xyxy[i].tolist()
                x1, y1, x2, y2 = xyxy
                detection.bbox_x = float(x1)
                detection.bbox_y = float(y1)
                detection.bbox_width = float(x2 - x1)
                detection.bbox_height = float(y2 - y1)
                detection.confidence = float(boxes.conf[i])
                detection.track_id = int(boxes.id[i]) if boxes.id is not None else -1

                if keypoints is not None:
                    kp_xy = keypoints.xy[i].tolist()
                    kp_conf = (
                        keypoints.conf[i].tolist()
                        if keypoints.conf is not None
                        else [1.0] * len(kp_xy)
                    )
                    for name, (x, y), conf in zip(COCO_KEYPOINT_NAMES, kp_xy, kp_conf):
                        kp = Keypoint2D()
                        kp.name = name
                        kp.x = float(x)
                        kp.y = float(y)
                        kp.confidence = float(conf)
                        detection.keypoints.append(kp)

                detection_array.detections.append(detection)

        self.detections_pub.publish(detection_array)

        if self.publish_annotated and self.annotated_pub is not None:
            annotated_frame = result.plot()
            annotated_msg = self.bridge.cv2_to_imgmsg(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), encoding="rgb8"
            )
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BboxKeypointsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
