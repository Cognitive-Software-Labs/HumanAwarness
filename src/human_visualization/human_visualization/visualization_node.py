"""ROS2 node: the live counterpart of the video scripts' annotated output.

Subscribes to the camera image and to the pipeline's final per-person output
(human_awareness/awareness by default), pairs each detections message with
the exact frame it was computed on (same header stamp), draws bbox, track id,
looking label and awareness bar, and publishes the result as an Image topic
to watch with rqt_image_view / RViz2 / Foxglove or record with ros2 bag. See
docs/context/visualization.md.

It only draws: nothing downstream depends on it, and it never runs a model.
"""

import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image

from human_awareness_msgs.msg import PersonDetectionArray
from human_visualization.adapters import ImageBuffer, person_detection_to_overlay, stamp_to_ns
from human_visualization.overlay import draw_info, draw_person

# A lag outside this range means the stamps aren't on the node's clock
# (a bag played without --clock, a camera with its own time base): not shown.
_MAX_PLAUSIBLE_LAG_S = 60.0


class VisualizationNode(Node):
    def __init__(self):
        super().__init__("visualization_node")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "human_awareness/awareness")
        self.declare_parameter("debug_image_topic", "human_visualization/debug_image")
        self.declare_parameter("image_buffer_size", 60)
        self.declare_parameter("aware_threshold_deg", 45.0)
        self.declare_parameter("show_gaze", True)
        self.declare_parameter("show_awareness", True)
        self.declare_parameter("show_info", True)

        image_topic = self.get_parameter("image_topic").value
        detections_topic = self.get_parameter("detections_topic").value
        debug_image_topic = self.get_parameter("debug_image_topic").value
        self.aware_threshold_deg = self.get_parameter("aware_threshold_deg").value
        self.show_gaze = self.get_parameter("show_gaze").value
        self.show_awareness = self.get_parameter("show_awareness").value
        self.show_info = self.get_parameter("show_info").value

        self.bridge = CvBridge()
        self.images = ImageBuffer(self.get_parameter("image_buffer_size").value)

        self.debug_pub = self.create_publisher(Image, debug_image_topic, 10)
        # Deeper than the other nodes' 10: _on_detections (decode, draw,
        # encode) can hold the executor long enough for a burst of frames to
        # queue up, and a frame dropped here is a frame that can't be drawn.
        self.image_sub = self.create_subscription(Image, image_topic, self._on_image, 30)
        self.detections_sub = self.create_subscription(
            PersonDetectionArray, detections_topic, self._on_detections, 10
        )

        self.get_logger().info(
            f"Subscribed to '{image_topic}' + '{detections_topic}', publishing the annotated "
            f"image on '{debug_image_topic}'"
        )

    def _on_image(self, msg: Image) -> None:
        self.images.add(stamp_to_ns(msg.header.stamp), msg)

    def _find_image(self, stamp_ns: int) -> Image | None:
        if stamp_ns == 0:
            # Unstamped source: every frame has the same stamp, so the best
            # available guess is the newest one.
            self.get_logger().warning(
                "Input header stamps are unset (0) - drawing on the newest image, which may not be "
                "the frame the detections came from",
                once=True,
            )
            return self.images.newest()
        image = self.images.pop_match(stamp_ns)
        if image is None:
            oldest = self.images.oldest_ns()
            if oldest is not None and stamp_ns < oldest:
                self.get_logger().warning(
                    f"Detections stamped {(oldest - stamp_ns) * 1e-9:.2f}s before the oldest buffered "
                    "image - its frame was never received here (startup) or already pushed out; if this "
                    f"repeats, the pipeline's delay exceeds image_buffer_size ({self.images.maxlen} "
                    "frames): raise it or lower the camera rate",
                    throttle_duration_sec=5.0,
                )
            else:
                self.get_logger().warning(
                    "No image with this detections message's stamp (frame dropped, or the image and "
                    "detections topics come from different sources)",
                    throttle_duration_sec=5.0,
                )
        return image

    def _on_detections(self, msg: PersonDetectionArray) -> None:
        stamp_ns = stamp_to_ns(msg.header.stamp)
        image_msg = self._find_image(stamp_ns)
        if image_msg is None:
            return
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8").copy()

        n_people = 0
        for det in msg.detections:
            person = person_detection_to_overlay(
                det, self.aware_threshold_deg, show_gaze=self.show_gaze, show_awareness=self.show_awareness
            )
            if person is None:
                continue
            n_people += 1
            draw_person(frame, person)

        if self.show_info:
            lines = [f"people: {n_people}"]
            if stamp_ns != 0:
                lag_s = self.get_clock().now().nanoseconds * 1e-9 - stamp_ns * 1e-9
                if 0.0 <= lag_s < _MAX_PLAUSIBLE_LAG_S:
                    lines.append(f"lag: {lag_s * 1000:.0f} ms")
            draw_info(frame, lines)

        out = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out.header = msg.header
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C (ros2 run) / SIGINT from ros2 launch
    finally:
        node.destroy_node()
        rclpy.try_shutdown()  # the signal handler may already have shut the context down


if __name__ == "__main__":
    main()
