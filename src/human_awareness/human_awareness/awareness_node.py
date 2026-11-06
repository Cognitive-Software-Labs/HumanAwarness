"""ROS2 node: per-person awareness over time ("knows the robot is there"),
the first dimension of stage D (docs/context/awareness.md).

Subscribes to human_gaze's PersonDetectionArray and republishes it, one
message out per message in with the same header, with `awareness_value` /
`awareness_confidence` filled in for every tracked person (ACTIVE and
LOST_*). Time comes from the header stamps, not the node clock, so a
recorded stream gives the same result at any playback speed.
"""

import rclpy
from rclpy.node import Node

from human_awareness.adapters import person_detection_to_cue
from human_awareness.awareness import AwarenessAggregator
from human_awareness_msgs.msg import PersonDetectionArray


def _stamp_to_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class AwarenessNode(Node):
    def __init__(self):
        super().__init__("awareness_node")

        self.declare_parameter("gazes_topic", "human_gaze/gazes")
        self.declare_parameter("awareness_topic", "human_awareness/awareness")
        self.declare_parameter("tau_awareness_s", 8.0)
        self.declare_parameter("cue_max_age_s", 0.5)
        self.declare_parameter("rise_time_s", 0.5)
        self.declare_parameter("confidence_time_s", 2.0)
        self.declare_parameter("weight_iris", 1.0)
        self.declare_parameter("weight_head", 0.5)
        self.declare_parameter("init_value", 0.5)

        gazes_topic = self.get_parameter("gazes_topic").value
        awareness_topic = self.get_parameter("awareness_topic").value
        self.aggregator = AwarenessAggregator(
            tau_s=self.get_parameter("tau_awareness_s").value,
            cue_max_age_s=self.get_parameter("cue_max_age_s").value,
            rise_time_s=self.get_parameter("rise_time_s").value,
            confidence_time_s=self.get_parameter("confidence_time_s").value,
            weight_iris=self.get_parameter("weight_iris").value,
            weight_head=self.get_parameter("weight_head").value,
            init_value=self.get_parameter("init_value").value,
        )
        self._last_stamp_s: float | None = None

        self.awareness_pub = self.create_publisher(PersonDetectionArray, awareness_topic, 10)
        self.gazes_sub = self.create_subscription(PersonDetectionArray, gazes_topic, self._on_gazes, 10)

        self.get_logger().info(f"Subscribed to '{gazes_topic}', publishing awareness on '{awareness_topic}'")

    def _on_gazes(self, msg: PersonDetectionArray) -> None:
        stamp_s = _stamp_to_seconds(msg.header.stamp)
        if self._last_stamp_s is not None and stamp_s < self._last_stamp_s:
            # The source restarted (a looping bag, a reset simulation): the old state describes another timeline.
            self.get_logger().warning(
                f"Header stamp went back {self._last_stamp_s - stamp_s:.3f}s - resetting all awareness state"
            )
            self.aggregator.reset()
        self._last_stamp_s = stamp_s

        tracked_ids = set()
        for det in msg.detections:
            if det.track_id < 0:
                continue  # untracked: no identity to accumulate over
            tracked_ids.add(det.track_id)
            estimate = self.aggregator.observe(det.track_id, stamp_s, person_detection_to_cue(det))
            det.awareness_value = estimate.value
            det.awareness_confidence = estimate.confidence
        # human_tracking publishes every live track (LOST_* included) in every
        # message, so an ID missing here was forgotten or merged into another.
        self.aggregator.prune(tracked_ids)

        self.awareness_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AwarenessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
