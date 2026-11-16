"""ROS2 node: per-person awareness over time ("knows the robot is there"),
the first dimension of stage D (docs/context/awareness.md).

Subscribes to human_gaze's PersonDetectionArray and republishes it, one
message out per message in with the same header, with `awareness_value` /
`awareness_confidence` filled in for every tracked person (ACTIVE and
LOST_*). Time comes from the header stamps, not the node clock, so a
recorded stream gives the same result at any playback speed. Only an
unstamped source (stamp 0) is timed by arrival instead.
"""

import rclpy
from rclpy.node import Node

from human_awareness.adapters import STAMP_REORDERED, STAMP_RESTART, classify_stamp, person_detection_to_cue
from human_awareness.awareness import AwarenessAggregator
from human_awareness_msgs.msg import PersonDetectionArray


class AwarenessNode(Node):
    def __init__(self):
        super().__init__("awareness_node")

        self.declare_parameter("gazes_topic", "human_gaze/gazes")
        self.declare_parameter("awareness_topic", "human_awareness/awareness")
        self.declare_parameter("tau_awareness_s", 20.0)
        self.declare_parameter("forget_delay_s", 4.0)
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
            forget_delay_s=self.get_parameter("forget_delay_s").value,
            cue_max_age_s=self.get_parameter("cue_max_age_s").value,
            rise_time_s=self.get_parameter("rise_time_s").value,
            confidence_time_s=self.get_parameter("confidence_time_s").value,
            weight_iris=self.get_parameter("weight_iris").value,
            weight_head=self.get_parameter("weight_head").value,
            init_value=self.get_parameter("init_value").value,
        )
        self._newest_stamp_s: float | None = None

        self.awareness_pub = self.create_publisher(PersonDetectionArray, awareness_topic, 10)
        self.gazes_sub = self.create_subscription(PersonDetectionArray, gazes_topic, self._on_gazes, 10)

        self.get_logger().info(f"Subscribed to '{gazes_topic}', publishing awareness on '{awareness_topic}'")

    def _message_time_s(self, stamp) -> float:
        if stamp.sec == 0 and stamp.nanosec == 0:
            # Unstamped source: every message would be the same instant and
            # awareness would never move, so use when it arrived instead.
            self.get_logger().warning(
                "Input header stamps are unset (0) - timing awareness by message arrival instead", once=True
            )
            return self.get_clock().now().nanoseconds * 1e-9
        return stamp.sec + stamp.nanosec * 1e-9

    def _on_gazes(self, msg: PersonDetectionArray) -> None:
        stamp_s = self._message_time_s(msg.header.stamp)
        order = classify_stamp(stamp_s, self._newest_stamp_s)
        if order == STAMP_REORDERED:
            self.get_logger().warning(
                f"Dropping a message {self._newest_stamp_s - stamp_s:.3f}s older than the newest one "
                "(delivered out of order)",
                throttle_duration_sec=5.0,
            )
            return
        if order == STAMP_RESTART:
            self.get_logger().warning(
                f"Header stamp went back {self._newest_stamp_s - stamp_s:.3f}s - the source restarted, "
                "resetting all awareness state"
            )
            self.aggregator.reset()
        self._newest_stamp_s = stamp_s

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
