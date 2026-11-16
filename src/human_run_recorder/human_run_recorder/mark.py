"""`ros2 run human_run_recorder mark`: drop a problem marker into the run
being recorded. Press Enter to mark the frame on screen now, or type a short
note first ("said looking, eyes were away") and press Enter. q quits.

One-off from another script or terminal:
    ros2 topic pub --once /human_run_recorder/mark std_msgs/msg/String "{data: 'note'}"
"""

import sys
import time

import rclpy
from std_msgs.msg import String

_TOPIC = "/human_run_recorder/mark"
_DISCOVERY_TIMEOUT_S = 3.0


def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node("run_marker")
    pub = node.create_publisher(String, _TOPIC, 10)

    deadline = time.monotonic() + _DISCOVERY_TIMEOUT_S
    while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print(f"No recorder is listening on {_TOPIC} - start the pipeline with record:=true. Marks sent now are lost.")

    print("Enter = mark this moment, or type a note then Enter. q + Enter (or Ctrl+C) quits.")
    try:
        while True:
            try:
                line = input("mark> ")
            except EOFError:
                break
            if line.strip().lower() == "q":
                break
            pub.publish(String(data=line.strip()))
            rclpy.spin_once(node, timeout_sec=0.05)
            print(f"  marked{': ' + line.strip() if line.strip() else ''}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
