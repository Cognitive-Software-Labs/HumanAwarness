"""ROS2 node: records a live run to its own folder for debugging later
(docs/context/run_recorder.md). Pairs each final-output message with the
camera frame it was computed on (same header stamp, like visualization_node)
and hands both to `RunWriter`; also keeps /rosout, every node's parameters
and the problem markers sent by `ros2 run human_run_recorder mark`.

It only writes files: nothing in the pipeline depends on it.
"""

import datetime
import threading
from pathlib import Path

import rclpy
import yaml
from cv_bridge import CvBridge
from rcl_interfaces.msg import Log
from rcl_interfaces.srv import GetParameters, ListParameters
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import parameter_value_to_python
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from human_awareness_msgs.msg import PersonDetectionArray
from human_run_recorder.compress import compress_run, ffmpeg_available, runs_to_compress
from human_run_recorder.run_files import RunWriter, git_info
from human_visualization.adapters import ImageBuffer, person_detection_to_overlay, stamp_to_ns
from human_visualization.overlay import draw_info, draw_person

_LEVEL_NAMES = {10: "DEBUG", 20: "INFO", 30: "WARN", 40: "ERROR", 50: "FATAL"}
# /rosout's publishers keep their last 1000 lines (transient local), so the
# recorder also gets what nodes logged before it started (model loading etc.).
_ROSOUT_QOS = QoSProfile(depth=1000, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="milliseconds")


class RunRecorderNode(Node):
    def __init__(self):
        super().__init__("run_recorder_node")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("detections_topic", "human_awareness/awareness")
        self.declare_parameter("mark_topic", "human_run_recorder/mark")
        self.declare_parameter("runs_dir", "~/human_awareness_runs")
        self.declare_parameter("run_name", "")
        self.declare_parameter("image_buffer_size", 60)
        self.declare_parameter("aware_threshold_deg", 45.0)
        self.declare_parameter("record_annotated", True)
        self.declare_parameter("video_probe_frames", 15)
        self.declare_parameter("fallback_fps", 15.0)
        self.declare_parameter("params_dump_delay_s", 5.0)
        self.declare_parameter("log_min_level", 20)
        self.declare_parameter("compress_previous_runs", True)
        self.declare_parameter("compress_crf", 23)
        self.declare_parameter("compress_threads", 2)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.aware_threshold_deg = p("aware_threshold_deg")
        self.log_min_level = p("log_min_level")
        self.started_at = _now_iso()
        run_name = p("run_name") or datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = Path(p("runs_dir")).expanduser() / run_name
        self.writer = RunWriter(
            self.run_dir,
            record_annotated=p("record_annotated"),
            probe_frames=p("video_probe_frames"),
            fallback_fps=p("fallback_fps"),
            aware_threshold_deg=self.aware_threshold_deg,
        )
        self.meta = {
            "run_name": run_name,
            "started_at": self.started_at,
            "image_topic": p("image_topic"),
            "detections_topic": p("detections_topic"),
            "git": git_info(Path(__file__).resolve()),
        }
        self.writer.write_json("meta.json", self.meta)

        self.bridge = CvBridge()
        self.images = ImageBuffer(p("image_buffer_size"))
        self.params: dict = {}
        self._closed = False

        self.create_subscription(Image, p("image_topic"), self._on_image, 30)
        self.create_subscription(PersonDetectionArray, p("detections_topic"), self._on_detections, 10)
        self.create_subscription(String, p("mark_topic"), self._on_mark, 10)
        self.create_subscription(Log, "/rosout", self._on_log, _ROSOUT_QOS)
        self._dump_timer = self.create_timer(p("params_dump_delay_s"), self._dump_params)

        self.get_logger().info(
            f"Recording this run to {self.run_dir} - mark a problem with: ros2 run human_run_recorder mark"
        )

        self._stop_compressing = threading.Event()
        self._ffmpeg = None
        self._compress_thread = None
        if p("compress_previous_runs"):
            self._compress_thread = threading.Thread(
                target=self._compress_previous_runs,
                args=(self.run_dir.parent, p("compress_crf"), p("compress_threads")),
                daemon=True,
            )
            self._compress_thread.start()

    # --- frames ---------------------------------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        self.images.add(stamp_to_ns(msg.header.stamp), msg)

    def _on_detections(self, msg: PersonDetectionArray) -> None:
        stamp_ns = stamp_to_ns(msg.header.stamp)
        if stamp_ns == 0:
            # Unstamped source: pair with the newest image and time by arrival.
            image_msg = self.images.newest()
            stamp_ns = self.get_clock().now().nanoseconds
        else:
            image_msg = self.images.pop_match(stamp_ns)
        if image_msg is None:
            self.get_logger().warning(
                "No camera frame with this output's stamp - not recorded (startup, or image_buffer_size "
                "shorter than the pipeline delay)",
                throttle_duration_sec=5.0,
            )
            return
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        self.writer.add_frame(frame, msg.detections, stamp_ns, draw_overlay=self._draw_overlay)

    def _draw_overlay(self, image, detections) -> None:
        for det in detections:
            person = person_detection_to_overlay(det, self.aware_threshold_deg)
            if person is not None:
                draw_person(image, person)
        draw_info(image, [f"frame {self.writer.frames_written}"])

    # --- markers and logs -----------------------------------------------------------------

    def _on_mark(self, msg: String) -> None:
        marker = self.writer.add_marker(msg.data, _now_iso())
        self.get_logger().info(f"Marker at frame {marker['frame']}: {msg.data or '(no note)'}")

    def _on_log(self, msg: Log) -> None:
        if msg.level < self.log_min_level:
            return
        stamp = f"{msg.stamp.sec}.{msg.stamp.nanosec:09d}"
        level = _LEVEL_NAMES.get(msg.level, str(msg.level))
        self.writer.add_log_line(f"{stamp} frame={self.writer.frames_written} {level} [{msg.name}] {msg.msg}")

    # --- parameters -----------------------------------------------------------------------

    def _dump_params(self) -> None:
        """Every running node's parameters as a params.yaml, the same shape
        as the config files, so a run can be repeated with exactly them."""
        self._dump_timer.cancel()
        for name, namespace in self.get_node_names_and_namespaces():
            if name == self.get_name() or name.startswith("_"):
                continue
            full_name = f"{namespace.rstrip('/')}/{name}"
            client = self.create_client(ListParameters, f"{full_name}/list_parameters")
            if not client.service_is_ready():
                self.get_logger().warning(f"No parameter service on {full_name} - not in params.yaml")
                continue
            future = client.call_async(ListParameters.Request())
            future.add_done_callback(lambda f, n=full_name: self._on_param_names(n, f))

    def _on_param_names(self, full_name: str, future) -> None:
        names = list(future.result().result.names)
        client = self.create_client(GetParameters, f"{full_name}/get_parameters")
        request = GetParameters.Request(names=names)
        future = client.call_async(request)
        future.add_done_callback(lambda f: self._on_param_values(full_name, names, f))

    def _on_param_values(self, full_name: str, names: list[str], future) -> None:
        values = [parameter_value_to_python(v) for v in future.result().values]
        self.params[full_name] = {"ros__parameters": dict(sorted(zip(names, values)))}
        (self.run_dir / "params.yaml").write_text(yaml.safe_dump(dict(sorted(self.params.items())), sort_keys=False))

    # --- compressing earlier runs ---------------------------------------------------------

    def _compress_previous_runs(self, runs_dir: Path, crf: int, threads: int) -> None:
        """Background thread: shrinks every earlier, finished run (H.264 video,
        gzipped text), at low priority so the live pipeline keeps its CPU."""
        runs = runs_to_compress(runs_dir, exclude=self.run_dir)
        if not runs:
            return
        if not ffmpeg_available():
            self.get_logger().warning("ffmpeg/ffprobe not installed: earlier runs' videos are left uncompressed")
        self.get_logger().info(f"Compressing {len(runs)} earlier run(s) in the background")
        for run_dir in runs:
            if self._stop_compressing.is_set():
                return
            try:
                state = compress_run(
                    run_dir, crf=crf, threads=threads, low_priority=True,
                    on_process=self._set_ffmpeg, should_stop=self._stop_compressing.is_set,
                )
            except Exception as e:  # noqa: BLE001 - one bad folder mustn't stop the others or the recording
                if not self._stop_compressing.is_set():
                    self.get_logger().warning(f"Couldn't compress {run_dir.name}, left as it was: {e}")
                continue
            for name, reason in state.get("failed", {}).items():
                self.get_logger().warning(f"{run_dir.name}/{name} left uncompressed: {reason}")
            if state.get("complete"):
                self.get_logger().info(
                    f"Compressed {run_dir.name}: {state['bytes_before'] / 1e6:.1f} MB -> "
                    f"{state['bytes_after'] / 1e6:.1f} MB"
                )

    def _set_ffmpeg(self, proc) -> None:
        self._ffmpeg = proc

    # --- shutdown -------------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Stop compressing first: an unfinished step is redone next time, never half-applied.
        self._stop_compressing.set()
        if self._ffmpeg is not None and self._ffmpeg.poll() is None:
            self._ffmpeg.terminate()
        if self._compress_thread is not None:
            self._compress_thread.join(timeout=10.0)
        self.writer.close()
        self.writer.write_json("meta.json", {**self.meta, "ended_at": _now_iso(), **self.writer.summary()})


def main(args=None):
    rclpy.init(args=args)
    node = RunRecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C (ros2 run) / SIGINT from ros2 launch
    finally:
        node.close()  # finalizes the videos and meta.json
        print(f"Run saved to {node.run_dir} ({node.writer.frames_written} frames)", flush=True)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
