"""Start the whole human awareness pipeline with one command.

    ros2 launch human_awareness_bringup full_pipeline.launch.py
    ros2 launch human_awareness_bringup full_pipeline.launch.py viewer:=true
    ros2 launch human_awareness_bringup full_pipeline.launch.py webcam:=true viewer:=true
    ros2 launch human_awareness_bringup full_pipeline.launch.py image_topic:=/my_robot/camera/color/image_raw
    ros2 launch human_awareness_bringup full_pipeline.launch.py record:=true   # save the run for debugging

Each stage is started through its own package's launch file, so it keeps its
own default config (override a stage's yaml with that stage's launch file
directly if you need to). The camera itself is not started unless
webcam:=true - on a robot, its camera driver is normally already running.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap

# The camera topic every stage's config subscribes to.
_DEFAULT_IMAGE_TOPIC = "/camera/image_raw"

_STAGES = [
    ("human_perception", "bbox_keypoints.launch.py"),
    ("human_tracking", "human_tracking.launch.py"),
    ("human_head_pose", "human_head_pose.launch.py"),
    ("human_gaze", "human_gaze.launch.py"),
    ("human_awareness", "human_awareness.launch.py"),
    ("human_visualization", "human_visualization.launch.py"),
]


def _include(package: str, launch_file: str, launch_arguments=None, condition=None) -> GroupAction:
    """Each stage's launch file declares its own `params_file` argument; in a
    shared scope the first one's value (bbox_keypoints.yaml) sticks and every
    later stage silently loads it instead of its own config. A scoped group
    per include keeps each `params_file` default separate."""
    path = os.path.join(get_package_share_directory(package), "launch", launch_file)
    return GroupAction(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(path),
                launch_arguments=(launch_arguments or {}).items(),
            )
        ],
        scoped=True,
        condition=condition,
    )


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")

    args = [
        DeclareLaunchArgument(
            "image_topic",
            default_value=_DEFAULT_IMAGE_TOPIC,
            description="Camera topic (raw sensor_msgs/Image, reliable QoS) the pipeline reads.",
        ),
        DeclareLaunchArgument(
            "viewer",
            default_value="false",
            description="Open an rqt_image_view window on the annotated view (needs a display).",
        ),
        DeclareLaunchArgument(
            "webcam",
            default_value="false",
            description="Also start a webcam publisher (image_tools cam2image) on image_topic.",
        ),
        DeclareLaunchArgument("webcam_device", default_value="0", description="Webcam device index."),
        DeclareLaunchArgument(
            "robot_rig_file",
            default_value=os.path.join(get_package_share_directory("human_gaze"), "config", "robot_rig.yaml"),
            description="Your robot's camera height/tilt/offset calibration, passed to the gaze stage.",
        ),
        DeclareLaunchArgument(
            "record",
            default_value="false",
            description="Record this run (video, per-person predictions, logs, params, markers) for debugging.",
        ),
        DeclareLaunchArgument(
            "runs_dir", default_value="~/human_awareness_runs", description="Where record:=true saves runs."
        ),
        DeclareLaunchArgument("run_name", default_value="", description="Run folder name (default: start time)."),
    ]

    pipeline = GroupAction(
        [SetRemap(src=_DEFAULT_IMAGE_TOPIC, dst=image_topic)]
        + [
            _include(
                package,
                launch_file,
                {"robot_rig_file": LaunchConfiguration("robot_rig_file")} if package == "human_gaze" else None,
            )
            for package, launch_file in _STAGES
        ]
        + [
            _include(
                "human_run_recorder",
                "human_run_recorder.launch.py",
                {"runs_dir": LaunchConfiguration("runs_dir"), "run_name": LaunchConfiguration("run_name")},
                condition=IfCondition(LaunchConfiguration("record")),
            )
        ]
    )

    webcam = Node(
        package="image_tools",
        executable="cam2image",
        name="webcam",
        output="screen",
        # cam2image defaults to 320x240, too small for faces at room distance.
        # 15 Hz: the chain runs ~5 Hz on CPU, so faster input only adds
        # copies for the 5 image subscribers to drop.
        parameters=[
            {
                "device_id": LaunchConfiguration("webcam_device"),
                "width": 640,
                "height": 480,
                "frequency": 15.0,
            }
        ],
        remappings=[("image", image_topic)],
        # cam2image logs an INFO line per published frame, burying the
        # pipeline's own output.
        ros_arguments=["--log-level", "warn"],
        condition=IfCondition(LaunchConfiguration("webcam")),
    )

    viewer = Node(
        package="rqt_image_view",
        executable="rqt_image_view",
        name="viewer",
        arguments=["/human_visualization/debug_image"],
        condition=IfCondition(LaunchConfiguration("viewer")),
    )

    return LaunchDescription(args + [pipeline, webcam, viewer])
