import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory("human_gaze"), "config")

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(config_dir, "human_gaze.yaml"),
        description="Path to a yaml file with gaze_node algorithm-tuning parameters.",
    )
    robot_rig_file_arg = DeclareLaunchArgument(
        "robot_rig_file",
        default_value=os.path.join(config_dir, "robot_rig.yaml"),
        description="Path to a yaml file with your robot's physical calibration "
        "(camera height/tilt/offset, robot radius) - see config/robot_rig.yaml.",
    )

    node = Node(
        package="human_gaze",
        executable="gaze_node",
        name="gaze_node",
        output="screen",
        # robot_rig_file loaded second so it can override params_file on
        # any key both happen to declare (neither does today, but this
        # keeps the physical-calibration file authoritative if that ever
        # changes).
        parameters=[LaunchConfiguration("params_file"), LaunchConfiguration("robot_rig_file")],
    )

    return LaunchDescription([params_file_arg, robot_rig_file_arg, node])
