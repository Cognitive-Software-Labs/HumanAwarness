import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config_path = os.path.join(
        get_package_share_directory("human_perception"),
        "config",
        "bbox_keypoints.yaml",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_config_path,
        description="Path to a yaml file with bbox_keypoints_node parameters.",
    )

    node = Node(
        package="human_perception",
        executable="bbox_keypoints_node",
        name="bbox_keypoints_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([params_file_arg, node])
