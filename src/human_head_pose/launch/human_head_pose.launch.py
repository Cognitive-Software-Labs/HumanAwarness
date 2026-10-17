import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config_path = os.path.join(
        get_package_share_directory("human_head_pose"),
        "config",
        "human_head_pose.yaml",
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_config_path,
        description="Path to a yaml file with head_pose_node parameters.",
    )

    node = Node(
        package="human_head_pose",
        executable="head_pose_node",
        name="head_pose_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([params_file_arg, node])
