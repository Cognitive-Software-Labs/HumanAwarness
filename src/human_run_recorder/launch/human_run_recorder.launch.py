import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config_path = os.path.join(
        get_package_share_directory("human_run_recorder"), "config", "human_run_recorder.yaml"
    )
    params_file_arg = DeclareLaunchArgument(
        "params_file", default_value=default_config_path, description="run_recorder_node parameters."
    )
    runs_dir_arg = DeclareLaunchArgument(
        "runs_dir", default_value="~/human_awareness_runs", description="Folder that gets one subfolder per run."
    )
    run_name_arg = DeclareLaunchArgument(
        "run_name", default_value="", description="Name for this run's folder (default: start time)."
    )
    node = Node(
        package="human_run_recorder",
        executable="run_recorder_node",
        name="run_recorder_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"runs_dir": LaunchConfiguration("runs_dir"), "run_name": LaunchConfiguration("run_name")},
        ],
    )
    return LaunchDescription([params_file_arg, runs_dir_arg, run_name_arg, node])
