from glob import glob

from setuptools import find_packages, setup

package_name = "human_head_pose"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Laura",
    maintainer_email="lauramuresan420@gmail.com",
    description=(
        "Head pose (yaw/pitch/roll) via 6DRepNet360, combined with each "
        "detection's bearing from the camera to estimate whether a tracked "
        "person is facing the robot."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "head_pose_node = human_head_pose.head_pose_node:main",
        ],
    },
)
