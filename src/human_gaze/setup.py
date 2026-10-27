from glob import glob

from setuptools import find_packages, setup

package_name = "human_gaze"

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
        "Iris/gaze estimation (MediaPipe Face Landmarker for gating, "
        "MobileGaze for yaw/pitch) fused with human_head_pose's head pose "
        "into a single facing-the-robot angle per tracked person."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "gaze_node = human_gaze.gaze_node:main",
        ],
    },
)
