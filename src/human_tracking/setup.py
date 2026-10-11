from glob import glob

from setuptools import find_packages, setup

package_name = "human_tracking"

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
        "Persistent person identity across occlusion: Kalman motion + "
        "OSNet appearance (ReID) matching on top of human_perception's "
        "per-frame detections."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "human_tracking_node = human_tracking.human_tracking_node:main",
        ],
    },
)
