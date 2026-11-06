from glob import glob

from setuptools import find_packages, setup

package_name = "human_awareness"

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
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Laura",
    maintainer_email="lauramuresan420@gmail.com",
    description=(
        "Per-person awareness over time (\"knows the robot is there\"), "
        "accumulated per track_id from human_gaze's sees_robot decisions."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "awareness_node = human_awareness.awareness_node:main",
        ],
    },
)
