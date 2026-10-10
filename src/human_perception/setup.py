from glob import glob

from setuptools import find_packages, setup

package_name = "human_perception"

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
        "Monocular RGB human detection: bounding box + body keypoints via "
        "YOLO26-pose."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "bbox_keypoints_node = human_perception.bbox_keypoints_node:main",
        ],
    },
)
