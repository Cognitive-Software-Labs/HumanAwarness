from glob import glob

from setuptools import find_packages, setup

package_name = "human_visualization"

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
        "Live annotated camera view: draws each tracked person's bbox, id, "
        "looking label and awareness onto the camera image as a ROS topic."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "visualization_node = human_visualization.visualization_node:main",
        ],
    },
)
