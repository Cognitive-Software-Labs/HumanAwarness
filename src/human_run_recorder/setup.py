from glob import glob

from setuptools import find_packages, setup

package_name = "human_run_recorder"

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
    description="Records live pipeline runs (video, per-person predictions, logs, params, markers) for debugging.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "run_recorder_node = human_run_recorder.run_recorder_node:main",
            "mark = human_run_recorder.mark:main",
            "compress_runs = human_run_recorder.compress:main",
        ],
    },
)
