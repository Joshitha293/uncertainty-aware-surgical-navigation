from setuptools import find_packages, setup


package_name = "surgical_navigation_ros"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(
        exclude=["test"]
    ),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [
                "resource/"
                + package_name
            ],
        ),
        (
            "share/"
            + package_name,
            [
                "package.xml"
            ],
        ),
    ],
    install_requires=[
        "setuptools",
    ],
    zip_safe=True,
    maintainer="Joshithaa",
    maintainer_email="",
    description=(
        "ROS 2 integration for "
        "uncertainty-aware surgical navigation."
    ),
    license="MIT",
    tests_require=[
        "pytest",
    ],
    entry_points={
        "console_scripts": [
            (
                "perception_node = "
                "surgical_navigation_ros."
                "perception_node:main"
            ),
            (
                "viewpoint_receiver_node = "
                "surgical_navigation_ros."
                "viewpoint_receiver_node:main"
            ),
            (
                "safety_gate_node = "
                "surgical_navigation_ros."
                "safety_gate_node:main"
            ),
            (
                "planner_node = "
                "surgical_navigation_ros."
                "planner_node:main"
            ),
            (
                "planner_safety_bridge_node = "
                "surgical_navigation_ros."
                "planner_safety_bridge_node:main"
            ),
            (
                "visualisation_node = "
                "surgical_navigation_ros."
                "visualisation_node:main"
            ),
        ],
    },
)