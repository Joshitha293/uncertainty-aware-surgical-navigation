"""Set up the surgical-navigation ROS 2 Python package."""

from setuptools import find_packages
from setuptools import setup


package_name = 'surgical_navigation_ros'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(
        exclude=[
            'test',
        ]
    ),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            [
                'resource/' + package_name,
            ],
        ),
        (
            'share/' + package_name,
            [
                'package.xml',
            ],
        ),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='joshithaa',
    maintainer_email='joshithaa@example.com',
    description=(
        'ROS 2 integration for the simulated uncertainty-aware '
        'surgical navigation research platform.'
    ),
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            (
                'runtime_safety_monitor = '
                'surgical_navigation_ros.safety_monitor_node:main'
            ),
            (
                'autonomous_supervisor = '
                'surgical_navigation_ros.autonomous_supervisor_node:main'
            ),
            (
                'navigation_execution = '
                'surgical_navigation_ros.navigation_execution_node:main'
            ),
            (
                'simulated_perception_bridge = '
                'surgical_navigation_ros.perception_bridge_node:main'
            ),
            (
                'demo_scene = '
                'surgical_navigation_ros.demo_scene_node:main'
            ),
        ],
    },
)
