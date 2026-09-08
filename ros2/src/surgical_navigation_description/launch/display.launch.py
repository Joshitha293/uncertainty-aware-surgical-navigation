"""Launch the surgical instrument model in RViz."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Create the RViz robot-description launch configuration."""
    package_share = Path(
        get_package_share_directory(
            'surgical_navigation_description'
        )
    )

    xacro_file = (
        package_share
        / 'urdf'
        / 'surgical_instrument.urdf.xacro'
    )

    rviz_file = (
        package_share
        / 'rviz'
        / 'surgical_navigation.rviz'
    )

    robot_description = {
        'robot_description': Command(
            [
                'xacro ',
                str(xacro_file),
            ]
        )
    }

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            robot_description
        ],
    )

    joint_state_publisher = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=[
            '-d',
            str(rviz_file),
        ],
    )

    return LaunchDescription(
        [
            robot_state_publisher,
            joint_state_publisher,
            rviz,
        ]
    )