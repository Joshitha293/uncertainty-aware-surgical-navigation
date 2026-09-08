"""Launch the simulated surgical instrument in Gazebo Harmonic."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """
    Create the Gazebo surgical-instrument launch description.
    """
    description_share = Path(
        get_package_share_directory(
            'surgical_navigation_description'
        )
    )

    ros_gz_sim_share = Path(
        get_package_share_directory(
            'ros_gz_sim'
        )
    )

    xacro_file = (
        description_share
        / 'urdf'
        / 'surgical_instrument.urdf.xacro'
    )

    robot_description_content = ParameterValue(
        Command(
            [
                'xacro ',
                str(xacro_file),
            ]
        ),
        value_type=str,
    )

    robot_description = {
        'robot_description': robot_description_content,
    }

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(
                ros_gz_sim_share
                / 'launch'
                / 'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': '-r empty.sdf',
            'on_exit_shutdown': 'true',
        }.items(),
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='simulation_clock_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        output='screen',
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            robot_description,
            {
                'use_sim_time': True,
            },
        ],
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_surgical_instrument',
        output='screen',
        arguments=[
            '-world',
            'empty',
            '-topic',
            'robot_description',
            '-name',
            'surgical_instrument',
            '-x',
            '0.0',
            '-y',
            '0.0',
            '-z',
            '0.10',
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '60',
        ],
        output='screen',
    )

    trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_trajectory_controller',
            '--controller-manager',
            '/controller_manager',
            '--controller-manager-timeout',
            '60',
        ],
        output='screen',
    )

    return LaunchDescription(
        [
            gazebo,
            clock_bridge,
            robot_state_publisher,
            spawn_robot,
            joint_state_broadcaster_spawner,
            trajectory_controller_spawner,
        ]
    )