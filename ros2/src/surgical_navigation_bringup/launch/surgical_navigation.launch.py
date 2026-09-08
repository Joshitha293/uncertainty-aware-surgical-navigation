"""Launch the complete simulated surgical-navigation ROS 2 stack."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Create the complete surgical-navigation launch description."""
    description_share = Path(
        get_package_share_directory(
            'surgical_navigation_description'
        )
    )

    bringup_share = Path(
        get_package_share_directory(
            'surgical_navigation_bringup'
        )
    )

    rviz_file = (
        description_share
        / 'rviz'
        / 'surgical_navigation.rviz'
    )

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(
                description_share
                / 'launch'
                / 'gazebo.launch.py'
            )
        )
    )

    demo_case = LaunchConfiguration('demo_case')

    research_scene_dir = (
        bringup_share
        / 'models'
        / 'research_scene'
    )

    scene_a_file = (
        research_scene_dir
        / 'model_demo_a.sdf'
    )

    scene_b_file = (
        research_scene_dir
        / 'model_demo_b.sdf'
    )

    scene_c_file = (
        research_scene_dir
        / 'model_demo_c.sdf'
    )

    case_a = IfCondition(
        PythonExpression(
            ["'", demo_case, "' == 'A'"]
        )
    )

    case_b = IfCondition(
        PythonExpression(
            ["'", demo_case, "' == 'B'"]
        )
    )

    case_c = IfCondition(
        PythonExpression(
            ["'", demo_case, "' == 'C'"]
        )
    )

    simulated_perception_bridge_a = Node(
        package='surgical_navigation_ros',
        executable='simulated_perception_bridge',
        name='simulated_perception_bridge',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'position_sigma_m': 0.003,
            }
        ],
        condition=case_a,
    )

    simulated_perception_bridge_b = Node(
        package='surgical_navigation_ros',
        executable='simulated_perception_bridge',
        name='simulated_perception_bridge',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'position_sigma_m': 0.020,
            }
        ],
        condition=case_b,
    )

    simulated_perception_bridge_c = Node(
        package='surgical_navigation_ros',
        executable='simulated_perception_bridge',
        name='simulated_perception_bridge',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'position_sigma_m': 0.035,
            }
        ],
        condition=case_c,
    )

    research_scene_a = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_research_scene_a',
        output='screen',
        arguments=[
            '-world',
            'empty',
            '-file',
            str(scene_a_file),
            '-name',
            'surgical_research_scene',
        ],
        condition=case_a,
    )

    research_scene_b = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_research_scene_b',
        output='screen',
        arguments=[
            '-world',
            'empty',
            '-file',
            str(scene_b_file),
            '-name',
            'surgical_research_scene',
        ],
        condition=case_b,
    )

    research_scene_c = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_research_scene_c',
        output='screen',
        arguments=[
            '-world',
            'empty',
            '-file',
            str(scene_c_file),
            '-name',
            'surgical_research_scene',
        ],
        condition=case_c,
    )

    research_scene_spawn = TimerAction(
        period=5.0,
        actions=[
            research_scene_a,
            research_scene_b,
            research_scene_c,
        ],
    )

    demo_scene = Node(
        package='surgical_navigation_ros',
        executable='demo_scene',
        name='demo_scene_visualiser',
        output='screen',
        parameters=[
            {
                'target_x': 0.10,
                'target_y': 0.00,
                'target_z': 0.00,
            }
        ],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=[
            '-d',
            str(rviz_file),
        ],
    )

    runtime_safety_monitor = Node(
        package='surgical_navigation_ros',
        executable='runtime_safety_monitor',
        name='runtime_safety_monitor',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
            }
        ],
    )

    autonomous_supervisor = Node(
        package='surgical_navigation_ros',
        executable='autonomous_supervisor',
        name='autonomous_supervisor',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'demo_mode': True,
            }
        ],
    )

    navigation_execution = Node(
        package='surgical_navigation_ros',
        executable='navigation_execution',
        name='navigation_execution',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'demo_case',
                default_value='A',
                choices=[
                    'A',
                    'B',
                    'C',
                ],
                description=(
                    'Research demo case: '
                    'A=3mm, B=20mm, C=35mm uncertainty.'
                ),
            ),
            simulation,
            research_scene_spawn,
            simulated_perception_bridge_a,
            simulated_perception_bridge_b,
            simulated_perception_bridge_c,
            demo_scene,
            rviz,
            runtime_safety_monitor,
            autonomous_supervisor,
            navigation_execution,
        ]
    )
