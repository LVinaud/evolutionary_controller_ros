from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='evolutionary_controller_ros',
            executable='gp_controller',
            name='gp_controller',
            output='screen',
        ),
    ])
