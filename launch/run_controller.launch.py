from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='evolutionary_controller_ros',
            executable='nn_controller',
            name='nn_controller',
            output='screen',
        ),
    ])
