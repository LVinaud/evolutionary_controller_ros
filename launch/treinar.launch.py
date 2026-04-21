from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    params = PathJoinSubstitution([
        FindPackageShare('evolutionary_controller_ros'),
        'config',
        'ga_params.yaml',
    ])
    return LaunchDescription([
        Node(
            package='evolutionary_controller_ros',
            executable='orquestrador',
            name='orquestrador',
            output='screen',
            parameters=[params],
        ),
    ])
