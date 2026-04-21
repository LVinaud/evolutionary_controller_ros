from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    genome_arg = DeclareLaunchArgument(
        'genome',
        default_value='genomes/best.npy',
        description='Path to the genome file to load into the controller',
    )
    return LaunchDescription([
        genome_arg,
        Node(
            package='evolutionary_controller_ros',
            executable='nn_controller',
            name='demo_controller',
            output='screen',
            parameters=[{'genome_path': LaunchConfiguration('genome')}],
        ),
    ])
