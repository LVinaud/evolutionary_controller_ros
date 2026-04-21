from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    genoma_arg = DeclareLaunchArgument(
        'genoma',
        default_value='genomas/melhor.npy',
        description='Caminho do arquivo de genoma a carregar no controlador',
    )
    return LaunchDescription([
        genoma_arg,
        Node(
            package='evolutionary_controller_ros',
            executable='controlador_nn',
            name='controlador_demo',
            output='screen',
            parameters=[{'genoma_path': LaunchConfiguration('genoma')}],
        ),
    ])
