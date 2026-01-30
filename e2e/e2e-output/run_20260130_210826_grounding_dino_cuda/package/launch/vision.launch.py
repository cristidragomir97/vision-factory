"""Launch file for e2e_grounding_dino."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('e2e_grounding_dino'),
        'config',
        'params.yaml',
    )

    return LaunchDescription([
        Node(
            package='e2e_grounding_dino',
            executable='e2e_grounding_dino_node',
            name='e2e_grounding_dino_node',
            parameters=[config],
            output='screen',
        ),
    ])
