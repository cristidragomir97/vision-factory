"""Launch file for e2e_depth_anything."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('e2e_depth_anything'),
        'config',
        'params.yaml',
    )

    return LaunchDescription([
        Node(
            package='e2e_depth_anything',
            executable='e2e_depth_anything_node',
            name='e2e_depth_anything_node',
            parameters=[config],
            output='screen',
        ),
    ])
