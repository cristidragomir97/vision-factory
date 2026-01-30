"""Launch file for e2e_yolo."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('e2e_yolo'),
        'config',
        'params.yaml',
    )

    return LaunchDescription([
        Node(
            package='e2e_yolo',
            executable='e2e_yolo_node',
            name='e2e_yolo_node',
            parameters=[config],
            output='screen',
        ),
    ])
