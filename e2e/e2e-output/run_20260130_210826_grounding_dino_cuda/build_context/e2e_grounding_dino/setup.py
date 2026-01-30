from setuptools import find_packages, setup

package_name = 'e2e_grounding_dino'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/params.yaml']),
        ('share/' + package_name + '/launch', ['launch/vision.launch.py']),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='ROS2 vision node: grounding_dino (grounding_dino_base)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'e2e_grounding_dino_node = e2e_grounding_dino.node:main',
        ],
    },
)
