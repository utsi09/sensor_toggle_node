from setuptools import setup

package_name = 'sensor_toggle_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Your Name',
    author_email='your_email@example.com',
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='ROS2 node to toggle CARLA sensors using keyboard input',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_toggle_node = sensor_toggle_ros2.sensor_toggle_node:main'
            # 'test_node = sensor_toggle_ros2.test_node:main',
        ],
    },
)
