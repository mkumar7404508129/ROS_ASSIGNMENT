from setuptools import find_packages, setup

package_name = 'pybullet_sim_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nk',
    maintainer_email='mkumar7404508129@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_simulator_node = pybullet_sim_pkg.robot_simulator_node:main',
            'rectangular_motion_node = pybullet_sim_pkg.rectangular_motion_node:main'
        ],
    },
)
