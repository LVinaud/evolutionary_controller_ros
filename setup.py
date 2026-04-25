from setuptools import find_packages, setup
from glob import glob

package_name = 'evolutionary_controller_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.py')),
        (f'share/{package_name}/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lazaro Pereira',
    maintainer_email='lazaropereiravn@gmail.com',
    description='Evolutionary controller for the prm_2026 robot (Capture The Flag).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gp_controller = evolutionary_controller_ros.controllers.gp_controller:main',
            'orchestrator = evolutionary_controller_ros.evaluation.orchestrator:main',
            # Optional HTTP-based parallel mode. Requires extra pip deps —
            # see README ("Optional: parallel evaluation across machines").
            'worker_server = evolutionary_controller_ros.evaluation.worker_server:main',
            'coordinator = evolutionary_controller_ros.evaluation.coordinator:main',
        ],
    },
)
