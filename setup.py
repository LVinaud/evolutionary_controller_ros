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
    description='Controlador evolutivo para o robô do prm_2026 (Capture The Flag).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controlador_nn = evolutionary_controller_ros.controladores.controlador_nn:main',
            'controlador_reativo = evolutionary_controller_ros.controladores.controlador_reativo:main',
            'orquestrador = evolutionary_controller_ros.avaliacao.orquestrador:main',
        ],
    },
)
