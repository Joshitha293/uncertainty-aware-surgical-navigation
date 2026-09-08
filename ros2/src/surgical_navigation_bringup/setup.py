"""Set up the surgical-navigation bringup package."""

from glob import glob
import os

from setuptools import find_packages
from setuptools import setup


package_name = 'surgical_navigation_bringup'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(
        exclude=[
            'test',
        ]
    ),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            [
                'resource/' + package_name,
            ],
        ),
        (
            'share/' + package_name,
            [
                'package.xml',
            ],
        ),
        (
            os.path.join(
                'share',
                package_name,
                'launch',
            ),
            glob(
                'launch/*.launch.py'
            ),
        ),
        (
            os.path.join(
                'share',
                package_name,
                'models',
                'research_scene',
            ),
            glob(
                'models/research_scene/*.sdf'
            ),
        ),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='joshithaa',
    maintainer_email='joshithaa@example.com',
    description=(
        'Bringup package for the simulated uncertainty-aware '
        'surgical navigation research platform.'
    ),
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [],
    },
)
