'''
Created on May 13, 2021

@author: lukas
'''

import setuptools
from setuptools import find_packages
from os import path

home = path.abspath(path.dirname(__file__))
with open(path.join(home, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()


setuptools.setup(
    name='spart',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description = 'The Soil-Plant-Atmosphere radiative transfer model',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author='NA',
    author_email ='NA',
    url='NA',
    packages = find_packages(
        where = 'src'
    ),
    package_dir={'':'src'},
    include_package_data=True,
    package_data={'': ['./model_parameters/*.pkl', './sensor_information/*.pkl']},
    classifiers = [
     "Programming Language :: Python :: 3",
     "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'setuptools-scm'
    ]
)