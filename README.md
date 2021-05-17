# SPART-python

## Overview
This is a Python port of the Matlab code for the SPART radiative transfer model.

The source code can be found at: https://github.com/wirrell/SPART-python

The original code can be found at: https://github.com/peiqiyang/SPART

The model paper is:
The SPART model: A soil-plant-atmosphere radiative transfer model for satellite measurements in the solar spectrum - Yang et al. (2020)

## Installation

There are two ways:

### Installation from Source

Create a clean virtual environment and execute the following command in the package root

	python setup.py install --user

This also install the [dependencies](#requirements).

### Installation from PyPI

For user with a ETH Gitlab account it is possible to install the latest version (or an older one) from the ETH Gitlab PyPI index. Therefore, users need a personal Gitlab Access Token. Please check the [official Gitlab docs](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html) about how to create tokens and what they mean.

Once a token is created, it is possible to install the package from the ETH PyPI repository into a clean virtual environment using pip:

	pip install spart --extra-index-url https://__token__:<your_personal_token>@gitlab.ethz.ch/api/v4/projects/24594/packages/pypi/simple

### Publishing to PyPI

Developers wishing to bring their own version of SPART to the ETH PyPI index, can use the [shell script](./build_publish.sh). The command in it should work in very similar form also under Windows (at least the python commands will definitely work).

To publish to PyPI the Python package [twine](https://pypi.org/project/twine/) must be install before.

Moreover, it is recommended to read the official [Gitlab documentation](https://docs.gitlab.com/ee/user/packages/pypi_repository/index.html) to learn about PyPI and how the publishing and configuration process works.

For the build script provided, it is necessary to configure a .pypirc file which Linux user have to place in the /home/ directory:

	~/.pypirc

The .pypirc file has the following content and needs again a personal Gitlab API access token:

```
[distutils]
index-servers =
    spart

[spart]
repository = https://gitlab.ethz.ch/api/v4/projects/24594/packages/pypi
username = __token__
password = <your_personal_access_token>
```

## Requirements
```
Python 3.4+
NumPy
SciPy
Pandas
```
All dependencies are installed when running 
	
	python setup.py install --user

in a clean virtual environment. This also applies to the package data required to run (pickled objects containing optical absorption coefficients and sensor specific information like spectral response functions and band wavelengths).

## Quickstart Example
```
   import SPART
   leafbio = SPART.LeafBiology(40, 10, 0.02, 0.01, 0, 10, 1.5)
   soilpar = SPART.SoilParameters(0.5, 0, 100, 15)
   canopy = SPART.CanopyStructure(3, -0.35, -0.15, 0.05)
   angles = SPART.Angles(40, 0, 0)
   atm = SPART.AtmosphericProperties(0.3246, 0.3480, 1.4116, 1013.25)
   spart = SPART.SPART(soilpar, leafbio, canopy, atm, angles, 'TerraAqua-MODIS',
                 100)
   results = spart.run()  # Pandas DataFrame containing R_TOC, R_TOA, L_TOA
```

See also [here](./example/example.py) for an executable code snippet.

## Documentation
Full documentation can be found at https://spart-python.readthedocs.io/
