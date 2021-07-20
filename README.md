# SPART-python

## Overview
This is a Python port of the Matlab code for the SPART radiative transfer model.

The source code can be found at: https://github.com/wirrell/SPART-python

The original code can be found at: https://github.com/peiqiyang/SPART

The optical absorption coefficients for PROSPECT-PRO (the successor of PROSPECT5-D) can be found at: https://github.com/jbferet/prospect

The model paper is:
The SPART model: A soil-plant-atmosphere radiative transfer model for satellite measurements in the solar spectrum - Yang et al. (2020)

### Update Notifications

- 2021-06-12: Sentinel2A+B coefficients were added in final version
- 2021-06-15: PROSPECT-PRO was added as second leaf model option that builds on top of PROSPECT5-D (Féret et al., 2021). More information about PROSPECT-PRO can be found [in the paper by Féret et al. (2021)](https://doi.org/10.1016/j.rse.2020.112173). The optical absorption coefficients for CBC and PROT were taken from [this Github repository](https://github.com/jbferet/prospect).

## Installation

There are two ways:

### Installation from Source

Create a clean virtual environment and execute the following command in the package root

	python setup.py install --user

This also install the [dependencies](#requirements).

### Installation from PyPI

Installation from PyPI:

    pip install spart

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
