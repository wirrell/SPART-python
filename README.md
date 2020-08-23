# SPART-python

## Overview
This is a Python port of the Matlab code for the SPART radiative transfer model.

The original code can be found at: https://github.com/peiqiyang/SPART

The model paper is:
The SPART model: A soil-plant-atmosphere radiative transfer model for satellite measurements in the solar spectrum - Yang et al. (2020)

## Requirements
Python 3.4+
NumPy
SciPy
Pandas

## Quickstart Example
'''
    leafbio = LeafBiology(40, 10, 0.02, 0.01, 0, 10, 1.5)
    soilpar = SoilParameters(0.5, 0, 100, 15)
    canopy = CanopyStructure(3, -0.35, -0.15, 0.05)
    angles = Angles(40, 0, 0)
    atm = AtmosphericProperties(0.3246, 0.3480, 1.4116, 1013.25)
    spart = SPART(soilpar, leafbio, canopy, atm, angles, 'TerraAqua-MODIS',
                  100)
    results = spart.run()  # Pandas DataFrame containing R_TOC, R_TOA, L_TOA
'''

## Documentation
Detailed dostrings in every script including parameters and units. Documentaiton to follow.
