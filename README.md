# SPART-python

## Overview
This is a Python port of the Matlab code for the SPART radiative transfer model.

The source code can be found at: https://github.com/wirrell/SPART-python

The original code can be found at: https://github.com/peiqiyang/SPART

The model paper is:
The SPART model: A soil-plant-atmosphere radiative transfer model for satellite measurements in the solar spectrum - Yang et al. (2020)

## Installation
`pip install SPART-python`

## Requirements
```
Python 3.4+
NumPy
SciPy
Pandas
```

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

## Documentation
Full documentation can be found at https://spart-python.readthedocs.io/
