.. SPART-python documentation master file, created by
   sphinx-quickstart on Sun Aug 23 23:40:51 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SPART-python
============
SPART-Python is a Python implementation of the Soil-Plant-Atmosphere Radiative Transfer (SPART) model.

This code is based on the original matlab code found at https://github.com/peiqiyang/SPART.

The original paper that outlines the SPART model can be found at https://doi.org/10.1016/j.rse.2020.111870

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   source/modules.rst



Installation
------------
``pip install SPART-python``



Quickstart
----------
Here is a quickstart example using the defaults from the original SPART paper.
::

   import SPART
   leafbio = SPART.LeafBiology(40, 10, 0.02, 0.01, 0, 10, 1.5)
   soilpar = SPART.SoilParameters(0.5, 0, 100, 15)
   canopy = SPART.CanopyStructure(3, -0.35, -0.15, 0.05)
   angles = SPART.Angles(40, 0, 0)
   atm = SPART.AtmosphericProperties(0.3246, 0.3480, 1.4116, 1013.25)
   spart = SPART.SPART(soilpar, leafbio, canopy, atm, angles, 'TerraAqua-MODIS',
                 100)
   results = SPART.run()  # Pandas DataFrame containing R_TOC, R_TOA, L_TOA




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
