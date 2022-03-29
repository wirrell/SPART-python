"""
Example for SAILH model
"""
import SPART
from SPART.prospect_5d import LeafBiology, PROSPECT_5D
from SPART.sailh import SAILH, CanopyStructure, Angles
from SPART.bsm import SoilParameters, BSM


optical_params = SPART.load_optical_parameters()

# Compute default soil optics
soil_params = SoilParameters(0.5, 0, 100, 20)
soil_optics = BSM(default_soil_parameters, optical_params)

# Compute default leaf reflectance
leaf_biology = LeafBiology(40, 0.01, 0.02, 0, 10, 10, 1.5)
leaf_optics = PROSPECT_5D(leaf_biology, optical_params)

# Use default canopy parameters and angles for test
angles = Angles(40, 0, 0)
canopy_structure = CanopyStructure(3, -0.35, -0.15, 0.05)

# TEST BEGINS
result = SAILH(soil_optics, leaf_optics, canopy_structure, angles)

# TEST ENDS
