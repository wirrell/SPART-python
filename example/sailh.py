"""
Example for SAILH model
"""
import SPART
import nvtx
from SPART.prospect_5d import LeafBiology, PROSPECT_5D
from SPART.sailh import SAILH, CanopyStructure, Angles
from SPART.bsm import SoilParameters, BSM


optical_params = SPART.load_optical_parameters()
spectral_info = SPART.SpectralBands()

# Compute default soil optics
soil_params = SoilParameters(0.5, 0, 100, 20)
soil_optics = BSM(soil_params, optical_params)
soil_optics = SPART.set_soil_refl_trans_assumptions(soil_optics, spectral_info)

# Compute default leaf reflectance
leaf_biology = LeafBiology(40, 0.01, 0.02, 0, 10, 10, 1.5)
leaf_optics = PROSPECT_5D(leaf_biology, optical_params)
# Assign sailh assumptions
leaf_optics = SPART.set_leaf_refl_trans_assumptions(leaf_optics, leaf_biology, spectral_info)

# Use default canopy parameters and angles for test
angles = Angles(40, 0, 0)
canopy_structure = CanopyStructure(3, -0.35, -0.15, 0.05)

# TEST BEGINS
@nvtx.annotate('SAILH test', color='green')
def test(soil_optics, leaf_optics, canopy_structure, angles):
    result = SAILH(soil_optics, leaf_optics, canopy_structure, angles)

test(soil_optics, leaf_optics, canopy_structure, angles)
# TEST ENDS
