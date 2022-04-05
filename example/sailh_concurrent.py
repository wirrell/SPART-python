"""
Example for SAILH model
"""
import SPART
import nvtx
import numpy as np
from SPART.prospect_5d import LeafBiology, PROSPECT_5D
from SPART.sailh import SAILH, CanopyStructure, Angles
from SPART.bsm import SoilParameters, BSM


optical_params = SPART.load_optical_parameters()
spectral_info = SPART.SpectralBands()

# Compute default soil optics
soil_params = SoilParameters(0.5, 0, 100, 20)
soil_optics = BSM(soil_params, optical_params)
soil_optics = SPART.set_soil_refl_trans_assumptions(soil_optics, spectral_info)
leaf_biology = LeafBiology(40, 0.01, 0.02, 0, 10, 10, 1.5)

# Compute default leaf reflectance
refl, tran, kChlrel = PROSPECT_5D(
    40, 0.01, 0.02, 0, 10, 10, 1.5,
    optical_params["nr"],
    optical_params["Kdm"],
    optical_params["Kab"],
    optical_params["Kca"],
    optical_params["Kw"],
    optical_params["Ks"],
    optical_params["Kant"],
    optical_params["cbc"],
    optical_params["prot"],
)
# Assign sailh assumptions
refl, tran = SPART.set_leaf_refl_trans_assumptions(refl, tran, leaf_biology, spectral_info)

# Use default canopy parameters and angles for test
angles = Angles(40, 0, 0)
canopy_structure = CanopyStructure(3, -0.35, -0.15, 0.05)

concurrent_tests = 100
soil_refl = np.concatenate(
    [soil_optics.refl for _ in range(concurrent_tests)], axis=1
)
leaf_tran = np.concatenate(
    [tran for _ in range(concurrent_tests)], axis=1
)
leaf_refl = np.concatenate(
    [refl for _ in range(concurrent_tests)], axis=1
)
lidf = np.concatenate(
    [canopy_structure.lidf for _ in range(concurrent_tests)], axis=1
)

# TEST BEGINS
rso, rdo, rsd, rdd = SAILH(
    soil_refl,
    leaf_refl,
    leaf_tran,
    np.array([canopy_structure.nlayers] * concurrent_tests),
    np.array([canopy_structure.LAI] * concurrent_tests),
    lidf,
    np.array([angles.sol_angle] * concurrent_tests),
    np.array([angles.obs_angle] * concurrent_tests),
    np.array([angles.rel_angle] * concurrent_tests),
    np.array([canopy_structure.q] * concurrent_tests),
    use_CUDA=True
)
