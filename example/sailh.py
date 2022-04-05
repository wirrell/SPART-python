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

# TEST BEGINS
rso, rdo, rsd, rdd = SAILH(
    soil_optics.refl,
    refl,
    tran,
    canopy_structure.nlayers,
    canopy_structure.LAI,
    canopy_structure.lidf,
    angles.sol_angle,
    angles.obs_angle,
    angles.rel_angle,
    canopy_structure.q,
    use_CUDA=True
)
