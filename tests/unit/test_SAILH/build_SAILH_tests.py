import itertools
import pandas as pd
import numpy as np
import progressbar
import SPART
from SPART.prospect_5d import LeafBiology, PROSPECT_5D
from SPART.sailh import CanopyStructure, Angles, SAILH
from SPART.bsm import SoilParameters, BSM


def get_default_leaf_optics():
    leaf_biology = LeafBiology(40, 0.01, 0.02, 0, 10, 10, 1.5)
    leaf_optics = PROSPECT_5D(leaf_biology, SPART.load_optical_parameters())
    spectral_info = SPART.SpectralBands()
    leaf_optics = SPART.set_leaf_refl_trans_assumptions(
        leaf_optics, leaf_biology, spectral_info
    )
    return leaf_optics


def get_default_soil_optics():
    soil_params = SoilParameters(0.5, 0, 100, 20)
    soil_optics = BSM(soil_params, SPART.load_optical_parameters())
    spectral_info = SPART.SpectralBands()
    soil_optics = SPART.set_soil_refl_trans_assumptions(soil_optics, spectral_info)
    return soil_optics


def build_SAILH_test_cases():
    value_combinations = sailh_value_combinations()
    input_columns = [
        "LAI",
        "LIDFa",
        "LIDFb",
        "q",
        "sol_angle",
        "obs_angle",
        "rel_angle",
    ]
    rso_columns = (
        [f"rso_{x}" for x in range(400, 2401)]
        + [f"rso_{x}" for x in range(2500, 15100, 100)]
        + [f"rso_{x}" for x in range(16000, 51000, 1000)]
    )
    rdo_columns = (
        [f"rdo_{x}" for x in range(400, 2401)]
        + [f"rdo_{x}" for x in range(2500, 15100, 100)]
        + [f"rdo_{x}" for x in range(16000, 51000, 1000)]
    )
    rsd_columns = (
        [f"rsd_{x}" for x in range(400, 2401)]
        + [f"rsd_{x}" for x in range(2500, 15100, 100)]
        + [f"rsd_{x}" for x in range(16000, 51000, 1000)]
    )
    rdd_columns = (
        [f"rdd_{x}" for x in range(400, 2401)]
        + [f"rdd_{x}" for x in range(2500, 15100, 100)]
        + [f"rdd_{x}" for x in range(16000, 51000, 1000)]
    )
    all_columns = input_columns + rso_columns + rdo_columns + rsd_columns + rdd_columns
    test_cases = pd.DataFrame(columns=all_columns)

    optical_params = SPART.load_optical_parameters()
    leaf_optics = get_default_leaf_optics()
    soil_optics = get_default_soil_optics()

    print("Total test cases: ", len(list(sailh_value_combinations())))

    for num, prod in progressbar.progressbar(enumerate(value_combinations)):
        canopy_structure = CanopyStructure(*prod[:4])
        angles = Angles(*prod[4:7])
        result = SAILH(soil_optics, leaf_optics, canopy_structure, angles)
        test_cases.loc[-1] = np.concatenate(
            [
                prod,
                result.rso.flatten(),
                result.rdo.flatten(),
                result.rsd.flatten(),
                result.rdd.flatten(),
            ]
        )
        test_cases.index = test_cases.index + 1
        test_cases = test_cases.sort_index()
    test_cases.to_parquet("SAILH_test_cases.gzip", compression="gzip")


def sailh_value_combinations():
    # For reference of ranges, see: https://doi.org/10.1016/j.rse.2020.111870
    LAI = np.arange(1, 8, 3)
    LIDFa = np.arange(-1, 1, 0.4)
    LIDFb = np.arange(-1, 1, 0.4)
    q = np.arange(0.01, 0.2, 0.05)
    sol_angle = np.arange(0, 75, 30)
    obs_angle = np.arange(0, 75, 30)
    rel_angle = np.arange(0, 180, 80)

    value_combinations = itertools.product(
        LAI, LIDFa, LIDFb, q, sol_angle, obs_angle, rel_angle
    )

    return value_combinations


build_SAILH_test_cases()
