"""
Tests for SAILH Canopy model.
"""
import argparse
import pytest
import numpy as np
import pandas as pd
import SPART
from SPART.prospect_5d import PROSPECT_5D
from SPART.bsm import BSM
from SPART.sailh import SAILH, CanopyStructure, Angles


@pytest.fixture
def sail_test_case(request):
    return request.param


def test_SAILH(
    sail_test_case, optical_params, default_leaf_optics, default_soil_optics
):
    num, test_case = sail_test_case
    canopy_structure = CanopyStructure(*test_case[:4].to_numpy())
    angles = Angles(*test_case[4:7].to_numpy())
    exp_rso = test_case[7:2169].to_numpy()
    exp_rdo = test_case[2169:4331].to_numpy()
    exp_rsd = test_case[4331:6493].to_numpy()
    exp_rdd = test_case[6493:].to_numpy()

    result = SAILH(default_soil_optics, default_leaf_optics, canopy_structure, angles)

    # raises error if not equal
    np.testing.assert_array_almost_equal(exp_rso, result.rso.flatten())
    np.testing.assert_array_almost_equal(exp_rdo, result.rdo.flatten())
    np.testing.assert_array_almost_equal(exp_rsd, result.rsd.flatten())
    np.testing.assert_array_almost_equal(exp_rdd, result.rdd.flatten())

    assert True
