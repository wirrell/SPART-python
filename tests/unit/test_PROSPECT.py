"""
Tests for the PROSPECT_5D model.
"""
import argparse
import pytest
import numpy as np
import pandas as pd
import SPART
from SPART.prospect_5d import LeafBiology, PROSPECT_5D


@pytest.fixture
def prospect_test_case(request):
    return request.param


def test_PROSPECT_5D(prospect_test_case, optical_params):
    num, test_case = prospect_test_case
    leaf_biology = LeafBiology(*test_case[:7])
    exp_refl = test_case[7:2008].to_numpy()
    exp_tran = test_case[2008:4009].to_numpy()
    exp_kchl = test_case[4009:6010].to_numpy()
    result = PROSPECT_5D(leaf_biology, optical_params)

    np.testing.assert_almost_equal(exp_refl, result.refl.flatten())
    np.testing.assert_almost_equal(exp_tran, result.tran.flatten())
    np.testing.assert_almost_equal(exp_kchl, result.kChlrel.flatten())
    assert True
