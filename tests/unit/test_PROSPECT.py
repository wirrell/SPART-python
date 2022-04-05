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
    # NOTE: we no longer use this class for direct interaction with prospect
    # leaf_biology = LeafBiology(*test_case[:7])
    exp_refl = test_case[7:2008].to_numpy()
    exp_tran = test_case[2008:4009].to_numpy()
    exp_kchl = test_case[4009:6010].to_numpy()

    refl, tran, kChlrel = PROSPECT_5D(
        *test_case[:7],
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

    np.testing.assert_almost_equal(exp_refl, refl[:, 0].flatten())
    np.testing.assert_almost_equal(exp_tran, tran[:, 0].flatten())
    np.testing.assert_almost_equal(exp_kchl, kChlrel[:, 0].flatten())
    assert True


def test_PROSPECT_5D_concurrent(prospect_test_case, optical_params):
    num, test_case = prospect_test_case
    leaf_biology = LeafBiology(*test_case[:7])
    exp_refl = test_case[7:2008].to_numpy()
    exp_tran = test_case[2008:4009].to_numpy()
    exp_kchl = test_case[4009:6010].to_numpy()

    Cab = np.array([test_case[0]] * 10)
    Cdm = np.array([test_case[1]] * 10)
    Cw = np.array([test_case[2]] * 10)
    Cs = np.array([test_case[3]] * 10)
    Cca = np.array([test_case[4]] * 10)
    Cant = np.array([test_case[5]] * 10)
    N = np.array([test_case[6]] * 10)

    refl, tran, kChlrel = PROSPECT_5D(
        Cab,
        Cdm,
        Cw,
        Cs,
        Cca,
        Cant,
        N,
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

    np.testing.assert_almost_equal(exp_refl, refl[:, 0].flatten())
    np.testing.assert_almost_equal(exp_tran, tran[:, 0].flatten())
    np.testing.assert_almost_equal(exp_kchl, kChlrel[:, 0].flatten())
    assert True
