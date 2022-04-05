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


def test_SAILH_CUDA(
    sail_test_case, optical_params, default_leaf_optics, default_soil_optics
):
    num, test_case = sail_test_case
    canopy_structure = CanopyStructure(*test_case[:4].to_numpy())
    angles = Angles(*test_case[4:7].to_numpy())
    exp_rso = test_case[7:2169].to_numpy()
    exp_rdo = test_case[2169:4331].to_numpy()
    exp_rsd = test_case[4331:6493].to_numpy()
    exp_rdd = test_case[6493:].to_numpy()

    rso, rdo, rsd, rdd = SAILH(
        default_soil_optics.refl,
        default_leaf_optics.refl,
        default_leaf_optics.tran,
        canopy_structure.nlayers,
        canopy_structure.LAI,
        canopy_structure.lidf,
        angles.sol_angle,
        angles.obs_angle,
        angles.rel_angle,
        canopy_structure.q,
        use_CUDA=True
    )

    # raises error if not equal
    np.testing.assert_array_almost_equal(exp_rso, rso.flatten(), decimal=4)
    np.testing.assert_array_almost_equal(exp_rdo, rdo.flatten(), decimal=4)
    np.testing.assert_array_almost_equal(exp_rsd, rsd.flatten(), decimal=4)
    np.testing.assert_array_almost_equal(exp_rdd, rdd.flatten(), decimal=4)

    assert True


def test_SAILH_concurrent_CUDA(
    sail_test_case, optical_params, default_leaf_optics, default_soil_optics
):
    num, test_case = sail_test_case
    canopy_structure = CanopyStructure(*test_case[:4].to_numpy())
    angles = Angles(*test_case[4:7].to_numpy())
    exp_rso = test_case[7:2169].to_numpy()
    exp_rdo = test_case[2169:4331].to_numpy()
    exp_rsd = test_case[4331:6493].to_numpy()
    exp_rdd = test_case[6493:].to_numpy()

    concurrent_tests = 10
    soil_refl = np.concatenate(
        [default_soil_optics.refl for _ in range(concurrent_tests)], axis=1
    )
    leaf_tran = np.concatenate(
        [default_leaf_optics.tran for _ in range(concurrent_tests)], axis=1
    )
    leaf_refl = np.concatenate(
        [default_leaf_optics.refl for _ in range(concurrent_tests)], axis=1
    )
    lidf = np.concatenate(
        [canopy_structure.lidf for _ in range(concurrent_tests)], axis=1
    )

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

    # raises error if not equal
    np.testing.assert_array_almost_equal(exp_rso, rso[:, 0].flatten(), decimal=4)
    np.testing.assert_array_almost_equal(exp_rdo, rdo[:, 0].flatten(), decimal=4)
    np.testing.assert_array_almost_equal(exp_rsd, rsd[:, 0].flatten(), decimal=4)
    np.testing.assert_array_almost_equal(exp_rdd, rdd[:, 0].flatten(), decimal=4)

    assert True


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

    rso, rdo, rsd, rdd = SAILH(
        default_soil_optics.refl,
        default_leaf_optics.refl,
        default_leaf_optics.tran,
        canopy_structure.nlayers,
        canopy_structure.LAI,
        canopy_structure.lidf,
        angles.sol_angle,
        angles.obs_angle,
        angles.rel_angle,
        canopy_structure.q,
    )

    # raises error if not equal
    np.testing.assert_array_almost_equal(exp_rso, rso.flatten())
    np.testing.assert_array_almost_equal(exp_rdo, rdo.flatten())
    np.testing.assert_array_almost_equal(exp_rsd, rsd.flatten())
    np.testing.assert_array_almost_equal(exp_rdd, rdd.flatten())

    assert True


def test_SAILH_concurrent(
    sail_test_case, optical_params, default_leaf_optics, default_soil_optics
):
    num, test_case = sail_test_case
    canopy_structure = CanopyStructure(*test_case[:4].to_numpy())
    angles = Angles(*test_case[4:7].to_numpy())
    exp_rso = test_case[7:2169].to_numpy()
    exp_rdo = test_case[2169:4331].to_numpy()
    exp_rsd = test_case[4331:6493].to_numpy()
    exp_rdd = test_case[6493:].to_numpy()

    concurrent_tests = 10
    soil_refl = np.concatenate(
        [default_soil_optics.refl for _ in range(concurrent_tests)], axis=1
    )
    leaf_tran = np.concatenate(
        [default_leaf_optics.tran for _ in range(concurrent_tests)], axis=1
    )
    leaf_refl = np.concatenate(
        [default_leaf_optics.refl for _ in range(concurrent_tests)], axis=1
    )
    lidf = np.concatenate(
        [canopy_structure.lidf for _ in range(concurrent_tests)], axis=1
    )

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
    )

    # raises error if not equal
    np.testing.assert_array_almost_equal(exp_rso, rso[:, 0].flatten())
    np.testing.assert_array_almost_equal(exp_rdo, rdo[:, 0].flatten())
    np.testing.assert_array_almost_equal(exp_rsd, rsd[:, 0].flatten())
    np.testing.assert_array_almost_equal(exp_rdd, rdd[:, 0].flatten())

    assert True
