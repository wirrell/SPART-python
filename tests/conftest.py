"""
Created on May 6, 2021

@author: graflu
"""
import os
import pandas as pd
import pytest
import SPART
from distutils import dir_util
from SPART.prospect_5d import LeafBiology
from SPART.sailh import CanopyStructure, Angles
from SPART.bsm import SoilParameters
from SPART.smac import AtmosphericProperties


def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")


def pytest_generate_tests(metafunc):
    if "prospect_test_case" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            tests = pd.read_parquet(
                "test_PROSPECT/PROSPECT_5D_test_cases.gzip"
            ).iterrows()
        else:
            tests = (
                pd.read_parquet("test_PROSPECT/PROSPECT_5D_test_cases.gzip")
                .sample(10, random_state=42)
                .iterrows()
            )
        metafunc.parametrize("prospect_test_case", tests)
    if "sail_test_case" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            tests = pd.read_parquet(
                "test_SAILH/SAILH_test_cases.gzip"
            ).iterrows()
        else:
            tests = (
                pd.read_parquet("test_SAILH/SAILH_test_cases.gzip")
                .sample(10, random_state=42)
                .iterrows()
            )
        metafunc.parametrize("sail_test_case", tests)


@pytest.fixture
def optical_params():
    return SPART.load_optical_parameters()


@pytest.fixture
def datadir(tmpdir, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.

    Taken from stackoverflow
    https://stackoverflow.com/questions/29627341/pytest-where-to-store-expected-data
    (May 6th 2021)
    """
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)

    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))

    return tmpdir


@pytest.fixture
def default_leaf_biology() -> LeafBiology:
    return LeafBiology(40, 0.01, 0.02, 0, 10, 10, 1.5)


@pytest.fixture
def default_canopy_structure() -> CanopyStructure:
    return CanopyStructure(3, -0.35, -0.15, 0.05)


@pytest.fixture
def default_angles() -> Angles:
    return Angles(40, 0, 0)


@pytest.fixture
def default_soil_parameters() -> SoilParameters:
    return SoilParameters(0.5, 0, 100, 20)


@pytest.fixture
def default_atmospheric_properties() -> AtmosphericProperties:
    return AtmosphericProperties(0.325, 0.35, 1.41)
