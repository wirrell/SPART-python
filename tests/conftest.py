"""
Created on May 6, 2021

@author: graflu
"""
import os
import pandas as pd
import pytest
import SPART
from distutils import dir_util
from SPART.prospect_5d import LeafBiology, PROSPECT_5D
from SPART.sailh import CanopyStructure, Angles
from SPART.bsm import SoilParameters, BSM
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


sensors = [
    "TerraAqua-MODIS",
    "LANDSAT4-TM",
    "LANDSAT5-TM",
    "LANDSAT7-ETM",
    "LANDSAT8-OLI",
    "Sentinel2A-MSI",
    "Sentinel2B-MSI",
    "Sentinel3A-OLCI",
    "Sentinel3B-OLCI",
]


@pytest.fixture(params=sensors[0:1])
def sensor(request):
    return request.param


@pytest.fixture
def default_SPARTSimulation(
    default_leaf_biology,
    default_canopy_structure,
    default_angles,
    default_soil_parameters,
    default_atmospheric_properties,
    sensor,
):
    return SPART.SPARTSimulation(
        default_soil_parameters,
        default_leaf_biology,
        default_canopy_structure,
        default_atmospheric_properties,
        default_angles,
        sensor,
        100,
    )



@pytest.fixture
def default_leaf_optics(default_leaf_biology, optical_params):
    leaf_optics =  PROSPECT_5D(default_leaf_biology, optical_params)
    spectral_info = SPART.SpectralBands()
    return SPART.set_leaf_refl_trans_assumptions(leaf_optics,
                                                 default_leaf_biology,
                                                 spectral_info)


@pytest.fixture
def default_soil_optics(default_soil_parameters, optical_params):
    soil_optics = BSM(default_soil_parameters, optical_params)
    spectral_info = SPART.SpectralBands()
    return SPART.set_soil_refl_trans_assumptions(soil_optics, spectral_info)



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
