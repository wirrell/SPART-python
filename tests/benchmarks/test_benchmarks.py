import pytest
import numpy as np

from SPART.prospect_5d import LeafBiology, PROSPECT_5D
from SPART.bsm import SoilParameters, BSM
from SPART.sailh import CanopyStructure, SAILH, Angles
from SPART.SPART import SpectralBands


def test_benchmark_PROSPECT_5D(benchmark, optical_params):
    # Benchmark using default prospect values
    leaf_biology = LeafBiology(40, 0.01, 0.02, 0, 10, 10, 1.5)
    result = benchmark(PROSPECT_5D, leaf_biology, optical_params)


def test_benchmark_SAILH(benchmark, soil_optics, leaf_optics, optical_params):
    # Benchmark using default prospect, BSM, and SAILH values
    canopy_structure = CanopyStructure(3, -0.35, -0.15, 0.05)
    angles = Angles(40, 0, 0)

    result = benchmark(SAILH, soil_optics, leaf_optics, canopy_structure, angles)
    return result


@pytest.fixture
def soil_optics(optical_params, spectral_bands):
    # Modify the soil output for input to SAILH
    soil_parameters = SoilParameters(0.5, 0, 100, 20)
    soilopt = BSM(soil_parameters, optical_params)
    _rsoil = np.zeros((spectral_bands.nwlP + spectral_bands.nwlT, 1))
    _rsoil[spectral_bands.IwlT] = 1
    _rsoil[spectral_bands.IwlP] = soilopt.refl
    _rsoil[spectral_bands.IwlT] = 1 * _rsoil[spectral_bands.nwlP - 1]
    soilopt.refl = _rsoil
    return soilopt


@pytest.fixture
def leaf_optics(optical_params, spectral_bands):
    # Modify the PROSPECT output for input to SAILH
    leafbio = LeafBiology(40, 0.01, 0.02, 0, 10, 10, 1.5)
    leafopt = PROSPECT_5D(leafbio, optical_params)
    _rho = np.zeros((spectral_bands.nwlP + spectral_bands.nwlT, 1))
    _tau = np.zeros((spectral_bands.nwlP + spectral_bands.nwlT, 1))
    _rho[spectral_bands.IwlT] = leafbio.rho_thermal
    _tau[spectral_bands.IwlT] = leafbio.tau_thermal
    _rho[spectral_bands.IwlP] = leafopt.refl
    _tau[spectral_bands.IwlP] = leafopt.tran
    leafopt.refl = _rho
    leafopt.tran = _tau
    return leafopt


@pytest.fixture
def spectral_bands():
    return SpectralBands()
