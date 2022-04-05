import pytest
import numpy as np

from SPART.prospect_5d import LeafBiology, PROSPECT_5D
from SPART.bsm import SoilParameters, BSM
from SPART.sailh import CanopyStructure, SAILH, Angles
from SPART.SPART import SpectralBands


def test_benchmark_PROSPECT_5D_one_sim(benchmark, optical_params):
    # Benchmark using default prospect values
    result = benchmark(
        PROSPECT_5D,
        40,
        0.01,
        0.02,
        0,
        10,
        10,
        1.5,
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


def test_benchmark_PROSPECT_5D_100_sim(benchmark, optical_params):
    # Benchmark using default prospect values
    num_sims = 100
    result = benchmark(
        PROSPECT_5D,
        np.array([40] * num_sims),
        np.array([0.01] * num_sims),
        np.array([0.02] * num_sims),
        np.array([0] * num_sims),
        np.array([10] * num_sims),
        np.array([10] * num_sims),
        np.array([1.5] * num_sims),
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


def test_benchmark_SAILH_one_sim(
    benchmark, default_soil_optics, default_leaf_optics, optical_params
):
    # Benchmark using default prospect, BSM, and SAILH values
    canopy_structure = CanopyStructure(3, -0.35, -0.15, 0.05)
    angles = Angles(40, 0, 0)

    result = benchmark(
        SAILH,
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
    return result


def test_benchmark_SAILH_100_sim_CUDA(
    benchmark, default_soil_optics, default_leaf_optics, optical_params
):
    # Benchmark using default prospect, BSM, and SAILH values
    canopy_structure = CanopyStructure(3, -0.35, -0.15, 0.05)
    angles = Angles(40, 0, 0)

    concurrent_tests = 100
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


    result = benchmark(
        SAILH,
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
    return result


def test_benchmark_SAILH_100_sim(
    benchmark, default_soil_optics, default_leaf_optics, optical_params
):
    # Benchmark using default prospect, BSM, and SAILH values
    canopy_structure = CanopyStructure(3, -0.35, -0.15, 0.05)
    angles = Angles(40, 0, 0)

    concurrent_tests = 100
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


    result = benchmark(
        SAILH,
        soil_refl,
        leaf_refl,
        leaf_tran,
        np.array([canopy_structure.nlayers] * concurrent_tests),
        np.array([canopy_structure.LAI] * concurrent_tests),
        lidf,
        np.array([angles.sol_angle] * concurrent_tests),
        np.array([angles.obs_angle] * concurrent_tests),
        np.array([angles.rel_angle] * concurrent_tests),
        np.array([canopy_structure.q] * concurrent_tests)
    )
    return result


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
