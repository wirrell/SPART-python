import pytest
import SPART


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


def test_SPART(
    default_leaf_biology,
    default_canopy_structure,
    default_angles,
    default_soil_parameters,
    default_atmospheric_properties,
    sensor,
):
    spart = SPART.SPART(
        default_soil_parameters,
        default_leaf_biology,
        default_canopy_structure,
        default_atmospheric_properties,
        default_angles,
        sensor,
        100,
    )
    result = spart.run()
    assert (result["L_TOA"] > 0).all()
    assert (result["R_TOA"] > 0).all()
    assert (result["R_TOC"] > 0).all()


def test_SPART_concurrent_test_cases_run(default_SPARTSimulation):

    sims = [default_SPARTSimulation] * 10

    concurrent_SPART = SPART.SPARTConcurrent()

    concurrent_SPART.run_simulations(sims)
