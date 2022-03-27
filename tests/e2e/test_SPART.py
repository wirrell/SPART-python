import pytest
import SPART

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


@pytest.fixture(params=sensors)
def sensor(request):
    return request.param


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
