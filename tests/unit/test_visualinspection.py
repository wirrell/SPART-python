import pytest
import matplotlib.pyplot as plt
import SPART
from SPART.prospect_5d import PROSPECT_5D
from SPART.sailh import SAILH


def test_visually_inspect_PROSPECT(default_leaf_biology, optical_params):
    result = PROSPECT_5D(default_leaf_biology, optical_params)
    plt.plot(range(400, 2401), result.refl.flatten(), label="Leaf reflectance")
    plt.plot(range(400, 2401), result.tran.flatten(), label="Leaf transmittance")
    plt.title("PROSPECT Test")
    plt.xlabel("Wavelength (nm)")
    plt.legend()
    plt.show()


def test_visually_inspect_SAILH(
    default_leaf_optics,
    default_soil_optics,
    default_canopy_structure,
    default_angles,
    optical_params,
):
    result = SAILH(
        default_soil_optics,
        default_leaf_optics,
        default_canopy_structure,
        default_angles,
    )
    plt.plot(
        range(400, 2401),
        result.rso.flatten()[:2001],
        label="Canopy bidirectional reflectance",
    )
    plt.plot(
        range(400, 2401),
        result.rdo.flatten()[:2001],
        label="Canopy directional reflectance of diffuse light",
    )
    plt.plot(
        range(400, 2401),
        result.rsd.flatten()[:2001],
        label="Canopy diffuse reflectance of specular incidence",
    )
    plt.plot(
        range(400, 2401),
        result.rdd.flatten()[:2001],
        label="Canopy diffuse reflectance of diffuse incidence",
    )
    plt.title("SAILH Test")
    plt.xlabel("Wavelength (nm)")
    plt.legend()
    plt.show()


def test_visually_inspect_SPART(
    default_leaf_biology,
    default_canopy_structure,
    default_angles,
    default_soil_parameters,
    default_atmospheric_properties,
):
    spart = SPART.SPART(
        default_soil_parameters,
        default_leaf_biology,
        default_canopy_structure,
        default_atmospheric_properties,
        default_angles,
        "Sentinel2A-MSI",
        100,
    )
    result = spart.run()
    plt.plot(result["L_TOA"], label="Radiance, top-of-atmosphere")
    plt.plot(result["R_TOA"], label="Reflectance, top-of-atmosphere")
    plt.plot(result["R_TOC"], label="Reflectance, top-of-canopy")
    plt.legend()
    plt.title("SPART outputs for Sentinel-2 sensor")
    plt.show()


def test_visually_compare_SAILH_SPART(
    default_leaf_optics,
    default_soil_optics,
    optical_params,
    default_leaf_biology,
    default_canopy_structure,
    default_angles,
    default_soil_parameters,
    default_atmospheric_properties,
):
    spart = SPART.SPART(
        default_soil_parameters,
        default_leaf_biology,
        default_canopy_structure,
        default_atmospheric_properties,
        default_angles,
        "Sentinel2A-MSI",
        100,
    )
    result = SAILH(
        default_soil_optics,
        default_leaf_optics,
        default_canopy_structure,
        default_angles,
    )
    plt.plot(
        range(400, 2401),
        result.rso.flatten()[:2001],
        label="Canopy bidirectional reflectance",
    )
    plt.plot(
        range(400, 2401),
        result.rdo.flatten()[:2001],
        label="Canopy directional reflectance of diffuse light",
    )
    plt.plot(
        range(400, 2401),
        result.rsd.flatten()[:2001],
        label="Canopy diffuse reflectance of specular incidence",
    )
    plt.plot(
        range(400, 2401),
        result.rdd.flatten()[:2001],
        label="Canopy diffuse reflectance of diffuse incidence",
    )
    result = spart.run()
    plt.scatter(
        result.index,
        result["R_TOA"],
        label="Reflectance, top-of-atmosphere",
        marker="X",
        color="black",
    )
    plt.scatter(
        result.index,
        result["R_TOC"],
        label="Reflectance, top-of-canopy",
        marker="o",
        color="red",
    )
    plt.title("Satellite bands versus full canopy reflecntance")
    plt.xlabel("Wavelength (nm)")
    plt.legend()
    plt.show()
