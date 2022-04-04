import pytest
import numpy as np
from SPART import load_optical_parameters
from SPART.bsm import BSM


def test_BSM(default_soil_parameters):

    optical_params = load_optical_parameters()

    rwet, rdry = BSM(
        default_soil_parameters.SMp,
        default_soil_parameters.SMC,
        default_soil_parameters.film,
        optical_params["GSV"],
        default_soil_parameters.B,
        default_soil_parameters.lon,
        default_soil_parameters.lat,
        optical_params["Kw"],
        optical_params["nw"],
    )


def test_BSM_multiple_sims(default_soil_parameters):
    optical_params = load_optical_parameters()

    rwet, rdry = BSM(
        default_soil_parameters.SMp,
        default_soil_parameters.SMC,
        default_soil_parameters.film,
        optical_params["GSV"],
        default_soil_parameters.B,
        default_soil_parameters.lon,
        default_soil_parameters.lat,
        optical_params["Kw"],
        optical_params["nw"],
    )
    rwet, rdry = BSM(
        np.array([default_soil_parameters.SMp] * 10),
        np.array([default_soil_parameters.SMC] * 10),
        np.array([default_soil_parameters.film] * 10),
        optical_params["GSV"],
        np.array([default_soil_parameters.B] * 10),
        np.array([default_soil_parameters.lon] * 10),
        np.array([default_soil_parameters.lat] * 10),
        optical_params["Kw"],
        optical_params["nw"],
    )
