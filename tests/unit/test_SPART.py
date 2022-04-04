import pytest
import numpy as np

from SPART import load_sensor_info, load_ET_parameters
from SPART import calculate_ET_radiance, calculate_spectral_convolution



def test_calculate_spectral_convolution(sensor,
                                        default_SPARTSimulation):
    ETpar = load_ET_parameters()
    simulations = [default_SPARTSimulation] * 10
    sensor_infos = [load_sensor_info(sim.sensor) for sim in simulations]

    sol_angles = np.array([sim.angles.sol_angle for sim in simulations])
    wl_srf = np.array([s_info['wl_srf_smac'] for s_info in sensor_infos])
    p_srf = np.array([s_info['p_srf_smac'] for s_info in sensor_infos])
    DOY = np.array([sim.DOY for sim in simulations])

    Ra = calculate_ET_radiance(ETpar['Ea'], 100, sol_angles)

    # Expected single result
    exp = np.load(f'test_SPART/{sensor_infos[0]["name"]}_default_spectral_conv.npy')

    La = calculate_spectral_convolution(ETpar['wl_Ea'], Ra, wl_srf,
                                        p_srf)
    assert (La[0] == exp).all()

