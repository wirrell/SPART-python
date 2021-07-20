import pytest
import spart
import pandas as pd
import numpy as np


class SPART_Parameters(object):
    """
    class defining which leaf, canopy, soil and atmosphere parameters
    are required to run SPART simulations.
    This class helps mapping the entries of the CSV with the SPART model
    parameterization to the actual SPART function call so that users do not
    have to take care about the order of parameters.
    """
    # define the entries required to run the SPART submodels
    SMB = ['B', 'lat', 'lon', 'SMp', 'SMC', 'film']  # soil model
    prospect_5d = ['Cab', 'Cca', 'Cw', 'Cdm', 'Cs', 'Cant', 'N', 'PROT', 'CBC']  # leaf model
    sailh = ['LAI', 'LIDFa', 'LIDFb', 'q']  # canopy model
    SMAC = ['aot550', 'uo3', 'uh2o', 'Pa']  # atmosphere model
    angles = ['sol_angle', 'obs_angle', 'rel_angle']  # sunobserver geometry


@pytest.fixture()
def run_spart():
    """fixture returning function prototype to run the SPART model"""
    
    def _run_spart(parameters: pd.Series,
                  sensor_name: str,
                  doy: int=100,
                  debug: bool=True
                  ) -> pd.DataFrame:
        """
        runs the SPART model on a single parameter combination specified as pandas series
        and stores the results as a pandas dataframe appended to the passed series
    
        NOTE: SPART always calculates three result sets - R_TOC, R_TOA, L_TOA:
            - R_TOC : top of canopy reflectance [%], aka bottom-of-atmosphere reflectance
            - R_TOA : top of atmosphere reflectance [%]
            - L_TOA : top of atmosphere radiance [W sr-1 m-2]
        All these outputs are stored in the 'simulated' entry of the series.
    
        R_TOC and R_TOA range between 0 and 1; thus a scaling factor of 100 might be required
        to convert the measurements to real percent values
    
        :param parameters:
            k-v pairs as pandas Series required to run SPART Python
        :param sensor_name:
            name of the sensor for which simulations are run
        :param doy:
            day of year (Julian calendar) of the simulation (Def=100)
        """
        spart_params = SPART_Parameters()
        # extract parameters for running the single SPART submodels by name
        soil_params = parameters[spart_params.SMB].to_dict()
        leaf_params = parameters[spart_params.prospect_5d].to_dict()
        canopy_params = parameters[spart_params.sailh].to_dict()
        atm_params = parameters[spart_params.SMAC].to_dict()
        angles = parameters[spart_params.angles].to_dict()
    
        # call the submodules to simulate the single levels
        soil_prop = spart.SoilParameters(**soil_params)
        leaf_prop = spart.LeafBiology(**leaf_params)
        canopy_prop = spart.CanopyStructure(**canopy_params)
        atm_prop = spart.AtmosphericProperties(**atm_params)
        geometry = spart.Angles(**angles)
    
        # run the entire model
        model = spart.SPART(soilpar=soil_prop,
                            leafbio=leaf_prop,
                            canopy=canopy_prop,
                            atm=atm_prop,
                            angles=geometry,
                            sensor=sensor_name,
                            DOY=doy)
        # return soil reflectance values from BSM if debug is set to True
        simulated = model.run(debug=debug)
        return simulated
    
    return _run_spart


@pytest.mark.parametrize('fname_spart_params',
                        ['SPART_Prospect5D.csv',
                         'SPART_ProspectPRO.csv'])
@pytest.mark.parametrize('sensor_name',
                        ['Sentinel2A-MSI',
                         'Sentinel2B-MSI',
                         'LANDSAT4-TM',
                         'LANDSAT5-TM',
                         'LANDSAT7-ETM',
                         'LANDSAT8-OLI',
                         'Sentinel3A-OLCI',
                         'Sentinel3B-OLCI',
                         'TerraAqua-MODIS'])
def test_run_spart(datadir, run_spart, fname_spart_params, sensor_name):
    """
    Run SPART simulations on a number of input parameter combinations
    Test both available leaf models, Prospect5-D and Prospect-Pro for
    the available remote sensing sensors.
    """

    # load SPART input parameters from the testdatadir
    testdata_file = datadir.join(fname_spart_params)
    spart_params = pd.read_csv(testdata_file)

    # run simulations and check soil spectra (debug)
    for idx, params in spart_params.iterrows():
        
            sim_specs = run_spart(parameters=params,
                                  sensor_name=sensor_name)
            # assert that simulated spectra are physically plausible
            # (small negative values can result from extreme atmospheric conditions like
            # extremely high aot500 values or unrealistic air pressure values;
            # personnel communication with Olivier Hagolle from CNES (France) on 10th June 2021
            assert all(sim_specs['L_TOA'] > -0.15), \
                f'negative simulated TOA irradiances (spectrum #{idx})'
            assert all(sim_specs['R_TOA'] > -0.15), \
                f'negative simulated TOA reflectances (spectrum #{idx})'
            assert all(sim_specs['R_TOC'] > -0.15), \
                f'negative simulated TOC reflectances (spectrum #{idx})'
            assert all(sim_specs['rsoil'] > -0.15), \
                f'negative simulated soil reflectances (spectrum #{idx})'
