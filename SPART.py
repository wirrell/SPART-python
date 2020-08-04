"""
Soil-Plant-Atmosphere Radiative Transfer model
for top-of-canopy and top-of-atmosphere reflectance 

Coupling BSM, PROSAIL and SMAC to simulate TOA reflectance 

Python port coded by George Worrall (gworrall@ufl.edu)
Center for Remote Sensing, University of Florida

Ported to Python from the original matlab code and model developed by:

Peiqi Yang               (p.yang@utwente.nl)
Christiaan van der Tol   (c.vandertol@utwente.nl)
Wout Verhoef             (w.verhoef@utwente.nl)

University of Twente
Faculty of Geo-Information Science and Earth Observation (ITC), 
Department of Water Resources

"""
import pickle
import scipy.io
import numpy as np
from pathlib import Path
from BSM import BSM, SoilParameters
from PROSPECT_5D import PROSPECT_5D, LeafBiology
from SAILH import SAILH, CanopyStructure, Angles


class SPART:
    """
    SPART model.

    Parameters
    ----------
    soilpar : SoilParameters
        Holds the soil parameters for the BSM model.
    leafbio : LeafBiology
        Hold the leaf biology parameters for the PROSPECT model.
    canopy : CanopyStructure
        Holds the canopy parameters for the SAILH model.
    atm : AtmosphericProperties
        Holds the atmospheric properties for the SMAC model.
    sensor : str
        Name of RS platform to simulate. This is done after the SAIL stage and
        saves on atmospheric diffuse reflectance calculations for TOC and
        atmospheric correction calculations for TOA.
        Currently available:
            'TerraAqua-MODIS'
            'LANDSAT4-TM'
            'LANDSAT5-TM'
            'LANDSAT7-ETM'
            'LANDSAT8-OLI'
            'Sentinel3A-OLCI'
            'Sentinel3B-OLCI'
    DOY : int
        Day of the year - Julian calendar
    """
    # TODO: finish docstring with all params and attributes
    def __init__(self, soilpar, leafbio, canopy, atm, angles, sensor, DOY):
        self.soilpar = soilpar
        self.leafbio = leafbio
        self.canopy = canopy
        self.atm = atm
        self.angles = angles
        self.spectral = SpectralBands()
        self.optipar = load_optical_parameters()
        self.ETpar = load_ET_parameters()
        self.sensorinfo = load_sensor_info(sensor)
        self.sensor = sensor
        self.DOY = DOY

        # Initialize arrays

        # Leaf reflectance array, spans 400 to 2400 nm in 1 nm increments
        # then 2500 to 15000 in 100 nm increments
        # then 16000 to 50000 in 1000 nm increments
        self.rho = np.zeros((self.spectral.nwlP + self.spectral.nwlT, 1))
        # Leaf transmittance array, as above
        self.tau = np.zeros((self.spectral.nwlP + self.spectral.nwlT, 1))
        # Soil reflectance array, as above
        self.rsoil = np.zeros((self.spectral.nwlP + self.spectral.nwlT, 1))

        # Set the model reflectance and transmittance assumptions 
        self.set_refl_trans_assumptions()

        # Calculate extra-terrestrial radiance for the day
        Ra = calculate_ET_radiance(self.ETpar['Ea'], self.DOY,
                                         self.angles.sol_angle)
        self._La = calculate_spectral_convolution(self.ETpar['wl_Ea'], Ra,
                                                  self.sensorinfo)

    def set_refl_trans_assumptions(self):
        """Sets the model assumptions about soil and leaf reflectance and
        transmittance in the thermal range.

        These are that soil reflectance is the value for 2400 nm
        in the entire thermal range and that leaf relctance and
        transmittance are 0.01 in the thermal range (this is
        a model assumption that is set in the LeafBiology class in the BSM
        script)

        Returns
        -------
        None
        """
        self.rho[self.spectral.IwlT] = self.leafbio.rho_thermal
        self.tau[self.spectral.IwlT] = self.leafbio.tau_thermal
        self.rsoil[self.spectral.IwlT] = 1

    def run(self):
        """Run the SPART model.

        Parameters
        ----------

        Returns
        -------
        TODO
        """
        # TODO: write docstring
        # TODO: use get and setters to order the flow here. If something
        # has already been calculated then it can be skipped but if it has
        # been changed then it should be reset the skip flag.
        # TODO: look at threading BSM and PROSPECT as they don't require each
        # other

        # Run the BSM model
        soilopt = BSM(self.soilpar, self.optipar)
        self.rsoil[self.spectral.IwlP] = soilopt.refl

        # Run the PROSPECT model
        leafopt = PROSPECT_5D(self.leafbio, self.optipar)
        self.rho[self.spectral.IwlP] = leafopt.refl
        self.tau[self.spectral.IwlP] = leafopt.tran
        
        # Update soil and leaf optics refl and trans to include thermal values
        # from model assumptions
        self.rsoil[self.spectral.IwlT] = 1 * self.rsoil[self.spectral.nwlP - 1]
        soilopt.refl = self.rsoil
        leafopt.refl = self.rho
        leafopt.tran = self.tau

        # Calculate canopy radiances
        rad = SAILH(soilopt, leafopt, self.canopy, self.angles) 

        # Interpolate whole wavelength radiances to sensor wavlengths
        rv_so = np.interp(self.sensorinfo['wl_smac'].T[0], self.spectral.wlS,
                          rad.rso.T[0])
        rv_do = np.interp(self.sensorinfo['wl_smac'].T[0], self.spectral.wlS,
                          rad.rdo.T[0])
        rv_dd = np.interp(self.sensorinfo['wl_smac'].T[0], self.spectral.wlS,
                          rad.rdd.T[0])
        rv_sd = np.interp(self.sensorinfo['wl_smac'].T[0], self.spectral.wlS,
                          rad.rsd.T[0])


class SpectralBands:
    """
    Class to hold definitions of spectral band ranges and wavelengths.

    Attributes
    ----------
    wlP : np.array
        Range of wavelengths over which the PROSPECT model operates.
    wlE : np.array
        Range of wavelengths in E-F excitation matrix
    wlF : np.array
        Range of wavelengths for chlorophyll fluorescence in E-F matrix
    wlO : np.array
        Range of wavelengths in the optical part of the spectrum
    wlT : np.array
        Range of wavelengths in the thermal part of the spectrum
    wlS : np.array
        Range of wavelengths for the solar spectrum. wlO and wlT combined.
    wlPAR : np.array
        Range of wavelengths for photosynthetically active radiation
    nwlP : int
        Number of optical bands
    nwlT : int
        Number of thermal bands
    IwlP : range
        Index of optical bands
    IwlT : range
        Index of thermal bands
    """
    def __init__(self):
        self.wlP = np.arange(400, 2401, 1)
        self.wlE = np.arange(400, 751, 1)
        self.WlF = np.arange(640, 851, 1)
        self.wlO = np.arange(400, 2401, 1)
        self.wlT = np.concatenate([np.arange(2500, 15001, 100),
                                   np.arange(16000, 50001, 1000)])
        self.wlS = np.concatenate([self.wlO, self.wlT])
        self.wlPAR = np.arange(400, 701, 1)
        self.nwlP = len(self.wlP)
        self.nwlT = len(self.wlT)
        self.IwlP = np.arange(0, self.nwlP, 1)
        self.IwlT = np.arange(self.nwlP, self.nwlP + self.nwlT, 1)


def calculate_ET_radiance(Ea, DOY, tts):
    """
    Calculate extraterrestrial radiation.

    Parameters
    ----------
    Ea : float
        Solar constant for spectral irradiance
    DOY : int
        Day of year, Julian calendar
    tts : float
        Solar zenith angle in degrees

    Returns
    -------
    np.array
        Solar extraterrestrial spectrum

    NOTE
    ----
    see: https://www.sciencedirect.com/topics/engineering/
            extraterrestrial-radiation
    """
    # NOTE:
    # In the original code the below line contains 'DOY / 265' rather than
    # 'DOY / 365' however this is assumed to be a mistake and is corrected
    # here.
    b = 2 * np.pi * DOY / 365
    correction_factor = 1.00011 + 0.034221 * np.cos(b) + 0.00128 * np.sin(b) \
            + 0.000719 * np.cos(2 * b) + 0.000077 * np.sin(2 * b)
    La = Ea * correction_factor * np.cos(tts * np.pi / 180) / np.pi

    return La


def calculate_spectral_convolution(wl_hi, radiation_spectra, sensorinfo):
    """
    Calculate the spectral convolution for a given spectral response function.

    Parameters
    ----------
    wl_hi : np.array
        Arrays of wavelengths to be convolved
    radiation_spectra : np.array
        irradiance or radiance to be convolved
    sensorinfo : dict
        Contains keys 'wl_srf' -> number of bands contrib * number of bands
                                  post con
                      'p_srf' -> relative contribution of each band

    Returns
    -------
    np.array
        Convolution result
    """
    wl_srf = sensorinfo['wl_srf_smac']
    p_srf = sensorinfo['p_srf_smac']

    def get_closest_index(V, N):
        # Find the indices of the closest entries in N to the values of those
        # in V
        V = np.reshape(V, (V.shape[0] * V.shape[1], 1), order='F')
        A = np.repeat(N, len(V), axis=1)
        closest_index = np.argmin(np.abs(A - V.T), 0)
        return closest_index

    indx = get_closest_index(wl_srf, wl_hi)
    rad = np.reshape(radiation_spectra[indx], (wl_srf.shape[0], wl_srf.shape[1]),
                     order='F')
    # Sum and normalize as p_srf is not normalized.
    rad_conv = np.sum(rad * p_srf, axis=0) / np.sum(p_srf, axis=0)
    
    return rad_conv


def load_optical_parameters():
    """Load optical parameters from saved arrays"""
    params_file = Path(__file__).parent / 'model_parameters/optical_params.pkl'

    with open(params_file, 'rb') as f:
        params = pickle.load(f)

    return params


def load_ET_parameters():
    """Load extratrerrestrial parameters from saved arrays"""
    params_file = Path(__file__).parent / 'model_parameters/ET_irradiance.pkl'

    with open(params_file, 'rb') as f:
        params = pickle.load(f)

    return params


def load_sensor_info(sensor):
    """Load RS sensor information from saved arrays"""
    sensor_path = Path(__file__).parent / f'sensor_information/{sensor}.pkl'
    with open(sensor_path, 'rb') as f:
        sensor_info = pickle.load(f)
    return sensor_info

    if False:
        sdir = sensor_path.parent.glob('*.mat')
        # TODO remove when you know that we don't need to reformat pkl files
        for x in sdir:

            mat = scipy.io.loadmat(x)['sensor']
            sensor = {}
            fields = mat.dtype.names
            for i in range(len(fields)):
                sensor[fields[i]] = mat[0][0][i]

            smac_keys = sensor['SMAC_coef'].dtype.names
            smac_coef = sensor['SMAC_coef'].copy()

            sensor['SMAC_coef'] = {}
            for i in range(len(smac_keys)):
                sensor['SMAC_coef'][smac_keys[i]] = smac_coef[0][0][i]

            sensor['mission'] = sensor['mission'][0]
            sensor['name'] = sensor['name'][0]
            sensor['band_id_all'] = [mat[0][0] for mat in sensor['band_id_all']]
            sensor['band_id_smac'] = [mat[0][0] for mat in sensor['band_id_smac']]

            sensor_name = x.stem[18:]

            sensor_path = Path(__file__).parent / f'sensor_information/{sensor_name}.pkl'
            with open(sensor_path, 'wb') as f:
                sensor_info = pickle.dump(sensor, f)


if __name__ == '__main__':
    leafbio = LeafBiology(40, 10, 0.02, 0.01, 0, 10, 1.5)
    soilpar = SoilParameters(0.5, 0, 100, 15)
    canopy = CanopyStructure(3, -0.35, -0.15, 0.05)
    angles = Angles(40, 0, 0)
    spart = SPART(soilpar, leafbio, canopy, False, angles, 'TerraAqua-MODIS',
                  100)
    spart.run()
