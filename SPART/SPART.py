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
import numpy as np
import pandas as pd
from pathlib import Path
from .BSM import BSM, SoilParameters, SoilParametersFromFile
from .PROSPECT_5D import PROSPECT_5D, LeafBiology
from .SAILH import SAILH, CanopyStructure, Angles
from .SMAC import SMAC, AtmosphericProperties

# TODO: look at threading BSM and PROSPECT as they don't require each
# other

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

    Attributes
    ----------
    soilopt : BSM.SoilOptics
        Contains soil reflectances
    leafopt : PROSPECT_5D.LeafOptics
        Contains leaf reflectance and transmittance and fraction contributed
        by chlorphyll
    canopyopt : SAILH.CanopyReflectances
        Contains bidrectional and directional, diffuce and specular reflectance
    R_TOC : np.array
        Top of canopy reflectance
    R_TOA : np.array
        Top of atmosphere reflectance
    L_TOA : np.array
        Top of atmosphere radiance
    """
    def __init__(self, soilpar, leafbio, canopy, atm, angles, sensor, DOY):
        self._tracker = {}  # tracks changes to input parameters
        self.soilpar = soilpar
        self.leafbio = leafbio
        self.canopy = canopy
        self.atm = atm
        self.angles = angles
        self.sensor = sensor
        self.DOY = DOY
        self.spectral = SpectralBands()
        self.optipar = load_optical_parameters()
        self.ETpar = load_ET_parameters()
        self.sensorinfo = load_sensor_info(sensor)

        # Initialize arrays

        # Leaf reflectance array, spans 400 to 2400 nm in 1 nm increments
        # then 2500 to 15000 in 100 nm increments
        # then 16000 to 50000 in 1000 nm increments
        self._rho = np.zeros((self.spectral.nwlP + self.spectral.nwlT, 1))
        # Leaf transmittance array, as above
        self._tau = np.zeros((self.spectral.nwlP + self.spectral.nwlT, 1))
        # Soil reflectance array, as above
        self._rsoil = np.zeros((self.spectral.nwlP + self.spectral.nwlT, 1))

        # Set the model reflectance and transmittance assumptions
        self.set_refl_trans_assumptions()


    @property
    def soilpar(self):
        return self._soilpar

    @soilpar.setter
    def soilpar(self, soilpar):
        self._soilpar = soilpar
        self._tracker['soil'] = True

    @property
    def leafbio(self):
        return self._leafbio

    @leafbio.setter
    def leafbio(self, leafbio):
        self._leafbio = leafbio
        self._tracker['leaf'] = True

    @property
    def canopy(self):
        return self._canopy

    @canopy.setter
    def canopy(self, canopy):
        self._canopy = canopy
        self._tracker['canp'] = True

    @property
    def atm(self):
        return self._atm

    @atm.setter
    def atm(self, atm):
        self._atm = atm
        self._tracker['atm'] = True

    @property
    def angles(self):
        return self._angles

    @angles.setter
    def angles(self, angles):
        self._angles = angles
        self._tracker['angles'] = True

    @property
    def sensor(self):
        return self._sensor

    @sensor.setter
    def sensor(self, sensor):
        self._sensor = sensor
        self._tracker['sensor'] = True

    @property
    def DOY(self):
        return self._DOY

    @DOY.setter
    def DOY(self, DOY):
        self._DOY = DOY
        self._tracker['DOY'] = True

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
        self._rho[self.spectral.IwlT] = self.leafbio.rho_thermal
        self._tau[self.spectral.IwlT] = self.leafbio.tau_thermal
        self._rsoil[self.spectral.IwlT] = 1

    def run(self):
        """Run the SPART model.

        Returns
        -------
        pd.DataFrame
            Contains the radiances and reflectances columns 'Band' 'L_TOA'
            'R_TOA' 'R_TOC' index by central band wavelength
        """
        # Calculate ET radiance from the sun fo look angles and DOY
        if self._tracker['DOY'] or self._tracker['angles']:
            # Calculate extra-terrestrial radiance for the day
            Ra = calculate_ET_radiance(self.ETpar['Ea'], self.DOY,
                                       self.angles.sol_angle)
            self._La = calculate_spectral_convolution(self.ETpar['wl_Ea'], Ra,
                                                      self.sensorinfo)
            self._tracker['DOY'] = False
            self._tracker['angles'] = False

        # Run the BSM model
        if self._tracker['soil']:
            soilopt = BSM(self._soilpar, self.optipar)
            # Update soil optics refl and trans to include thermal
            # values from model assumptions
            self._rsoil[self.spectral.IwlP] = soilopt.refl
            self._rsoil[self.spectral.IwlT] = 1 * self._rsoil[self.spectral.nwlP
                                                            - 1]
            soilopt.refl = self._rsoil
            self.soilopt = soilopt
            self._tracker['soil'] = False

        # Run the PROSPECT model
        if self._tracker['leaf']:
            leafopt = PROSPECT_5D(self._leafbio, self.optipar)
            # Update leaf optics refl and trans to include thermal
            # values from model assumptions
            self._rho[self.spectral.IwlP] = leafopt.refl
            self._tau[self.spectral.IwlP] = leafopt.tran
            leafopt.refl = self._rho
            leafopt.tran = self._tau
            self.leafopt = leafopt
            self._tracker['leaf'] = False


        # Run the SAIL model
        rad = SAILH(self.soilopt, self.leafopt, self._canopy, self._angles)
        self.canopyopt = rad

        sensor_wavelengths = self.sensorinfo['wl_smac'].T[0]

        # Interpolate whole wavelength radiances to sensor wavlengths
        rv_so = np.interp(sensor_wavelengths, self.spectral.wlS,
                          rad.rso.T[0])
        rv_do = np.interp(sensor_wavelengths, self.spectral.wlS,
                          rad.rdo.T[0])
        rv_dd = np.interp(sensor_wavelengths, self.spectral.wlS,
                          rad.rdd.T[0])
        rv_sd = np.interp(sensor_wavelengths, self.spectral.wlS,
                          rad.rsd.T[0])

        # Run the SMAC atmosphere model
        if (self._tracker['atm'] or self._tracker['angles'] or
            self._tracker['sensor']):
            atmopt = SMAC(self._angles, self._atm, self.sensorinfo['SMAC_coef'])
            self.atmopt = atmopt
            self._tracker['atm'] = False
            self._tracker['angles'] = False
            self._tracker['sensor'] = False

        # Upscale TOC to TOA
        ta_ss = self.atmopt.Ta_ss
        ta_sd = self.atmopt.Ta_sd
        ta_oo = self.atmopt.Ta_oo
        ta_do = self.atmopt.Ta_do
        ra_dd = self.atmopt.Ra_dd
        ra_so = self.atmopt.Ra_so
        T_g = self.atmopt.Tg

        rtoa0 = ra_so + ta_ss * rv_so * ta_oo
        rtoa1 = (ta_sd * rv_do + ta_ss * rv_sd * ra_dd * rv_do) * ta_oo / \
                (1 - rv_dd * ra_dd)
        rtoa2 = (ta_ss * rv_sd + ta_sd * rv_dd) * ta_do / (1 - rv_dd * ra_dd)
        self.R_TOC = (ta_ss * rv_so + ta_sd * rv_do) / (ta_ss + ta_sd)
        self.R_TOA = T_g * (rtoa0 + rtoa1 + rtoa2)
        self.L_TOA = self._La * self.R_TOA

        bands = self.sensorinfo['band_id_smac']

        results_table = pd.DataFrame(zip(bands, self.L_TOA[0], self.R_TOA[0],
                                         self.R_TOC[0]),
                                     index=sensor_wavelengths,
                                     columns=['Band', 'L_TOA', 'R_TOA', 'R_TOC'])

        return results_table


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
    rad = np.reshape(radiation_spectra[indx], (wl_srf.shape[0],
                                               wl_srf.shape[1]),
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


if __name__ == '__main__':
    leafbio = LeafBiology(40, 10, 0.02, 0.01, 0, 10, 1.5)
    soilpar = SoilParameters(0.5, 0, 100, 15)
    canopy = CanopyStructure(3, -0.35, -0.15, 0.05)
    angles = Angles(40, 0, 0)
    atm = AtmosphericProperties(0.3246, 0.3480, 1.4116, 1013.25)
    spart = SPART(soilpar, leafbio, canopy, atm, angles, 'LANDSAT8-OLI',
                  100)
    print(spart.run())
