"""
Soil-Plant-Atmosphere Radiative Transfer model
for top-of-canopy and top-of-atmosphere reflectance

Coupling bsm, PROSAIL and smac to simulate TOA reflectance

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
import nvtx
from pathlib import Path
from typing import List
from dataclasses import dataclass
from SPART.bsm import BSM, SoilParameters, SoilParametersFromFile
from SPART.prospect_5d import PROSPECT_5D, LeafBiology
from SPART.sailh import SAILH, CanopyStructure, Angles
from SPART.smac import SMAC, AtmosphericProperties


# TODO: look at threading bsm and PROSPECT as they don't require each
# other


@dataclass
class SPARTSimulation:
    """
    Holds all the different parameter classes for the constituent models.
    """

    soilpar: SoilParameters
    leafbio: LeafBiology
    canopy: CanopyStructure
    atm: AtmosphericProperties
    angles: Angles
    sensor: str
    DOY: int


class SPARTConcurrent:
    """
    Concurrent implementation of SPART model.
    """

    def __init__(self):
        self.spectral = SpectralBands()
        self.optipar = load_optical_parameters()
        self.ETpar = load_ET_parameters()

    def run_simulations(self, simulations: List[SPARTSimulation]):
        """Run the run_spart model for all simulations.

        Parameters
        ----------
        debug : bool
            if True, returns the simulated BSM derived soil spectra as well
            as additional output column. Default: False

        Returns
        -------
        list of pd.DataFrame
            Contains the radiances and reflectances columns 'Band' 'L_TOA'
            'R_TOA' 'R_TOC' index by central band wavelength
        """

        num_simulations = len(simulations)

        sensor_infos = [load_sensor_info(sim.sensor) for sim in simulations]

        # Calculate ET radiance from the sun for look angles and DOY
        # Calculate extra-terrestrial radiance for the day
        sol_angles = np.array([sim.angles.sol_angle for sim in simulations])
        wl_srf = np.array([s_info["wl_srf_smac"] for s_info in sensor_infos])
        p_srf = np.array([s_info["p_srf_smac"] for s_info in sensor_infos])
        DOY = np.array([sim.DOY for sim in simulations])

        Ra = calculate_ET_radiance(self.ETpar["Ea"], DOY, sol_angles)
        La = calculate_spectral_convolution(self.ETpar["wl_Ea"], Ra, wl_srf, p_srf)

        # Calculate soil optics of each
        soilpars = np.array([sim.soilpar for sim in simulations])

        soilopts = [BSM(soilpar, self.optipar) for soilpar in soilpars]
        # Update soil optics refl and trans to include thermal
        # values from model assumptions
        soilopts = [
            set_soil_refl_trans_assumptions(soilopt, self.spectral)
            for soilopt in soilopts
        ]

        leafopt = PROSPECT_5D(self._leafbio, self.optipar)
        # Update leaf optics refl and trans to include thermal
        # values from model assumptions
        with nvtx.annotate("Assign leaf assumptions", color="purple"):
            self.leafopt = set_leaf_refl_trans_assumptions(
                leafopt, self.leafbio, self.spectral
            )
        self._tracker["leaf"] = False

        # Run the SAIL model
        with nvtx.annotate("SAILH model run", color="green"):
            rad = SAILH(self.soilopt, self.leafopt, self._canopy, self._angles)
        self.canopyopt = rad

        sensor_wavelengths = self.sensorinfo["wl_smac"].T[0]

        # Interpolate whole wavelength radiances to sensor wavlengths
        with nvtx.annotate(
            "Interpolating SAILH outputs to sensor wavelengths", color="purple"
        ):
            rv_so = np.interp(sensor_wavelengths, self.spectral.wlS, rad.rso.T[0])
            rv_do = np.interp(sensor_wavelengths, self.spectral.wlS, rad.rdo.T[0])
            rv_dd = np.interp(sensor_wavelengths, self.spectral.wlS, rad.rdd.T[0])
            rv_sd = np.interp(sensor_wavelengths, self.spectral.wlS, rad.rsd.T[0])

        # Run the smac atmosphere model
        if self._tracker["atm"] or self._tracker["angles"] or self._tracker["sensor"]:
            with nvtx.annotate("SMAC model run", color="blue"):
                atmopt = SMAC(self._angles, self._atm, self.sensorinfo["SMAC_coef"])
            self.atmopt = atmopt
            self._tracker["atm"] = False
            self._tracker["angles"] = False
            self._tracker["sensor"] = False

        # Upscale TOC to TOA
        ta_ss = self.atmopt.Ta_ss
        ta_sd = self.atmopt.Ta_sd
        ta_oo = self.atmopt.Ta_oo
        ta_do = self.atmopt.Ta_do
        ra_dd = self.atmopt.Ra_dd
        ra_so = self.atmopt.Ra_so
        T_g = self.atmopt.Tg

        rtoa0 = ra_so + ta_ss * rv_so * ta_oo
        rtoa1 = (
            (ta_sd * rv_do + ta_ss * rv_sd * ra_dd * rv_do)
            * ta_oo
            / (1 - rv_dd * ra_dd)
        )
        rtoa2 = (ta_ss * rv_sd + ta_sd * rv_dd) * ta_do / (1 - rv_dd * ra_dd)
        self.R_TOC = (ta_ss * rv_so + ta_sd * rv_do) / (ta_ss + ta_sd)
        self.R_TOA = T_g * (rtoa0 + rtoa1 + rtoa2)
        self.L_TOA = self._La * self.R_TOA

        bands = self.sensorinfo["band_id_smac"]

        results_table = pd.DataFrame(
            zip(bands, self.L_TOA[0], self.R_TOA[0], self.R_TOC[0]),
            index=sensor_wavelengths,
            columns=["Band", "L_TOA", "R_TOA", "R_TOC"],
        )

        return results_table


class SPART:
    """
    run_spart model.

    Parameters
    ----------
    soilpar : SoilParameters
        Holds the soil parameters for the bsm model.
    leafbio : LeafBiology
        Hold the leaf biology parameters for the PROSPECT model.
    canopy : CanopyStructure
        Holds the canopy parameters for the sailh model.
    atm : AtmosphericProperties
        Holds the atmospheric properties for the smac model.
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
        'Sentinel2A-MSI'
        'Sentinel2B-MSI'
        'Sentinel3A-OLCI'
        'Sentinel3B-OLCI'
    DOY : int
        Day of the year - Julian calendar

    Attributes
    ----------
    soilopt : bsm.SoilOptics
        Contains soil reflectances
    leafopt : prospect_5d.LeafOptics
        Contains leaf reflectance and transmittance and fraction contributed
        by chlorophyll
    canopyopt : sailh.CanopyReflectances
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

    @property
    def soilpar(self):
        return self._soilpar

    @soilpar.setter
    def soilpar(self, soilpar):
        self._soilpar = soilpar
        self._tracker["soil"] = True

    @property
    def leafbio(self):
        return self._leafbio

    @leafbio.setter
    def leafbio(self, leafbio):
        self._leafbio = leafbio
        self._tracker["leaf"] = True

    @property
    def canopy(self):
        return self._canopy

    @canopy.setter
    def canopy(self, canopy):
        self._canopy = canopy
        self._tracker["canp"] = True

    @property
    def atm(self):
        return self._atm

    @atm.setter
    def atm(self, atm):
        self._atm = atm
        self._tracker["atm"] = True

    @property
    def angles(self):
        return self._angles

    @angles.setter
    def angles(self, angles):
        self._angles = angles
        self._tracker["angles"] = True

    @property
    def sensor(self):
        return self._sensor

    @sensor.setter
    def sensor(self, sensor):
        self._sensor = sensor
        self._tracker["sensor"] = True

    @property
    def DOY(self):
        return self._DOY

    @DOY.setter
    def DOY(self, DOY):
        self._DOY = DOY
        self._tracker["DOY"] = True

    def run(self, debug=False):
        """Run the run_spart model.

        Parameters
        ----------
        debug : bool
            if True, returns the simulated BSM derived soil spectra as well
            as additional output column. Default: False

        Returns
        -------
        pd.DataFrame
            Contains the radiances and reflectances columns 'Band' 'L_TOA'
            'R_TOA' 'R_TOC' index by central band wavelength
        """
        # Calculate ET radiance from the sun fo look angles and DOY
        if self._tracker["DOY"] or self._tracker["angles"]:
            # Calculate extra-terrestrial radiance for the day
            Ra = calculate_ET_radiance(
                self.ETpar["Ea"], self.DOY, self.angles.sol_angle
            )
            self._La = calculate_spectral_convolution(
                self.ETpar["wl_Ea"], Ra, self.sensorinfo
            )
            self._tracker["DOY"] = False
            self._tracker["angles"] = False

        # Run the bsm model
        if self._tracker["soil"]:
            with nvtx.annotate("BSM model run", color="red"):
                soilopt = BSM(self._soilpar, self.optipar)
            # Update soil optics refl and trans to include thermal
            # values from model assumptions
            with nvtx.annotate("Assign soil assumptions", color="purple"):
                self.soilopt = set_soil_refl_trans_assumptions(soilopt, self.spectral)
            self._tracker["soil"] = False

        # Run the PROSPECT model
        if self._tracker["leaf"]:
            with nvtx.annotate("PROSPECT 5D model run", color="yellow"):
                leafopt = PROSPECT_5D(self._leafbio, self.optipar)
            # Update leaf optics refl and trans to include thermal
            # values from model assumptions
            with nvtx.annotate("Assign leaf assumptions", color="purple"):
                self.leafopt = set_leaf_refl_trans_assumptions(
                    leafopt, self.leafbio, self.spectral
                )
            self._tracker["leaf"] = False

        # Run the SAIL model
        with nvtx.annotate("SAILH model run", color="green"):
            rad = SAILH(self.soilopt, self.leafopt, self._canopy, self._angles)
        self.canopyopt = rad

        sensor_wavelengths = self.sensorinfo["wl_smac"].T[0]

        # Interpolate whole wavelength radiances to sensor wavlengths
        with nvtx.annotate(
            "Interpolating SAILH outputs to sensor wavelengths", color="purple"
        ):
            rv_so = np.interp(sensor_wavelengths, self.spectral.wlS, rad.rso.T[0])
            rv_do = np.interp(sensor_wavelengths, self.spectral.wlS, rad.rdo.T[0])
            rv_dd = np.interp(sensor_wavelengths, self.spectral.wlS, rad.rdd.T[0])
            rv_sd = np.interp(sensor_wavelengths, self.spectral.wlS, rad.rsd.T[0])

        # Run the smac atmosphere model
        if self._tracker["atm"] or self._tracker["angles"] or self._tracker["sensor"]:
            with nvtx.annotate("SMAC model run", color="blue"):
                atmopt = SMAC(self._angles, self._atm, self.sensorinfo["SMAC_coef"])
            self.atmopt = atmopt
            self._tracker["atm"] = False
            self._tracker["angles"] = False
            self._tracker["sensor"] = False

        # Upscale TOC to TOA
        ta_ss = self.atmopt.Ta_ss
        ta_sd = self.atmopt.Ta_sd
        ta_oo = self.atmopt.Ta_oo
        ta_do = self.atmopt.Ta_do
        ra_dd = self.atmopt.Ra_dd
        ra_so = self.atmopt.Ra_so
        T_g = self.atmopt.Tg

        rtoa0 = ra_so + ta_ss * rv_so * ta_oo
        rtoa1 = (
            (ta_sd * rv_do + ta_ss * rv_sd * ra_dd * rv_do)
            * ta_oo
            / (1 - rv_dd * ra_dd)
        )
        rtoa2 = (ta_ss * rv_sd + ta_sd * rv_dd) * ta_do / (1 - rv_dd * ra_dd)
        self.R_TOC = (ta_ss * rv_so + ta_sd * rv_do) / (ta_ss + ta_sd)
        self.R_TOA = T_g * (rtoa0 + rtoa1 + rtoa2)
        self.L_TOA = self._La * self.R_TOA

        bands = self.sensorinfo["band_id_smac"]

        results_table = pd.DataFrame(
            zip(bands, self.L_TOA[0], self.R_TOA[0], self.R_TOC[0]),
            index=sensor_wavelengths,
            columns=["Band", "L_TOA", "R_TOA", "R_TOC"],
        )

        if debug:
            # wet soil reflectance
            rsoil = np.interp(
                sensor_wavelengths, self.spectral.wlS, self.soilopt.refl[:, 0]
            )
            results_table["rsoil"] = rsoil

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
        self.wlT = np.concatenate(
            [np.arange(2500, 15001, 100), np.arange(16000, 50001, 1000)]
        )
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
    correction_factor = (
        1.00011
        + 0.034221 * np.cos(b)
        + 0.00128 * np.sin(b)
        + 0.000719 * np.cos(2 * b)
        + 0.000077 * np.sin(2 * b)
    )
    La = Ea * correction_factor * np.cos(tts * np.pi / 180) / np.pi

    return La


def calculate_spectral_convolution(wl_hi, radiation_spectra, wl_srf, p_srf):
    """
    Calculate the spectral convolution for a given spectral response function.

    Parameters
    ----------
    wl_hi : np.array
        Arrays of wavelengths to be convolved
    radiation_spectra : np.array
        irradiance or radiance to be convolved
    wl_srf : np.array
        number of bands contrib * number of bands post con
    p_srf : np.array 
        relative contribution of each band

    Returns
    -------
    np.array
        Convolution result
    """
    # reshape wl_hi array to dimensions of radiation_spectra

    def get_closest_index(V, N):
        # Find the indices of the closest entries in N to the values of those
        # in V
        V = np.reshape(V, (V.shape[0], V.shape[1] * V.shape[2]), order="F")
        A = np.repeat(N, V.shape[1], axis=1)
        # add new axis to form matrices so we can do simulataneous argmin
        V = V[:, np.newaxis, :]
        A = np.repeat(A[np.newaxis, :, :], V.shape[0], axis=0)
        closest_index = np.argmin(np.abs(A - V), 1)
        return closest_index

    indx = get_closest_index(wl_srf, wl_hi)
    # NOTE: shape was coming out 3D, we want 2D and it appears last index was just duplicate
    rad_idx = radiation_spectra[indx][:, :, 0]
    rad = np.reshape(rad_idx, wl_srf.shape, order="F")
    # Sum and normalize as p_srf is not normalized.
    rad_conv = np.sum(rad * p_srf, axis=1) / np.sum(p_srf, axis=1)

    return rad_conv


def load_optical_parameters():
    """Load optical parameters from saved arrays"""
    params_file = Path(__file__).parent / "model_parameters/optical_params.pkl"

    with open(params_file, "rb") as f:
        params = pickle.load(f)

    return params


def load_ET_parameters():
    """Load extratrerrestrial parameters from saved arrays"""
    params_file = Path(__file__).parent / "model_parameters/ET_irradiance.pkl"

    with open(params_file, "rb") as f:
        params = pickle.load(f)

    return params


def load_sensor_info(sensor):
    """Load RS sensor information from saved arrays"""
    sensor_path = Path(__file__).parent / f"sensor_information/{sensor}.pkl"
    with open(sensor_path, "rb") as f:
        sensor_info = pickle.load(f)
    return sensor_info


def set_soil_refl_trans_assumptions(soilopt, spectral):
    """Sets the model assumptions about soil and leaf reflectance and
    transmittance in the thermal range.

    These are that soil reflectance is the value for 2400 nm
    in the entire thermal range 

    Returns
    -------
    SoilOptics
    """
    rsoil = np.zeros((spectral.nwlP + spectral.nwlT, 1))
    rsoil[spectral.IwlP] = soilopt.refl
    rsoil[spectral.IwlT] = 1 * rsoil[spectral.nwlP - 1]
    soilopt.refl = rsoil
    return soilopt


def set_leaf_refl_trans_assumptions(refl, tran, leafbio, spectral):
    """Sets the model assumptions about soil and leaf reflectance and
    transmittance in the thermal range.

    These are that leaf relctance and
    transmittance are 0.01 in the thermal range (this is
    a model assumption that is set in the LeafBiology class in the prospect_5d 
    script)

    Returns
    -------
    LeafOptics
    """
    # Leaf reflectance array, spans 400 to 2400 nm in 1 nm increments
    # then 2500 to 15000 in 100 nm increments
    # then 16000 to 50000 in 1000 nm increments
    rho = np.zeros((spectral.nwlP + spectral.nwlT, 1))
    tau = np.zeros((spectral.nwlP + spectral.nwlT, 1))
    rho[spectral.IwlT] = leafbio.rho_thermal
    tau[spectral.IwlT] = leafbio.tau_thermal
    rho[spectral.IwlP] = refl
    tau[spectral.IwlP] = tran

    return rho, tau
