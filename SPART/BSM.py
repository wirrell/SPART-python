"""
Brightness-Shape-Moisture soil model.

Ported from the original matlat SPART code.

Model as outlined in:
    The SPART model: A soil-plant-atmosphere radiative transfer model
    for satellite measurements in the solar spectrum - Yang et al.
"""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import poisson
from .PROSPECT_5D import calculate_tav


def BSM(soilpar, soilspec):
    """
    Run the BSM soil model

    Parameters
    ----------
    soilpar : SoilParameters
        Object with attributes [B, lat, lon] / dry soil spectra, and SMp, SMC,
        film
    soilspec : dict
        Contains keys ['GSV', 'kw', 'nw'] which key the Global Soil Vectors,
        water absorption constants for the spectrum and water refraction index
        for the spectrum. Loaded in in the main SPART script and passed to this
        function.

    Returns
    -------
    np.array
        Containing the relfectance spectrum for the soil, shape = (2001,)
    """
    # Soil parameters - See object SoilParameters for variable details
    SMp = soilpar.SMp
    SMC = soilpar.SMC  # NOTE: in the original code these are held in
    film = soilpar.film  # a struct named 'emp' and set to constants.

    # Spectral parameters of the soil
    if soilpar.rdry_set:
        rdry = soilpar.rdry
    else:
        GSV = soilspec['GSV']  # Global Soil Vectors spectra
        B = soilpar.B
        lat = soilpar.lat
        lon = soilpar.lon
        f1 = B * np.sin(lat * np.pi / 180)
        f2 = B * np.cos(lat * np.pi / 180) * np.sin(lon * np.pi / 180)
        f3 = B * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
        rdry = f1 * GSV[:, [0]] + f2 * GSV[:, [1]] + f3 * GSV[:, [2]]

    kw = soilspec['Kw']  # Water absoprtion specturm
    nw = soilspec['nw']  # Water refraction index spectrum

    rwet = soilwat(rdry, nw, kw, SMp, SMC, film)

    return rwet


def soilwat(rdry, nw, kw, SMp, SMC, deleff):
    """
    Model soil water effects on soil reflectance and return wet reflectance.

    From original matlab code:
        In this model it is assumed that the water film area is built up
        according to a Poisson process.

    See the description in the original model paper in the top of script
    docstring.

    Parameters
    ----------
    rdry : np.array
        Dry soil reflectance
    nw : np.array
        Refraction index of water
    kw : np.array
        Absorption coefficient of water
    SMp : float
        Soil moisture volume [%]
    SMC : float
        Soil moisture carrying capacity
    deleff : float
        Effective optical thickness of single water film, cm

    Returns
    -------
    np.array
        Wet soil reflectance spectra across 400 nm to 2400 nm

    NOTE
    ----
    The original matlab script accepts SMp row vectors for different SM
    percentages. This is not implemented here but may need to be in future
    if there is a significant speed bonus to doing so.
    """
    k = [0, 1, 2, 3, 4, 5, 6]
    nk = len(k)
    mu = (SMp - 5) / SMC
    if mu <= 0:  # below 5 % SM -> model assumes no effect
        rwet = rdry
    else:
        # From original matlab: Lekner & Dorf (1988)
        #
        # Uses t_av calculation from PROSPECT-5D model. If you want BSM
        # script to operate independently you will need to paste that
        # function in here.
        rbac = 1 - (1 - rdry) * (rdry * calculate_tav(90, 2 / nw) /
                                 calculate_tav(90, 2) + 1 - rdry)

        # total reflectance at bottom of water film surface
        p = 1 - calculate_tav(90, nw) / nw ** 2

        # reflectance of water film top surface, use 40 degrees incidence angle
        # like in PROSPECT (note from original matlab script)
        Rw = 1 - calculate_tav(40, nw)

        fmul = poisson.pmf(k, mu)
        tw = np.exp(-2 * kw * deleff * k)
        Rwet_k = Rw + (1 - Rw) * (1 - p) * tw * rbac / (1 - p * tw * rbac)
        rwet = (rdry * fmul[0]) + Rwet_k[:, 1:nk].dot(fmul[1:nk])[:,
                                                                  np.newaxis]

    soilopt = SoilOptics(rwet, rdry)

    return soilopt


class SoilOptics:
    """
    Class to hold soil optical reflectance spectra.

    Parameters
    ----------
    refl : np.array
        Soil reflectance spectra (with SM taken into account)
    refl_dry : np.array
        Dry soil reflectance spectra

    Attributes
    ----------
    refl : np.array
        Soil reflectance spectra (with SM taken into account)
    refl_dry : np.array
        Dry soil reflectance spectra
    """
    def __init__(self, refl, refl_dry):
        self.refl = refl
        self.refl_dry = refl_dry


class SoilParametersFromFile:
    """
    Class to load and hold soil reflectance spectrum from the JPL soil
    reflectance data available at https://speclib.jpl.nasa.gov/

    Parameters
    ----------
    file_path : str
        path to JPL soil reflectance spectra file
    SMp : float
        Soil moisture percentage [%]
    SMC : float, optional
        Soil moisture carrying capacity of the soil
    film : float, optional
        Single water film optical thickness, cm

    Attributes
    ----------
    rdry_set : bool
        True. Declares that the object contains a dry soil reflectance spectra
    rdry : np.array
        Array containing soil reflectance spectrum extracted from file and
        interpolated to 1 nm intervals between 400 nm and 2400 nm
    """
    def __init__(self, file_path, SMp, SMC=None, film=None):
        self.rdry = self._load_jpl_soil_refl(file_path)
        self.SMp = SMp
        if isinstance(SMC, type(None)):
            print("BSM soil model: SMC not supplied, set to default of 25 %")
            self.SMC = 25
        else:
            self.SMC = SMC
        if isinstance(film, type(None)):
            print("BSM soil model: water film optical thickness not supplied,")
            print("\t set to default of 0.0150 cm")
            self.film = 0.0150
        else:
            self.film = film
        self.rdry_set = True

    def _load_jpl_soil_refl(self, file_path):
        """Load and format the JPL supplied soil reflectance file."""
        # Load in table
        soil_refl = pd.read_csv(file_path, sep='\t', skiprows=21,
                                index_col=0, header=None)
        # turn micrometers index to nanometers
        soil_refl.index = soil_refl.index * 1000
        # convert percentage reflectance to fraction
        if (soil_refl.loc[:, 1] > 1).any():
            soil_refl = soil_refl / 100

        # Get only 400 nm to 2400 nm and order so 400 nm comes first
        soil_refl = soil_refl[2401:400][::-1]

        # Interpolate so reflectances on whole nm values
        wls = np.arange(400, 2401, 1)
        for wl in wls:
            if not wl in soil_refl.index:
                soil_refl.loc[wl, 1] = np.nan
        soil_refl = soil_refl.sort_index()

        soil_refl = soil_refl.interpolate('linear')
        soil_refl = soil_refl.loc[wls].to_numpy()

        return soil_refl


class SoilParameters:
    """
    Class to hold the soil characteristics for BSM.

    Parameters
    ----------
    B : float
        Soil brightness as defined in the paper.
    lat : float
        Soil spectral coordinate, latitiude, realistic range 80 - 120 deg
        for soil behavior (see paper, phi)
    lon : float
        Soil spectral coordinate, longitude, realistic range -30 - 30 deg
        for soil behaviour (see paper, lambda)
    SMp : float
        Soil moisture percentage [%]
    SMC : float, optional
        Soil moisture carrying capacity of the soil
    film : float, optional
        Single water film optical thickness, cm

    Attributes
    ----------
    B : float
        Soil brightness as defined in the paper.
    lat : float
        Soil spectral coordinate, latitiude
    lon : float
        Soil spectral coordinate, longitude
    SMp : float
        Soil moisture percentage [%]
    SMC : float
        Soil moisture carrying capacity of the soil
    film : float
        Single water film optical thickness, cm
    rdry_set : bool
        False. Declares that the object doesnt' contain a dry soil reflectance
        spectra
    """
    def __init__(self, B, lat, lon, SMp, SMC=None, film=None):
        self.B = B
        self.lat = lat
        self.lon = lon
        self.SMp = SMp
        if isinstance(SMC, type(None)):
            warnings.warn("BSM soil model: SMC not supplied,"
                          " set to default of 25 %")
            self.SMC = 25
        else:
            self.SMC = SMC
        if isinstance(film, type(None)):
            warnings.warn("BSM soil model: water film optical thickness"
                          " not supplied, set to default of 0.0150 cm")
            self.film = 0.0150
        else:
            self.film = film
        self.rdry_set = False


if __name__ == '__main__':
    # Test cases, compare to original matlab outputs
    from SPART import load_optical_parameters
    soilpar = SoilParameters(0.5, 0, 100, 15)
    soilopt = BSM(soilpar, load_optical_parameters())
    print(soilopt.refl)
