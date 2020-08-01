"""
Brightness-Shape-Moisture soil model.

Ported from the original matlat SPART code.

Model as outlined in:
    The SPART model: A soil-plant-atmosphere radiative transfer model
    for satellite measurements in the solar spectrum - Yang et al.
"""
import numpy as np
from scipy.stats import poisson
from PROSPECT_5D import calculate_tav


def BSM(soilpar, soilspec):
    """
    Run the BSM soil model

    Parameters
    ----------
    soilpar : SoilParameters
        Object with attributes B, lat, lon, and SMp, SMC, film
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
    # Spectral parameters of the soil
    GSV = soilspec['GSV']  # Global Soil Vectors spectra
    kw = soilspec['Kw']  # Water absoprtion specturm
    nw = soilspec['nw']  # Water refraction index spectrum

    # Soil parameters - See object SoilParameters for variable details
    B = soilpar.B  
    lat = soilpar.lat
    lon = soilpar.lon
    SMp = soilpar.SMp
    SMC = soilpar.SMC  # NOTE: in the original code these are held in
    film = soilpar.film  # a struct named 'emp' and set to constants.

    f1 = B * np.sin(lat * np.pi / 180)
    f2 = B * np.cos(lat * np.pi / 180) * np.sin(lon * np.pi / 180)
    f3 = B * np.cos(lat * np.pi / 180) * np.cos(lon * np.pi / 180)

    rdry = f1 * GSV[:, [0]] + f2 * GSV[:, [1]] + f3 * GSV[:, [2]]

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
        rwet = (rdry * fmul[0]) + Rwet_k[:, 1:nk].dot(fmul[1:nk])[:, np.newaxis]

    return rwet


class SoilParameters:
    """
    Class to hold the soil characteristics for BSM.

    Parameters
    ----------
    B : float
        Soil brightness as defined in the paper.
    lat : float
        Soil spectral coordinate, latitiude, realistic range 20 - 40 deg
        for soil behavior (see paper)
    lon : float
        Soil spectral coordinate, longitude, realistic range 45 - 65 deg
        for soil behaviour (see paper)
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
    SMC : float, optional
        Soil moisture carrying capacity of the soil
    film : float, optional
        Single water film optical thickness, cm
    """
    def __init__(self, B, lat, lon, SMp, SMC=None, film=None):
        self.B = B
        self.lat = lat
        self.lon = lon
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


if __name__ == '__main__':
    # Test cases, compare to original matlab outputs
    from SPART import load_optical_parameters
    soilpar = SoilParameters(0.5, 0, 100, 15)
    rwet = BSM(soilpar, load_optical_parameters())
    print(rwet)

