"""
SAILH Canopy model.

Ported from the original SPART matlab code.

SAILH model outlined in:
    Theory of radiative transfer models applied in optical remote sensing
        - W Verhoef 1998
"""
import numpy as np


def SAILH(soil, leafopt, canopy, sol_angle, obs_angle, rel_angle):
    """
    Run the SAILH model.

    Parameters
    ----------
    soil : BSM.SoilOptics
        Contains soil reflectance spectra for 400 nm to 2400 nm
    leafopt : PROSPECT_5D.LeafOptics
        Contains leaf reflectance and transmittance spectra, 400 nm to 2400 nm
    canopy : CanopyStructure
        Contains canopy information and SAIL model assumptions
    sol_angle : float
        Solar zenith angle, degrees
    obs_angle : float
        Observer zenith angle, degrees
    rel_angle : float
        Relative azimuth angle, degrees
    """
    # TODO: write Returns part of docstring

    deg2rad = np.pi / 180

    nl = canopy.nlayers
    litab = np.array([*range(5, 80, 10), *range(81, 91, 2)])[:, np.newaxis]
    LAI = canopy.LAI
    lidf = canopy.lidf
    x = np.arange(-1 / nl, -1 - (1/nl), -1 / nl)[:, np.newaxis]
    xl = np.arange(0, -1 - (1/nl), -1 / nl)[:, np.newaxis]
    dx = 1 / nl
    iLAI = LAI * dx

    rho = leafopt.refl
    tau = leafopt.tran
    rs = soil.refl
    tts = sol_angle
    tto = obs_angle

    # Set geometric quantities

    # ensures symmetry at 90 and 270 deg
    psi = abs(rel_angle - 360 * round(rel_angle / 360))
    psi_rad = psi * deg2rad
    sin_tts = np.sin(tts * deg2rad)
    cos_tts = np.cos(tts * deg2rad)
    tan_tts = np.tan(tts * deg2rad)

    sin_tto = np.sin(tto * deg2rad)
    cos_tto = np.cos(tto * deg2rad)
    tan_tto = np.tan(tto * deg2rad)

    sin_ttli = np.sin(litab * deg2rad)
    cos_ttli = np.cos(litab * deg2rad)

    dso = np.sqrt(tan_tts ** 2 + tan_tto **2 - 2 * tan_tts * tan_tto *
                  np.cos(psi * deg2rad))
    
    # geometric factors associated with extinction and scattering
    chi_s, chi_o, frho, ftau = _volscatt(sin_tts, cos_tts, sin_tto, cos_tto,
                                         psi_rad, sin_ttli, cos_ttli)



class CanopyStructure:
    """
    Class to hold canopy properties. Some are user specified, others are
    SAIL model assumptions.

    Parameters
    ----------
    LAI : float
        Leaf area index, 0 to 8
    LIDFa : float
        Leaf inclination distribution function parameter a, range -1 to 1
    LIDFb : float
        Leaf inclination distribution function parameter b, range -1 to 1
    q : float
        Canopy hotspot parameter: leaf width / canopy height, range 0 to 0.2

    Attributes
    ----------
    LAI : float
        Leaf area index, 0 to 8
    LIDFa : float
        Leaf inclination distribution function parameter a, range -1 to 1
    LIDFb : float
        Leaf inclination distribution function parameter b, range -1 to 1
    q : float
        Canopy hotspot parameter: leaf width / canopy height, range 0 to 0.2
    nlayers : int
        Number of layers in canopy, 60 (SAIL assumption)
    nlincl : int
        Number of different leaf inclination angles, 13 (SAIL assumption)
    nlazi : int
        Number of different leaf azimuth angles, 36 (SAIL assumption)
    lidf : np.array
        Leaf inclination distribution function, calculated from LIDF params
    """
    def __init__(self, LAI, LIDFa, LIDFb, q):
        self.LAI = LAI
        self.LIDFa = LIDFa
        self.LIDFb = LIDFb
        self.q = q
        self.nlayers = 60
        self.nlincl = 13
        self.nlazi = 36
        self.lidf = calculate_leafangles(LIDFa, LIDFb)


def calculate_leafangles(LIDFa, LIDFb):
    """Calculate the Leaf Inclination Distribution Function as outlined
    by Verhoef in paper cited at the top of this script.

    Parameters
    ----------
    LIDFa : float
        Leaf inclination distribution function parameter a, range -1 to 1
    LIDFb : float
        Leaf inclination distribution function parameter b, range -1 to 1

    Returns
    -------
    np.array
        Leaf inclination distribution function, calculated from LIDF
    """
    def dcum(a, b, theta):
        # Calculate cumulative distribution
        rd = np.pi / 180
        if LIDFa > 1:
            f = 1 - np.cos(theta * rd)
        else:
            eps = 1e-8
            delx = 1
            x = 2 * rd * theta
            theta2 = x
            while delx > eps:
                y = a * np.sin(x) + 0.5 * b * np.sin(2 * x)
                dx = 0.5 * (y - x + theta2)
                x = x + dx
                delx = abs(dx)
            f = (2 * y + theta2) / np.pi
        return f

    # F sized to 14 entries so diff for actual LIDF becomes 13 entries
    F = np.zeros((14, 1))  
    for i in range(1, 9):
        theta = i * 10
        F[i] = dcum(LIDFa, LIDFb, theta)
    for i in range(10, 13):
        theta = 80 + (i - 8) * 2
        F[i] = dcum(LIDFa, LIDFb, theta)
    F[13] = 1

    lidf = np.diff(F, axis=0)

    return lidf


def _volscatt(sin_tts, cos_tts, sin_tto, cos_tto, psi_rad, sin_ttli, cos_ttli):
    # Calculate geometric factors. See SAILH.m code.
    # See original matlab code. Adapted here to save recalculating trigs.
    Cs = cos_ttli * cos_tts
    Ss = sin_ttli * sin_tts

    Co = cos_ttli * cos_tto
    So = sin_ttli * sin_tto

    As = np.maximum(Ss, Cs)
    Ao = np.maximum(So, Co)

    bts = np.arccos(-Cs / As)
    bto = np.arccos(-Co / Ao)

    chi_o = 2 / pi * ((bto - np.pi / 2) * Co + np.sin(bto) * So)
    chi_o = 2 / pi * ((bts - np.pi / 2) * Co + np.sin(bts) * Ss)

    delta1 = np.abs(bts - bto)
    delta2 = np.pi - np.abs(bts + bto - np.pi)

    Tot = psi_rad + delta1 + delta2

    # TODO continue from here. See line 269 in SAILH.m file




if __name__ == '__main__':
    from SPART import load_optical_parameters
    from PROSPECT_5D import PROSPECT_5D, LeafBiology
    from BSM import BSM, SoilParameters
    leafbio = LeafBiology(40, 10, 0.02, 0.01, 0, 10, 1.5)
    leafopt = PROSPECT_5D(leafbio, load_optical_parameters())
    soilpar = SoilParameters(0.5, 0, 100, 15)
    soilopt = BSM(soilpar, load_optical_parameters())
    canopy = CanopyStructure(3, -0.35, -0.15, 0.05)
    SAILH(soilopt, leafopt, canopy, 40, 0, 0)

