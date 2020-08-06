"""
SAILH Canopy model.

Ported from the original SPART matlab code.

SAILH model outlined in:
    Theory of radiative transfer models applied in optical remote sensing
        - W Verhoef 1998
"""
import numpy as np
import scipy.integrate as integrate


def SAILH(soil, leafopt, canopy, angles):
    """
    Run the SAILH model.

    Parameters
    ----------
    soil : BSM.SoilOptics
        Contains soil reflectance spectra for 400 nm to 2400 nm
    leafopt : PROSPECT_5D.LeafOptics
        Contains leaf reflectance and transmittance spectra, 400 nm to 2400 nm,
        2500 to 15000 nm, and 16000 to 50000 nm.
    canopy : CanopyStructure
        Contains canopy information and SAIL model assumptions
    angles : Angles
        Holds solar zenith, observer zenith, and relative azimuth angles

    Returns
    -------
    CanopyReflectances
        Contains the four canopy reflectances arrays as attributes rso, rdo,
        rsd, rdd.
    """

    if len(leafopt.refl) != 2162:
        raise RuntimeError('Parameter leafopt.refl must be of len 2162'
                           ' i.e. include thermal specturm. \n This error'
                           ' usually occurs if you are feeding the PROSPECT_5D'
                           ' output directly into the SAILH model with adding'
                           '\n the neccessary thermal wavelengths.')

    deg2rad = np.pi / 180

    nl = canopy.nlayers
    litab = np.array([*range(5, 80, 10), *range(81, 91, 2)])[:, np.newaxis]
    LAI = canopy.LAI
    lidf = canopy.lidf
    xl = np.arange(0, -1 - (1/nl), -1 / nl)[:, np.newaxis]
    dx = 1 / nl
    iLAI = LAI * dx

    rho = leafopt.refl
    tau = leafopt.tran
    rs = soil.refl
    tts = angles.sol_angle
    tto = angles.obs_angle

    # Set geometric quantities

    # ensures symmetry at 90 and 270 deg
    psi = abs(angles.rel_angle - 360 * round(angles.rel_angle / 360))
    psi_rad = psi * deg2rad
    sin_tts = np.sin(tts * deg2rad)
    cos_tts = np.cos(tts * deg2rad)
    tan_tts = np.tan(tts * deg2rad)

    sin_tto = np.sin(tto * deg2rad)
    cos_tto = np.cos(tto * deg2rad)
    tan_tto = np.tan(tto * deg2rad)

    sin_ttli = np.sin(litab * deg2rad)
    cos_ttli = np.cos(litab * deg2rad)

    dso = np.sqrt(tan_tts ** 2 + tan_tto ** 2 - 2 * tan_tts * tan_tto *
                  np.cos(psi_rad))

    # geometric factors associated with extinction and scattering
    chi_s, chi_o, frho, ftau = _volscatt(sin_tts, cos_tts, sin_tto, cos_tto,
                                         psi_rad, sin_ttli, cos_ttli)
    # extinction coefficient in direction of sun per
    ksli = chi_s / cos_tts  # leaf angle
    koli = chi_o / cos_tto  # observer angle
    # area scattering coefficient fractions
    sobli = frho * np.pi / (cos_tts * cos_tto)
    sofli = ftau * np.pi / (cos_tts * cos_tto)
    bfli = cos_ttli ** 2

    # integration over angles using dot product
    k = ksli.T.dot(lidf)
    K = koli.T.dot(lidf)
    bf = bfli.T.dot(lidf)
    sob = sobli.T.dot(lidf)
    sof = sofli.T.dot(lidf)

    # geometric factors for use with rho and tau
    sdb = 0.5 * (k + bf)  # specular to diffuse backward scattering
    sdf = 0.5 * (k - bf)  # specular to diffuse forward scattering
    ddb = 0.5 * (1 + bf)  # diffuse to diffuse backward scattering
    ddf = 0.5 * (1 - bf)  # diffuse to diffuse forward scattering
    dob = 0.5 * (K + bf)  # diffuse to directional backward scattering
    dof = 0.5 * (K - bf)  # diffuse to directional forward scattering

    # Probabilites
    Ps = np.exp(k * xl * LAI)  # of viewing a leaf in solar direction
    Po = np.exp(K * xl * LAI)  # of viewing a leaf in observation direction

    Ps[0:nl] = Ps[0:nl] * (1 - np.exp(-k * LAI * dx)) / (k * LAI * dx)
    Po[0:nl] = Po[0:nl] * (1 - np.exp(-k * LAI * dx)) / (k * LAI * dx)

    q = canopy.q
    Pso = np.zeros(Po.shape)

    def Psofunction(x, K, k, LAI, q, dso):
        # From APPENDIX IV of original matlab code
        if dso != 0:
            alpha = (dso / q) * 2 / (k + K)
            pso = np.exp((K + k) * LAI * x + np.sqrt(K * k) * LAI / alpha *
                         (1 - np.exp(x * alpha)))
        else:
            pso = np.exp((K + k) * LAI * x - np.sqrt(K * k) * LAI * x)

        return pso

    for j in range(len(xl)):
        Pso[j, :] = integrate.quad(Psofunction, xl[j] - dx, xl[j],
                                   args=(K, k, LAI, q, dso))[0] / dx

    # NOTE: there are two lines in the original script here that deal with
    # rounding errors. I have excluded them. If this becomes a problem see
    # lines 115 / 116 in SAILH.m

    # scattering coefficients for
    sigb = ddb * rho + ddf * tau  # diffuse backscatter incidence
    sigf = ddf * rho + ddb * tau  # forward incidence
    sb = sdb * rho + sdf * tau  # specular backscatter incidence
    sf = sdf * rho + sdb * tau  # specular forward incidence
    vb = dob * rho + dof * tau  # directional backscatter diffuse
    vf = dof * rho + dob * tau  # directional forward scatter diffuse
    w = sob * rho + sof * tau  # bidirectional scattering
    a = 1 - sigf  # attenuation
    m = np.sqrt(a ** 2 - sigb ** 2)
    rinf = (a - m) / sigb
    rinf2 = rinf * rinf

    def calcJ1(x, m, k, LAI):
        # For getting numerically stable solutions
        J1 = np.zeros((len(m), 1))
        sing = np.abs((m - k) * LAI) < 1e-6

        CS = np.where(sing)
        CN = np.where(~sing)

        J1[CN] = (np.exp(m[CN] * LAI * x) - np.exp(k * LAI * x)) / (k - m[CN])
        J1[CS] = -0.5 * (np.exp(m[CS] * LAI * x) + np.exp(k * LAI * x)) * LAI \
            * x * (1 - 1/12 * (k - m[CS]) ** 2 * LAI ** 2)
        return J1

    def calcJ2(x, m, k, LAI):
        # For getting numerically stable solutions
        J2 = (np.exp(k * LAI * x) - np.exp(-k * LAI) * np.exp(-m * LAI *
                                                              (1 + x))
              ) / (k + m)
        return J2

    # direct solar radiation
    J1k = calcJ1(-1, m, k, LAI)
    J2k = calcJ2(0, m, k, LAI)
    J1K = calcJ1(-1, m, K, LAI)
    J2K = calcJ2(0, m, K, LAI)

    e1 = np.exp(-m * LAI)
    e2 = e1 ** 2
    re = rinf * e1

    denom = 1 - rinf2 ** 2

    s1 = sf + rinf * sb
    s2 = sf * rinf + sb
    v1 = vf + rinf * vb
    v2 = vf * rinf + vb
    Pss = s1 * J1k
    Qss = s2 * J2k
    Poo = v1 * J1K
    Qoo = v2 * J2K

    tau_ss = np.exp(-k * LAI)
    tau_oo = np.exp(-K * LAI)

    Z = (1 - tau_ss * tau_oo) / (K + k)

    tau_dd = (1 - rinf2) * e1 / denom
    rho_dd = rinf * (1 - e2) / denom
    tau_sd = (Pss - re * Qss) / denom
    tau_do = (Poo - re * Qoo) / denom
    rho_sd = (Qss - re * Pss) / denom
    rho_do = (Qoo - re * Poo) / denom

    T1 = v2 * s1 * (Z - J1k * tau_oo) / (K + m) + v1 * s2 * (Z - J1K * tau_ss)\
        / (k + m)
    T2 = -(Qoo * rho_sd + Poo * tau_sd) * rinf
    rho_sod = (T1 + T2) / (1 - rinf2)

    rho_sos = w * np.sum(Pso[0: nl]) * iLAI
    rho_so = rho_sod + rho_sos

    Pso2w = Pso[nl]

    # Sail analytical reflectances
    denom = 1 - rs * rho_dd

    rso = rho_so + rs * Pso2w + (
        (tau_sd + tau_ss * rs * rho_dd) * tau_oo + (tau_sd + tau_ss) * tau_do)\
        * rs / denom
    rdo = rho_do + (tau_oo + tau_do) * rs * tau_dd / denom
    rsd = rho_sd + (tau_ss + tau_sd) * rs * tau_dd / denom
    rdd = rho_dd + tau_dd * rs * tau_dd / denom

    rad = CanopyReflectances(rso, rdo, rsd, rdd)

    return rad


class CanopyReflectances:
    """CanopyReflectances"""
    """Class to hold canopy reflectances computed by the SAILH model.

    Parameters
    ----------
    rso : np.array
        Bidirectional reflectance of the canopy
    rdo : np.array
        Directional reflectance for diffuse incidence of the canopy
    rsd : np.array
        Diffuse reflectance for specular incidence of the canopy
    rdd : np.array
        Diffuse reflectance for diffuse incidence of the canopy

    Attributes
    ----------
    rso : np.array
        Bidirectional reflectance of the canopy
    rdo : np.array
        Directional reflectance for diffuse incidence of the canopy
    rsd : np.array
        Diffuse reflectance for specular incidence of the canopy
    rdd : np.array
        Diffuse reflectance for diffuse incidence of the canopy
    """
    def __init__(self, rso, rdo, rsd, rdd):
        self.rso = rso
        self.rdo = rdo
        self.rsd = rsd
        self.rdd = rdd


class Angles:
    """Class to hold solar zenith, observation zenith, and relative azimuth
    angles.

    Parameters
    ----------
    sol_angle : float
        Solar zenith angle, degrees
    obs_angle : float
        Observer zenith angle, degrees
    rel_angle : float
        Relative azimuth angle, degrees

    Attributes
    ----------
    sol_angle : float
        Solar zenith angle, degrees
    obs_angle : float
        Observer zenith angle, degrees
    rel_angle : float
        Relative azimuth angle, degrees
    """
    def __init__(self, sol_angle, obs_angle, rel_angle):
        self.sol_angle = sol_angle
        self.obs_angle = obs_angle
        self.rel_angle = rel_angle


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
    for i in range(9, 13):
        theta = 80 + (i - 8) * 2
        F[i] = dcum(LIDFa, LIDFb, theta)
    F[13] = 1

    lidf = np.diff(F, axis=0)

    return lidf


def _volscatt(sin_tts, cos_tts, sin_tto, cos_tto, psi_rad, sin_ttli, cos_ttli):
    # Calculate geometric factors. See SAILH.m code.
    # See original matlab code. Adapted here to save recalculating trigs.
    nli = len(cos_ttli)

    psi_rad = psi_rad * np.ones((nli, 1))
    cos_psi = np.cos(psi_rad)

    Cs = cos_ttli * cos_tts
    Ss = sin_ttli * sin_tts

    Co = cos_ttli * cos_tto
    So = sin_ttli * sin_tto

    As = np.maximum(Ss, Cs)
    Ao = np.maximum(So, Co)

    bts = np.arccos(-Cs / As)
    bto = np.arccos(-Co / Ao)

    chi_o = 2 / np.pi * ((bto - np.pi / 2) * Co + np.sin(bto) * So)
    chi_s = 2 / np.pi * ((bts - np.pi / 2) * Cs + np.sin(bts) * Ss)

    delta1 = np.abs(bts - bto)
    delta2 = np.pi - np.abs(bts + bto - np.pi)

    Tot = psi_rad + delta1 + delta2

    bt1 = np.minimum(psi_rad, delta1)
    bt3 = np.maximum(psi_rad, delta2)
    bt2 = Tot - bt1 - bt3

    T1 = 2 * Cs * Co + Ss * So * cos_psi
    T2 = np.sin(bt2) * (2 * As * Ao + Ss * So * np.cos(bt1) * np.cos(bt3))

    Jmin = bt2 * T1 - T2
    Jplus = (np.pi - bt2) * T1 + T2

    frho = Jplus / (2 * np.pi ** 2)
    ftau = -Jmin / (2 * np.pi ** 2)

    zeros = np.zeros((nli, 1))
    frho = np.maximum(zeros, frho)
    ftau = np.maximum(zeros, ftau)

    return chi_s, chi_o, frho, ftau
