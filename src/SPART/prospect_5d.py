"""
run_spart-python

PROSPECT 5D or PROSPECT-PRO model.

Feret et al. - PROSPECT-D: Towards modeling leaf optical properties
    through a complete lifecycle

PROSPECT-PRO model.

Féret et al. (2021) - PROSPECT-PRO for estimating content of nitrogen-containing leaf proteins
and other carbon-based constituents 
"""
import numpy as np
import numba
import scipy.integrate as integrate
from dataclasses import dataclass


@dataclass
class LeafBiology:
    """
    Class to hold leaf biology variables.

    Parameters
    ----------
    Cab : float
        Chlorophyll concentration, micro g / cm ^ 2
    Cdm : float
        Leaf mass per unit area, g / cm ^ 2
    Cw : float
        Equivalent water thickness, cm
    Cs : float
        Brown pigments (from run_spart paper, unitless)
    Cca : float
        Carotenoid concentration, micro g / cm ^ 2
    Cant : float
        Anthocyanin content, micro g / cm ^ 2
    N : float
        Leaf structure parameter. Unitless.
    PROT : float
        protein content, g / cm ^ 2
    CBC : float
        non-protein carbon-based constituent content, g cm ^ 2

    Attributes
    ----------
    Cab : float
        Chlorophyll concentration, micro g / cm ^ 2
    Cdm : float
        Leaf mass per unit area, g / cm ^ 2
    Cw : float
        Equivalent water thickness, cm
    Cs : float
        Brown pigments (from run_spart paper, unitless)
    Cca : float
        Carotenoid concentration, micro g / cm ^ 2
    Cant : float
        Anthocyanin content, micro g / cm ^ 2
    N : float
        Leaf structure parameter. Unitless.
    PROT : float
        leaf protein content, g / cm ^ 2. Default: 0.0
        Range of values: 0 - 0.003 g / cm ^ 2 (Féret et al., 2021)
    CBC : float
        non-protein carbon-based constituent content, g cm ^ 2. Default: 0.0
        Range of values: 0 - 0.01 g / cm ^ 2 (Féret et al., 2021)
    rho_thermal : float
        Reflectance in the thermal range. run_spart assumption: 0.01
    tau_thermal : float
        Transmittance in the thermal range. run_spart assumption: 0.01
    """

    Cab: float
    Cdm: float
    Cw: float
    Cs: float
    Cca: float
    Cant: float
    N: float
    PROT: float = 0.0
    CBC: float = 0.0
    rho_thermal: float = 0.01
    tau_thermal: float = 0.01


@dataclass
class LeafOptics:
    """
    Class to hold leaf optics information.

    Parameters
    ----------
    refl : np.array
        Spectral reflectance of the leaf, 400 to 2400 nm
    tran : np.array
        Spectral transmittance of the leaf, 400 to 2400 nm
    kChlrel : np.array
        Relative portion of chlorophyll contribution to reflecntace
        / transmittance in the spectral range, 400 to 2400 nm

    Attributes
    ----------
    refl : np.array
        Spectral reflectance of the leaf, 400 to 2400 nm
    tran : np.array
        Spectral transmittance of the leaf, 400 to 2400 nm
    kChlrel : np.array
        Relative portion of chlorophyll contribution to reflecntace
        / transmittance in the spectral range, 400 to 2400 nm
    """

    refl: np.ndarray
    tran: np.ndarray
    kChlrel: np.ndarray


# mangling __PROSPECT_5D_ - runs as an internal function
# see we can pull out all non-jittable functions and then just have one jit call to this larger-scoped function
def PROSPECT_5D(leafbio, optical_params):
    """
    PROSPECT_5D model.

    Parameters
    ----------
    leafbio : LeafBiology
        Object holding user specified leaf biology model parameters.
    optical_params : dict
        Optical parameter constants. Loaded externally and passed in.

    Returns
    -------
    LeafOptics
        Contains attributes relf, tran, kChlrel for reflectance, transmittance
        and contribution of chlorophyll over the 400 nm to 2400 nm spectrum
    """
    # Leaf parameters
    Cab = leafbio.Cab
    Cca = leafbio.Cca
    Cw = leafbio.Cw
    Cdm = leafbio.Cdm
    Cs = leafbio.Cs
    Cant = leafbio.Cant
    N = leafbio.N
    PROT = leafbio.PROT  # PROSPECT-PRO
    CBC = leafbio.CBC  # PROSPECT-PRO

    # check if PROT and/or CBC are non-zero. If true, PROSPECT-PRO is run.
    # Before, check if the parameterization is physically plausible
    # (Cdm = PROT + CBC)
    if PROT > 0.0 or CBC > 0.0:
        if Cdm > 0:
            print(
                "WARNING: When setting PROT and/or CBC > 0. we\n"
                "assume that PROSPECT-PRO was called. Cdm will be\n"
                "therefore set to zero (Cdm = PROT + CBC)"
            )
            Cdm = 0.0

    # Model constants
    nr = optical_params["nr"]
    Kdm = optical_params["Kdm"]
    Kab = optical_params["Kab"]
    Kca = optical_params["Kca"]
    Kw = optical_params["Kw"]
    Ks = optical_params["Ks"]
    Kant = optical_params["Kant"]
    # add PROSPECT-PRO optical parameters (Féret et al., 2021)
    Kcbc = optical_params["cbc"]
    Kprot = optical_params["prot"]

    Kall = make_Kall(
        Cab,
        Cca,
        Cdm,
        Cw,
        Cs,
        Cant,
        CBC,
        PROT,
        Kab,
        Kca,
        Kdm,
        Kw,
        Ks,
        Kant,
        Kcbc,
        Kprot,
        N
    )

    t1, j = make_j_t1(Kall)

    t2 = Kall ** 2 * np.vectorize(expint)(Kall)[0]

    tau = make_tau(t1, t2, j)

    kChlrel = make_KChlrel(t1, Cab, Kab, j, N, Kall)

    t_alph = calculate_tav(40, nr)

    t12 = calculate_tav(90, nr)


    # Call PROSPECT computation
    refl, tran, kChlrel = _PROSPECT_5D(
        t1, j, t2, tau, Kall, kChlrel, t_alph, t12, nr, N
    )

    # We flatten the arrays here so they go from (2001, 1), to (2001,)
    leafopt = LeafOptics(refl, tran, kChlrel)

    return leafopt


@numba.njit
def _PROSPECT_5D(t1, j, t2, tau, Kall, kChlrel, t_alph, t12, nr, N):

    r_alph = 1 - t_alph
    r12 = 1 - t12
    t21 = t12 / (nr ** 2)
    r21 = 1 - t21

    # top surface side
    denom = 1 - r21 * r21 * tau ** 2
    Ta = t_alph * tau * t21 / denom
    Ra = r_alph + r21 * tau * Ta

    # bottom surface side
    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t

    # Stokes equations to compute properties of next N-1 layers (N real)

    # Normal case
    D = np.sqrt((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t))
    rq = r ** 2
    tq = t ** 2
    a = (1 + rq - tq + D) / (2 * r)
    b = (1 - rq + tq + D) / (2 * t)

    bNm1 = b ** (N - 1)
    bN2 = bNm1 ** 2
    a2 = a ** 2
    denom = a2 * bN2 - 1
    Rsub = a * (bN2 - 1) / denom
    Tsub = bNm1 * (a2 - 1) / denom

    # Case of zero absorption
    j = np.where(r + t >= 1)[0]
    Tsub[j] = t[j] / (t[j] + (1 - t[j]) * (N - 1))
    Rsub[j] = 1 - Tsub[j]

    # Reflectance and transmittance of the leaf:
    #   combine top llayer with next N-1 layers
    denom = 1 - Rsub * r
    tran = Ta * Tsub / denom
    refl = Ra + Ta * Rsub * t / denom

    return refl, tran, kChlrel


# expint can't be accelerated via numba because of scipy integrate
def expint(x):
    # NOTE: differences in final output come from this integral
    # which evaluates slightly different (10 decimal places) than matlab
    # Exponential integral from expint command in matlab
    def intergrand(t):
        return np.exp(-t) / t

    return integrate.quad(intergrand, x, np.inf)


@numba.njit
def make_Kall(
    Cab, Cca, Cdm, Cw, Cs, Cant, CBC, PROT, Kab, Kca, Kdm, Kw, Ks, Kant, Kcbc, Kprot, N
):
    # Compact leaf layer
    Kall = (
        Cab * Kab
        + Cca * Kca
        + Cdm * Kdm
        + Cw * Kw
        + Cs * Ks
        + Cant * Kant
        + CBC * Kcbc
        + PROT * Kprot
    ) / N
    return Kall


# Non-conservative scattering (normal case)
@numba.njit
def make_j_t1(Kall):
    j = np.where(Kall > 0)[0]
    t1 = (1 - Kall) * np.exp(-Kall)
    return t1, j


@numba.njit
def make_tau(t1, t2, j):
    tau = np.ones((len(t1), 1))
    tau[j] = t1[j] + t2[j]
    return tau


@numba.njit
def make_KChlrel(t1, Cab, Kab, j, N, Kall):
    kChlrel = np.zeros((len(t1), 1))
    kChlrel[j] = Cab * Kab[j] / (Kall[j] * N)
    return kChlrel


@numba.njit
def calculate_tav(alpha, nr):
    """
    Calculate average transmissitivity of a dieletrie plane surface.

    Parameters
    ----------
    alpha : float
        Maximum incidence angle defining the solid angle.
    nr : float
        Refractive index

    Returns
    -------
    float
        Transmissivity of a dielectric plane surface averages over all
        directions of incidence and all polarizations.

    NOTE
    ----
    Lifted directly from original run_spart matlab calculations.
    Papers cited in original PROSPECT model:
        Willstatter-Stoll Theory of Leaf Reflectance Evaluated
        by Ray Tracinga - Allen et al.
        Transmission of isotropic radiation across an interface
        between two dielectrics - Stern
    """

    rd = np.pi / 180
    n2 = nr ** 2
    n_p = n2 + 1
    nm = n2 - 1
    a = (nr + 1) * (nr + 1) / 2
    k = -(n2 - 1) * (n2 - 1) / 4
    sa = np.full((2001, 1), np.sin(alpha * rd))

    b1 = np.zeros((2001, 1))
    if alpha != 90:
        b1 = np.sqrt((sa ** 2 - n_p / 2) * (sa ** 2 - n_p / 2) + k)

    b2 = sa ** 2 - n_p / 2
    b = b1 - b2
    b3 = b ** 3
    a3 = a ** 3
    ts = (k ** 2 / (6 * b3) + k / b - b / 2) - (k ** 2 / (6 * a3) + k / a - a / 2)

    tp1 = -2 * n2 * (b - a) / (n_p ** 2)
    tp2 = -2 * n2 * n_p * np.log(b / a) / (nm ** 2)
    tp3 = n2 * (1 / b - 1 / a) / 2
    tp4 = (
        16
        * n2 ** 2
        * (n2 ** 2 + 1)
        * np.log((2 * n_p * b - nm ** 2) / (2 * n_p * a - nm ** 2))
        / (n_p ** 3 * nm ** 2)
    )
    tp5 = (
        16
        * n2 ** 3
        * (1 / (2 * n_p * b - nm ** 2) - 1 / (2 * n_p * a - nm ** 2))
        / n_p ** 3
    )
    tp = tp1 + tp2 + tp3 + tp4 + tp5
    tav = (ts + tp) / (2 * sa ** 2)

    return tav
