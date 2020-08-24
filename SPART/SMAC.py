"""
SMAC model.

Library for atmospheric correction using SMAC method (Rahman and Dedieu, 1994)
Original translated and improved by:
Peiqi Yang, ITC, University of Twente

Adapted to Python by:
George Worrall, CRS, University of Florida
"""
import numpy as np


def SMAC(angles, atm, coefs):
    """
    Run the SMAC atmosphere model.

    Parameters
    ----------
    angles : SAILH.Angles
        Contains solar zenith, observational zentiy, and relative difference of
        azimuth angles in degrees.
    atm : AtmosphericProperties
        Contains the atmospheric properties for the model.
    coefs : dict
        Contains all the transmittance coefficients for different atmospheric
        gases at the wavelengths of the specified sensor.

    Returns
    -------
    AtmosphericOptics
        Contains atmospheric reflectance and transmittance arrays
    """
    # Extract variables for more concise use below.
    # Model coefficients for days.
    tts = angles.sol_angle
    tto = angles.obs_angle
    psi = angles.rel_angle
    Pa = atm.Pa
    taup550 = atm.aot550
    uo3 = atm.uo3
    uh2o = atm.uh2o

    ah2o = coefs['ah2o']
    nh2o = coefs['nh2o']
    ao3 = coefs['ao3']
    no3 = coefs['no3']
    ao2 = coefs['ao2']
    no2 = coefs['no2']
    po2 = coefs['po2']
    aco2 = coefs['aco2']
    nco2 = coefs['nco2']
    pco2 = coefs['pco2']
    ach4 = coefs['ach4']
    nch4 = coefs['nch4']
    pch4 = coefs['pch4']
    ano2 = coefs['ano2']
    nno2 = coefs['nno2']
    pno2 = coefs['pno2']
    aco = coefs['aco']
    nco = coefs['nco']
    pco = coefs['pco']
    a0s = coefs['a0s']
    a1s = coefs['a1s']
    a2s = coefs['a2s']
    a3s = coefs['a3s']
    a0T = coefs['a0T']
    a1T = coefs['a1T']
    a2T = coefs['a2T']
    a3T = coefs['a3T']
    taur = coefs['taur']
    # sr = coefs['sr']
    a0taup = coefs['a0taup']
    a1taup = coefs['a1taup']
    wo = coefs['wo']
    gc = coefs['gc']
    a0P = coefs['a0P']
    a1P = coefs['a1P']
    a2P = coefs['a2P']
    a3P = coefs['a3P']
    a4P = coefs['a4P']
    Rest1 = coefs['Rest1']
    Rest2 = coefs['Rest2']
    Rest3 = coefs['Rest3']
    Rest4 = coefs['Rest4']
    Resr1 = coefs['Resr1']
    Resr2 = coefs['Resr2']
    Resr3 = coefs['Resr3']
    Resa1 = coefs['Resa1']
    Resa2 = coefs['Resa2']
    Resa3 = coefs['Resa3']
    Resa4 = coefs['Resa4']

    cdr = np.pi / 180
    crd = 180 / np.pi

    # Calculate TOA reflectance
    us = np.cos(tts * cdr)
    uv = np.cos(tto * cdr)
    Peq = Pa / 1013.25

    m = 1 / us + 1 / uv  # air mass
    taup = a0taup + a1taup * taup550  # aerosol optical depth in spectral band

    uo2 = Peq ** po2  # gaseous transmissions (down and up paths)
    uco2 = Peq ** pco2
    uch4 = Peq ** pch4
    uno2 = Peq ** pno2
    uco = Peq ** pco

    to3 = np.exp(ao3 * (uo3 * m) ** no3)
    th2o = np.exp(ah2o * (uh2o * m) ** nh2o)
    to2 = np.exp(ao2 * (uo2 * m) ** no2)
    tco2 = np.exp(aco2 * (uco2 * m) ** nco2)
    tch4 = np.exp(ach4 * (uch4 * m) ** nch4)
    tno2 = np.exp(ano2 * (uno2 * m) ** nno2)
    tco = np.exp(aco * (uco * m) ** nco)

    tg = th2o * to3 * to2 * tco2 * tch4 * tco * tno2  # Eq 6

    # spherical albedo of the atmosphere
    s = a0s * Peq + a3s + a1s * taup550 + a2s * taup550 ** 2  # mod of Eq 8

    # total scattering transmission
    ttetas = a0T + a1T * taup550 / us + (a2T * Peq + a3T) / (1 + us)
    ttetav = a0T + a1T * taup550 / uv + (a2T * Peq + a3T) / (1 + uv)

    # scattering angle cosine
    cksi = - ((us * uv) + (np.sqrt(1 - us * us) * np.sqrt(1 - uv * uv) *
                           np.cos(psi * crd)))

    # hard limit on cksi -> from original matlab
    if cksi < -1:
        cksi = -1

    # scattering angle in degrees
    ksiD = crd * np.arccos(cksi)

    # rayleigh atmospheric reflectance
    ray_phase = 0.7190443 * (1 + (cksi * cksi)) + 0.0412742  # Eq 13
    ray_ref = (taur * ray_phase) / (4 * us * uv)  # Eq 11
    ray_ref = ray_ref * Pa / 1013.25  # correction for pressure variation
    taurz = taur * Peq  # Eq 12

    aer_phase = a0P + a1P * ksiD + a2P * ksiD * ksiD + a3P * ksiD ** 3 + \
        a4P * ksiD ** 4  # extension of Eq 17
    ak2 = (1 - wo) * (3 - wo * 3 * gc)
    ak = np.sqrt(ak2)

    # X Y Z Appendix
    e = -3 * us * us * wo / (4 * (1 - ak2 * us * us))
    f = -(1 - wo) * 3 * gc * us * us * wo / (4 * (1 - ak2 * us * us))
    dp = e / (3 * us) + us * f
    d = e + f
    b = 2 * ak / (3 - wo * 3 * gc)
    delta = np.exp(ak * taup) * (1 + b) ** 2 - np.exp(-ak * taup) * \
        (1 - b) ** 2
    ww = wo / 4
    ss = us / (1 - ak2 * us * us)
    q1 = 2 + 3 * us + (1 - wo) * 3 * gc * us * (1 + 2 * us)
    q2 = 2 - 3 * us - (1 - wo) * 3 * gc * us * (1 - 2 * us)
    q3 = q2 * np.exp(-taup / us)
    c1 = ((ww * ss) / delta) * (q1 * np.exp(ak * taup) * (1 + b) + q3 *
                                (1 - b))
    c2 = -((ww * ss) / delta) * (q1 * np.exp(-ak * taup) * (1 - b) + q3 *
                                 (1 + b))
    cp1 = c1 * ak / (3 - wo * 3 * gc)
    cp2 = -c2 * ak / (3 - wo * 3 * gc)
    z = d - wo * 3 * gc * uv * dp + wo * aer_phase / 4
    x = c1 - wo * 3 * gc * uv * cp1
    y = c2 - wo * 3 * gc * uv * cp2
    aa1 = uv / (1 + ak * uv)
    aa2 = uv / (1 - ak * uv)
    aa3 = us * uv / (us + uv)

    aer_ref1 = x * aa1 * (1 - np.exp(-taup / aa1))  # Eq 13
    aer_ref2 = y * aa2 * (1 - np.exp(-taup / aa2))
    aer_ref3 = z * aa3 * (1 - np.exp(-taup / aa3))

    aer_ref = (aer_ref1 + aer_ref2 + aer_ref3) / (us * uv)

    # Residual rayleigh (not in the paper)
    Res_ray = Resr1 + Resr2 * taur * ray_phase / (us * uv) + Resr3 * \
        ((taur * ray_phase / (us * uv)) ** 2)

    # Residual aerosol
    Res_aer = (Resa1 + Resa2 * (taup * m * cksi) + Resa3 *
               ((taup * m * cksi) ** 2)) + Resa4 * (taup*m*cksi) ** 3

    # Term coupling molecule / aerosol
    tautot = taup + taurz

    Res_6s = (Rest1 + Rest2 * (tautot * m * cksi) + Rest3 *
              ((tautot * m * cksi) ** 2)) + Rest4 * ((tautot * m * cksi) ** 3)

    # Total atmospheric reflectance
    atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s

    # For non-lambertian surface
    tdir_tts = np.exp(-tautot / us)
    tdir_ttv = np.exp(-tautot / uv)
    tdif_tts = ttetas - tdir_tts
    tdif_ttv = ttetav - tdir_ttv

    atm_opts = AtmosphericOptics(ttetas, ttetav, tg, s, atm_ref, tdir_tts,
                                 tdif_tts, tdir_ttv, tdif_ttv)

    return atm_opts


class AtmosphericOptics:
    """
    Class to hold atmospheric optics results from the SMAC model.

    Parameters
    ----------
    Ta_s : np.array
        Total scattering transmission downard
    Ta_o : np.array
        Total scattering transmission upward
    Tg : np.array
        Transmittance for all gases
    Ra_dd : np.array
        Hemispherical atmospheric reflectance for diffuse light
    Ra_so : np.array
        Directional atmospheric reflectance for direct incidence
    Ta_ss : np.array
        Directional transmittance for direct incidence
    Ta_sd : np.array
        Hemispherical transmittance for direct incidence
    Ta_oo : np.array
        Directional transmittance for direct incidence (in viewing direction)
    Ta_do : np.array
        Hemispherical transmittance for direct incidence (in viewing direction)

    Attributes
    ----------
    Ta_s : np.array
        Total scattering transmission downard
    Ta_o : np.array
        Total scattering transmission upward
    Tg : np.array
        Transmittance for all gases
    Ra_dd : np.array
        Hemispherical atmospheric reflectance for diffuse light
    Ra_so : np.array
        Directional atmospheric reflectance for direct incidence
    Ta_ss : np.array
        Directional transmittance for direct incidence
    Ta_sd : np.array
        Hemispherical transmittance for direct incidence
    Ta_oo : np.array
        Directional transmittance for direct incidence (in viewing direction)
    Ta_do : np.array
        Hemispherical transmittance for direct incidence (in viewing direction)
    """
    def __init__(self, Ta_s, Ta_o, Tg, Ra_dd, Ra_so, Ta_ss, Ta_sd, Ta_oo,
                 Ta_do):
        self.Ta_s = Ta_s
        self.Ta_o = Ta_o
        self.Tg = Tg
        self.Ra_dd = Ra_dd
        self.Ra_so = Ra_so
        self.Ta_ss = Ta_ss
        self.Ta_sd = Ta_sd
        self.Ta_oo = Ta_oo
        self.Ta_do = Ta_do


class AtmosphericProperties:
    """
    Class to hold the properties of the atmosphere used in SMAC calculations.

    Parameters
    ----------
    aot550 : float
        Aerosol optical thickness at 550 nm
    uo3 : float
        Ozone content, cm-atm
    uh2o : float
        Water vapour content, g cm^-2
    Pa : float, optional
        Air pressure, hPa, defaults to 1013.25
    alt_m : int, optional
        Altitude of observation site, used to calculate air pressure if air
        pressure not known.
    temp_k : float, optional
        Temperature in kelvin used to estimate air pressure

    Attributes
    ----------
    aot550 : float
        Aerosol optical thickness at 550 nm
    uO3 : float
        Ozone content, cm-atm
    uh20 : float
        Water vapour content, g cm^-2
    Pa : float
        Air pressure, hPa
    """
    def __init__(self, aot550, uo3, uh2o, Pa=None, alt_m=None, temp_k=None):
        self.aot550 = aot550
        self.uo3 = uo3
        self.uh2o = uh2o
        if isinstance(Pa, type(None)):
            if not isinstance(alt_m, type(None)) and not isinstance(
                    temp_k, type(None)):
                self.Pa = _calculate_pressure_from_altitude(alt_m, temp_k)
            else:
                self.Pa = 1013.25
        else:
            self.Pa = Pa


def _calculate_pressure_from_altitude(alt_m, temp_k):
    # Caluclate pressure from altitude and temperature

    g = 9.80665
    M = 0.02896968
    R0 = 8.314462618
    Pa0 = 1013.25

    Pa = Pa0 * np.exp(-(g * alt_m * M / (temp_k * R0)))

    return Pa
