"""
Soil-Plant-Atmosphere Radiative Transfer model
for top-of-canopy and top-of-atmosphere reflectance 

Python port coded by George Worrall (gworrall@ufl.edu)
Center for Remote Sensing, University of Florida

Ported to Python from the original matlab code and model developed by:

Peiqi Yang               (p.yang@utwente.nl)
Christiaan van der Tol   (c.vandertol@utwente.nl)
Wout Verhoef             (w.verhoef@utwente.nl)

University of Twente
Faculty of Geo-Information Science and Earth Observation (ITC), 
Department of Water Resources

main inputs
soil properties: 
soilpar.B         =   soil brightness [0,1] unitless
soilpar.lat       =   soil spectral shape parameters [25-45] unitless
soilpar.lon       =   soil spectral shape parameters [30-60] unitless
soilpar.SMp       =   soil mositure in percentage [5-55] unitless

leaf (bio)physical properties:
leafbio.Cab       =   Chlorophyll content [0- 80], ug cm-2 
leafbio.Cdm       =   leaf mass per unit area [0-0.02] g cm-2
leafbio.Cw        =   Equivalent water thickness [0-0.1] cm
leafbio.Cca       =   Caratenoid content [0-30] ug cm-2
leafbio.Cs        =   brown pigments [0-1] unitless, 0=green 1=brown
leafbio.Cant      =   Anthocyanin content [0-30] ug cm-2
leafbio.N         =   leaf structure parameter [0-3] related to thickness

canopy structural properties
canopy.LAI        =   Leaf area index [0,7]  m m-2
canopy.LIDFa      =   leaf inclination parameter a [-1,1]
canopy.LIDFb      =   parameter b [-1,1], abs(a)+abs(b)<=1
canopy.hot        =   hot spot parameters, [0-0.2] leaf width/canopy height

atmopsheric properties
atm.Pa            =   Air pressure [500-1300] hPa/estimated from elevation
atm.aot550        =   aerosol optical thickness at 550nm [0-2] unitless
atm.uo3           =   Ozone content [0-0.8] cm-atm or [0-0.0171] kg m-2
atm.uh2o          =   Water vapour  [0-8.5] g cm-2 or [0-85] kg m-2
atm_alt_m         =   groudn elevation in meters in case no Pa available
atm.Ea            =   extraterrestrial irradiance spectral

sun-observer geometry
angles.tts        =   solar zenith angle
angles.tto        =   viewing zenith angle
angles.psi        =   relative azumith angle between the sun and viewer

 main ouputs
R_TOC     Top-of-Canopy reflectance
R_TOA     Top-of-Atmopshere reflectance
L_TOA     Top-of-Atmopshere radiance


main functions 
- BSM (Brightness-shape-mositure)
- PROSAIL
- SAIL (SAILH, with hotspot effects)
- SMAC (modified SMAC model)

Coupling BSM, PROSAIL and SMAC to simulate TOA reflectance 
"""
import pickle
import scipy.io
from os import path
from PROSPECT_5D import PROSPECT_5D, LeafBiology


def run_SPART(soilpar, leafbio, canopy, atm, angles, spectral, optipar):
    """
    Run the SPART model.

    Parameters
    ----------
    pass
    """



def load_optical_parameters():
    # Load optical parameters from saved arrays
    params_file = path.join(path.dirname(__file__),
                            'model_parameters/optical_params.pkl')

    with open(params_file, 'rb') as f:
        params = pickle.load(f)

    return params
