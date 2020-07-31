"""
SPART-python

PROSPECT 5D model.

Feret et al. - PROSPECT-D: Towards modeling leaf optical properties
    through a complete lifecycle
"""
import pickle
import numpy as np
import scipy.io
import pandas as pd
from os import path

# TODO: write optipar read in, see set_input_from_excel.

class LeafBiology():
    """
    Class to hold leaf biology variables.

    Parameters
    ----------
    Cab : float
        Chlorophyll concentration, micro g / cm ^ 2
    Cca : float
        Carotenoid concentration, micro g / cm ^ 2
    Cdm : float
        Leaf mass per unit area, g / cm ^ 2
    Cs : float
        Brown pigments (from SPART paper, unitless)
    Cant : float
        Anthocyanin content, micro g / cm ^ 2
    N : float
        Leaf structure parameter. Unitless.

    Attributes
    ----------
    Cab : float
        Chlorophyll concentration, micro g / cm ^ 2
    Cca : float
        Carotenoid concentration, micro g / cm ^ 2
    Cdm : float
        Leaf mass per unit area, g / cm ^ 2
    Cs : float
        Brown pigments (from SPART paper, unitless)
    Cant : float
        Anthocyanin content, micro g / cm ^ 2
    N : float
        Leaf structure parameter. Unitless.
    """

    def __init__(Cab, Cca, Cdm, Cs, Cant, N):
        self.Cab = Cab
        self.Cca = Cca
        self.Cdm = Cdm
        self.Cs = Cs
        self.Cant = Cant
        self.N = N


def _load_optical_parameters():
    # Load optical parameters from saved arrays
    params_file = path.join(path.dirname(__file__), 'optical_params.pkl')

    with open(params_file, 'rb') as f:
        params = pickle.load(f)

    return params


def PROSPECT_5D(leafbio):
    """
    PROSPECT_5D model.

    Parameters
    ----------
    leafbio : LeafBiology
        Object holding user specified leaf biology model parameters.
    """

    # Leaf parameters
    Cab = leafbio.Cab
    Cca = leafbio.Cca
    Cdm = leafbio.Cdm
    Cs = leafbio.Cs
    Cant = leafbio.Cant
    N = leafbio.N

    # Model constants
    optical_params = _load_optical_parameters
    nr = optical_params['nr']
    Kdm = optical_params['Kdm']
    Kab = optical_params['Kab']
    Kca = optical_params['Kca']
    Kw = optical_params['Kw']
    Ks = optical_params['Ks']
    Kant = optical_params['Kant']

    # TODO: continue from line 55 in eponemous .m file
    pass
