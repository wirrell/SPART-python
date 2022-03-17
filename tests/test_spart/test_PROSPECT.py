"""
Tests for the PROSPECT_5D model.
"""
import itertools
import progressbar
import pytest
import numpy as np
import pandas as pd
import SPART
from SPART.prospect_5d import LeafBiology, PROSPECT_5D


def build_PROSPECT_5D_test_cases():
    value_combinations =  leaf_value_combinations()
    input_columns = ["Cab", "Cdm", "Cw", "Cs", "Cca", "Cant", "N"]
    refl_columns = [f"refl_{x}" for x in range(400, 2401)]
    tran_columns = [f"tran_{x}" for x in range(400, 2401)]
    kchl_columns = [f"kChlrel_{x}" for x in range(400, 2401)]
    all_columns = input_columns + refl_columns + tran_columns + kchl_columns
    test_cases = pd.DataFrame(columns=all_columns)

    optical_params = SPART.load_optical_parameters()

    for num, prod in progressbar.progressbar(enumerate(value_combinations)):
        leaf_biology = LeafBiology(*prod)
        result = PROSPECT_5D(leaf_biology, optical_params)
        test_cases.loc[-1] = np.concatenate([prod, result.refl, result.tran, result.kChlrel])
        test_cases.index = test_cases.index + 1
        test_cases = test_cases.sort_index()
    test_cases.to_csv('PROSPECT_5D_test_cases.csv')


def leaf_value_combinations():
    # For reference of ranges, see: https://doi.org/10.1016/j.rse.2020.111870
    Cab = np.arange(10, 85, 20)
    Cdm = np.arange(0.005, 0.025, 0.01)
    Cw = np.arange(0.02, 0.12, 0.04)
    Cs = np.arange(0, 1.5, 0.5)
    Cca = np.arange(10, 35, 10)
    Cant = np.arange(10, 35, 10)
    N = np.arange(1.0, 3.5, 0.5)

    value_combinations = itertools.product(Cab, Cdm, Cw, Cs, Cca, Cant, N)

    return value_combinations


def test_PROSPECT_5D(leaf_biology, optical_params):
    result = PROSPECT_5D(leaf_biology, optical_params)


@pytest.fixture(params=leaf_value_combinations())
def leaf_biology(request):
    return LeafBiology(*request.param)


if __name__ == "__main__":
    # Do initial build for test cases
    build_PROSPECT_5D_test_cases()
