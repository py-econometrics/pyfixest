import pandas as pd
import numpy as np
import pyfixest as pf
import pytest
from pyfixest.estimation.model_matrix_fixest_ import _get_na_index

def test_fweights():

    data = pf.get_data()
    data2_w = data[["Y", "X1"]].groupby(["Y", "X1"]).size().reset_index().rename(columns = {0:"count"})
    data3_w = data[["Y", "X1", "f1"]].groupby(["Y", "X1", "f1"]).size().reset_index().rename(columns = {0:"count"})


    fit1 = pf.feols("Y ~ X1", data = data)
    fit2 = pf.feols("Y ~ X1", data = data2_w, weights = "count", weights_type = "fweights")

    np.testing.assert_allclose(fit1.tidy().values,  fit2.tidy().values)

    fit3 = pf.feols("Y ~ X1 | f1", data = data)
    fit4 = pf.feols("Y ~ X1 | f1", data = data3_w, weights = "count", weights_type = "fweights")

    np.testing.assert_allclose(fit3.tidy().values,  fit4.tidy().values)

def test_aweights():

    data = pf.get_data()
    data["weights"] = np.ones(data.shape[0])

    fit1 = pf.feols("Y ~ X1", data = data)
    fit2 = pf.feols("Y ~ X1", data = data, weights_type = "aweights")
    fit3 = pf.feols("Y ~ X1", data = data, weights = "weights", weights_type = "aweights")

    np.testing.assert_allclose(fit1.tidy().values, fit2.tidy().values)
    np.testing.assert_allclose(fit1.tidy().values, fit3.tidy().values)


