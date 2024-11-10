import duckdb

import numpy as np

import pyfixest as pf
from pyfixest.utils.utils import get_data


def test_api():
    df1 = get_data()
    df2 = get_data(model="Fepois")

    fit1 = pf.feols("Y ~ X1 + X2 | f1", data=df1)
    fit2 = pf.estimation.fepois(
        "Y ~ X1 + X2 + f2 | f1", data=df2, vcov={"CRV1": "f1+f2"}
    )
    pf.summary(fit1)
    pf.report.summary(fit2)
    pf.etable([fit1, fit2])
    pf.coefplot([fit1, fit2])


def test_feols_args():
    """
    Check feols function arguments.

    Arguments to check:
    - copy_data
    - store_data
    - fixef_tol
    - solver
    """
    df = pf.get_data()

    fit1 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, copy_data=True)
    fit2 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, copy_data=False)

    assert (fit1.coef() == fit2.coef()).all()

    fit3 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, store_data=False, fixef_tol=1e-02)
    if hasattr(fit3, "_data"):
        raise AttributeError(
            "The 'fit3' object has the attribute '_data', which should not be present."
        )

    assert fit1.coef().xs("X1") != fit3.coef().xs("X1")
    assert np.abs(fit1.coef().xs("X1") - fit3.coef().xs("X1")) < 0.01

    fit4 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, solver="np.linalg.solve")
    fit5 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, solver="np.linalg.lstsq")

    assert (fit4.coef() == fit5.coef()).all()


def test_fepois_args():
    """
    Check feols function arguments.

    Arguments to check:
    - copy_data
    - store_data
    - fixef_tol
    - solver
    """
    df = pf.get_data(model="Fepois")

    fit1 = pf.fepois(fml="Y ~ X1 | f1 + f2", data=df, copy_data=True)
    fit2 = pf.fepois(fml="Y ~ X1 | f1 + f2", data=df, copy_data=False)

    assert (fit1.coef() == fit2.coef()).all()

    fit3 = pf.fepois(fml="Y ~ X1 | f1 + f2", data=df, store_data=False, fixef_tol=1e-02)
    if hasattr(fit3, "_data"):
        raise AttributeError(
            "The 'fit3' object has the attribute '_data', which should not be present."
        )

    assert fit1.coef().xs("X1") != fit3.coef().xs("X1")
    assert np.abs(fit1.coef().xs("X1") - fit3.coef().xs("X1")) < 0.01

    fit4 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, solver="np.linalg.solve")
    fit5 = pf.feols(fml="Y ~ X1 | f1 + f2", data=df, solver="np.linalg.lstsq")

    np.testing.assert_allclose(fit4.coef(), fit5.coef(), rtol=1e-12)


def test_lean():
    data = pf.get_data()
    fit = pf.feols("Y ~ X1 + X2 | f1", data=data, lean=True)

    assert not hasattr(fit, "_data")
    assert not hasattr(fit, "_X")
    assert not hasattr(fit, "_Y")


def test_duckdb_input():
    data_pandas = pf.get_data()
    data_duckdb = duckdb.query("SELECT * FROM data_pandas")
    fit_pandas = pf.feols("Y ~ X1 | f1 + f2", data=data_pandas)
    fit_duckdb = pf.feols("Y ~ X1 | f1 + f2", data=data_duckdb)
    assert type(fit_pandas) == type(fit_duckdb)
    assert fit_pandas.coef() == fit_duckdb.coef()
    assert fit_pandas.se() == fit_duckdb.se()
    np.testing.assert_allclose(fit_pandas.coef(), fit_duckdb.coef(), rtol=1e-12)
    np.testing.assert_allclose(fit_pandas.se(), fit_duckdb.se(), rtol=1e-12)
