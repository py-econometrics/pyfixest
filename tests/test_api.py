import duckdb
import numpy as np
import pandas as pd
import pytest
from formulaic.errors import FactorEvaluationError

import pyfixest as pf
from pyfixest.utils.utils import get_data


def test_api():
    df1 = get_data()
    df2 = get_data(model="Fepois")

    fit1 = pf.feols("Y ~ X1 + X2 | f1", data=df1)
    fit2 = pf.estimation.fepois(
        "Y ~ X1 + X2 + f2 | f1", data=df2, vcov={"CRV1": "f1+f2"}
    )
    fit_multi = pf.feols("Y + Y2 ~ X1", data=df2)

    pf.summary(fit1)
    pf.report.summary(fit2)
    pf.etable([fit1, fit2])
    pf.coefplot([fit1, fit2])

    pf.summary(fit_multi)
    pf.etable(fit_multi)
    pf.coefplot(fit_multi)


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

    assert np.allclose(fit4.coef().values, fit5.coef().values)


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
    assert type(fit_pandas) is type(fit_duckdb)
    np.testing.assert_allclose(fit_pandas.coef(), fit_duckdb.coef(), rtol=1e-12)
    np.testing.assert_allclose(fit_pandas.se(), fit_duckdb.se(), rtol=1e-12)


def _lspline(series: pd.Series, knots: list[float]) -> np.array:
    """Generate a linear spline design matrix for the input series based on knots."""
    vector = series.values
    columns = []

    for i, knot in enumerate(knots):
        column = np.minimum(vector, knot if i == 0 else knot - knots[i - 1])
        columns.append(column)
        vector = vector - column

    # Add the remainder as the last column
    columns.append(vector)

    # Combine columns into a design matrix
    return np.column_stack(columns)


@pytest.fixture
def spline_data():
    """Fixture to prepare data with spline splits."""
    data = pf.get_data()
    data["Y"] = np.where(data["Y"] > data["Y"].median(), 1, 0)
    spline_split = _lspline(data["X2"], [0, 1])
    data["X2_0"], data["0_X2_1"], data["1_X2"] = spline_split.T
    return data


@pytest.mark.parametrize(
    "method,family",
    [
        ("feols", None),
        ("feglm", "logit"),
        ("feglm", "probit"),
        ("feglm", "gaussian"),
    ],
)
def test_context_capture(spline_data, method, family):
    method_kwargs = {"data": spline_data}
    if family:
        method_kwargs["family"] = family

    explicit_fit = getattr(pf, method)("Y ~ X2_0 + 0_X2_1 + 1_X2", **method_kwargs)
    context_captured_fit = getattr(pf, method)(
        "Y ~ _lspline(X2,[0,1])", context=0, **method_kwargs
    )
    context_captured_fit_map = getattr(pf, method)(
        "Y ~ _lspline(X2,[0,1])", context={"_lspline": _lspline}, **method_kwargs
    )

    for context_fit in [context_captured_fit, context_captured_fit_map]:
        np.testing.assert_allclose(context_fit.coef(), explicit_fit.coef(), rtol=1e-12)
        np.testing.assert_allclose(context_fit.se(), explicit_fit.se(), rtol=1e-12)

    # FactorEvaluationError for `feols` when context is not set
    if method == "feols":
        with pytest.raises(
            FactorEvaluationError, match="Unable to evaluate factor `_lspline"
        ):
            pf.feols("Y ~ _lspline(X2,[0,1]) | f1 + f2", data=spline_data)
