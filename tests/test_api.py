import pyfixest as pf
import numpy as np
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
    """

    df = pf.get_data()

    fit1 = pf.feols(
        fml = "Y ~ X1 | f1 + f2",
        data = df,
        copy_data = True
    )
    fit2 = pf.feols(
        fml = "Y ~ X1 | f1 + f2",
        data = df,
        copy_data = False
    )

    assert (fit1.coef() == fit2.coef()).all()

    fit3 = pf.feols(
        fml = "Y ~ X1 | f1 + f2",
        data = df,
        store_data = False,
        fixef_tol = 1e-02
    )
    assert fit3._data is None

    assert fit1.coef().xs("X1") != fit3.coef().xs("X1")
    assert np.abs(fit1.coef().xs("X1") - fit3.coef().xs("X1")) < 0.01


def test_fepois_args():

    """
    Check feols function arguments.

    Arguments to check:
    - copy_data
    - store_data
    - fixef_tol
    """

    df = pf.get_data(model = "Fepois")

    fit1 = pf.fepois(
        fml = "Y ~ X1 | f1 + f2",
        data = df,
        copy_data = True
    )
    fit2 = pf.fepois(
        fml = "Y ~ X1 | f1 + f2",
        data = df,
        copy_data = False
    )

    assert (fit1.coef() == fit2.coef()).all()

    fit3 = pf.fepois(
        fml = "Y ~ X1 | f1 + f2",
        data = df,
        store_data = False,
        fixef_tol = 1e-02
    )
    assert fit3._data is None

    assert fit1.coef().xs("X1") != fit3.coef().xs("X1")
    assert np.abs(fit1.coef().xs("X1") - fit3.coef().xs("X1")) < 0.01






