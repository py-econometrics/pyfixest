import pandas as pd
import pytest
import statsmodels.formula.api as smf

import pyfixest as pf
from pyfixest.estimation.estimation import feols, fepois
from pyfixest.report.summarize import _select_order_coefs, etable, summary
from pyfixest.utils.utils import get_data


def test_summary():
    """Just run etable() and summary() on a few models."""
    df1 = get_data()
    df1 = pd.concat(
        [df1, df1], axis=0
    )  # Make it a bit larger, for examining the thousands separator
    df2 = get_data(model="Fepois")

    fit1 = feols("Y ~ X1 + X2 | f1", data=df1)
    fit1a = feols("Y ~ X1 + X2 + f1", data=df1)
    fit2 = fepois("Y ~ X1 + X2 + f2 | f1", data=df2, vcov={"CRV1": "f1+f2"})
    fit3 = feols("Y ~ X1", data=df1)
    fit4 = feols("Y ~ X1", data=df1, weights="weights")
    fit5 = feols("Y ~ 1 | Z1 ~ X1", data=df1)

    fit_qreg = pf.quantreg("Y ~ X1", data=df1, vcov="nid")

    summary(fit1)
    summary(fit2)
    summary([fit1, fit2])
    summary([fit4])
    fit5.summary()

    etable(fit1)
    etable(fit2)
    etable([fit1, fit2])

    etable([fit3])
    etable([fit1, fit2, fit3])

    fit_iv = feols("Y ~ X2 | f1 | X1 ~ Z1", data=df1)
    etable([fit_iv, fit1])

    fit_multi = feols("Y + Y2 ~ X1 + X2 | f1", data=df1)
    etable(fit_multi.to_list())

    # Test significance code
    etable([fit1, fit2], signif_code=[0.01, 0.05, 0.1])
    etable([fit1, fit2], signif_code=[0.02, 0.06, 0.1])

    # Test coefficient format
    etable([fit1, fit2], coef_fmt="b (se)\nt [p]")

    # Test custom statistics
    etable(
        models=[fit1, fit2],
        custom_stats={
            "conf_int_lb": [fit1._conf_int[0], fit2._conf_int[0]],
            "conf_int_ub": [fit1._conf_int[1], fit2._conf_int[1]],
        },
        coef_fmt="b [conf_int_lb, conf_int_ub]",
    )

    # Test scientific notation
    etable(
        models=[fit1],
        custom_stats={
            "test_digits": [[0.1, 12300]],
        },
        coef_fmt="b [test_digits]",
        digits=2,
    )

    # Test scientific notation, thousands separator
    etable(
        models=[fit1],
        custom_stats={
            "test_digits": [[0.1, 12300]],
        },
        coef_fmt="b [test_digits]",
        digits=2,
        scientific_notation=False,
        thousands_sep=True,
    )

    # Test select / order coefficients
    etable([fit1, fit2, fit3], coef_fmt="b (se)\nt [p]")
    etable([fit1, fit2, fit3], coef_fmt="b (se)\nt [p]", keep=["X1", "cep"])
    etable([fit1, fit2, fit3], coef_fmt="b (se)\nt [p]", drop=[r"\d$"])
    etable([fit1, fit2, fit3], coef_fmt="b (se)\nt [p]", keep=[r"\d"], drop=["f"])
    etable([fit1, fit2, fit3], coef_fmt="b (se)\nt [p]", keep="X")
    etable([fit1, fit2, fit3], coef_fmt="b (se)\nt [p]", drop=r"\d$")

    # test labels, felabels args
    etable([fit1, fit1a], labels={"X1": "X1_label"}, felabels={"f1": "f1_label"})
    etable(
        [fit1, fit1a], labels={"X1": "X1_label"}, felabels={"f1": "f1_label"}, keep="X1"
    )
    etable(
        [fit1, fit1a], labels={"X1": "X1_label"}, felabels={"f1": "f1_label"}, drop="X1"
    )
    etable([fit1, fit1a], felabels={"f1": "f1_renamed2"}, keep=["f1"])

    cols = ["x1", "x2", "x11", "x21"]
    assert _select_order_coefs(cols, keep=["x1"]) == ["x1", "x11"]
    assert _select_order_coefs(cols, drop=["x1"]) == ["x2", "x21"]
    assert _select_order_coefs(cols, keep=["x1"], exact_match=True) == ["x1"]
    assert _select_order_coefs(cols, drop=["x1"], exact_match=True) == [
        "x2",
        "x11",
        "x21",
    ]

    # API tests for new tex args

    etable([fit1, fit2], type="tex")
    etable([fit1, fit2], type="tex", print_tex=True)

    etable([fit1, fit2], type="tex", notes="You can add notes here.")
    etable([fit1, fit2], type="md", notes="You can add notes here.")

    etable([fit1, fit2], type="tex", model_heads=["Model 1", "Model 2"])
    etable(
        [fit1, fit2], type="tex", model_heads=["Model 1", "Model 2"], head_order="dh"
    )
    etable(
        [fit1, fit2], type="tex", model_heads=["Model 1", "Model 2"], head_order="hd"
    )
    etable([fit1, fit2], type="tex", model_heads=["Model 1", "Model 2"], head_order="d")
    etable([fit1, fit2], type="tex", model_heads=["Model 1", "Model 2"], head_order="h")
    etable([fit1, fit2], type="tex", model_heads=["Model 1", "Model 2"], head_order="")
    etable([fit1, fit2], type="tex", filename="tests/texfiles/test.tex")

    summary(fit_qreg)
    etable(fit_qreg)


@pytest.mark.skip("Pyfixest PR is not yet merged into stargazer.")
def test_stargazer():
    data = pf.get_data()

    fit = pf.feols("Y ~ X1", data=data)
    fit_smf = smf.ols("Y ~ X1", data=data).fit()

    pf.Stargazer([fit, fit_smf])
