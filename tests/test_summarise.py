import pandas as pd

from pyfixest.estimation import feols, fepois
from pyfixest.summarize import etable, summary
from pyfixest.utils import get_data


def test_summary():
    """Just run etable() and summary() on a few models."""
    df1 = get_data()
    df1 = pd.concat(
        [df1, df1], axis=0
    )  # Make it a bit larger, for examining the thousands separator
    df2 = get_data(model="Fepois")

    fit1 = feols("Y ~ X1 + X2 | f1", data=df1)
    fit2 = fepois("Y ~ X1 + X2 + f2 | f1", data=df2, vcov={"CRV1": "f1+f2"})
    fit3 = feols("Y ~ X1", data=df1)

    summary(fit1)
    summary(fit2)
    summary([fit1, fit2])

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
    etable([fit1, fit2], signif_code=None)

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
