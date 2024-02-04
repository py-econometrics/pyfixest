from pyfixest.estimation import feols, fepois
from pyfixest.utils import get_data
from pyfixest.summarize import summary, etable


def test_summary():
    """Just run etable() and summary() on a few models."""

    df1 = get_data()
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